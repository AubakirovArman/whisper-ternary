"""Обучение QAT на двух картах (DDP) с оптимизациями под H200.

Отличия от однокарточного ``qat_train.py``
------------------------------------------
* **DDP на две карты.**  Запуск через ``torchrun --nproc_per_node=2``.
  Эффективный батч равен ``--batch`` × число карт, поэтому для сохранения
  прежнего рецепта (батч 64) ставится ``--batch 32``.
* **Калибровка пропускается при ``--init-from``.**  Активационные моменты нужны
  только для инициализации масштабов, а чекпоинт их всё равно перезаписывает —
  считать их заново значит терять минуту на пустом месте.
* **Оптимизации, померенные на этой же машине** (изолированный стенд, шум ±6%)::

      fused AdamW                     −13% времени
      torch.compile(dynamic=True)     −16% времени, −13% памяти
      вместе                          −26% времени
      TF32                            0% — считаем под autocast(bf16),
                                      fp32-умножений в горячем пути нет

  ``--compile`` по умолчанию выключен: компиляция стоит около 6 минут на старте
  и взаимодействует с DDP, поэтому включать её осознанно.

Что сохранено без изменений
---------------------------
Функция потерь побитово та же, что в ``wal_tat.qat.distill``: KL с
``log_target=True`` и множителем ``temperature**2``, CE только по позициям с
меткой, feature-matching как MSE между выходами блоков.  Расписание —
косинус с прогревом.  Группы learning rate те же: латентные веса на базовом lr,
масштабы на ``scale_lr_mult``, остальное на ``fp_lr_mult``.

Ранг 0 делает всё, что пишет на диск: журнал, чекпоинты, валидацию.  Остальные
ранги только считают градиенты.
"""

import os as _os
import pathlib as _pl
# корень репозитория: от расположения файла, можно переопределить WALTAT_ROOT
_REPO = _pl.Path(_os.environ.get('WALTAT_ROOT',
                                 _pl.Path(__file__).resolve().parents[2]))
_REPO_STR = str(_REPO)

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path(f"{_REPO_STR}/wal_tat")
sys.path.insert(0, str(ROOT / "src"))

from wal_tat.qat import (  # noqa: E402
    WhisperFeatureShards,
    convert_model_to_qat,
    load_qat_checkpoint,
    qat_modules,
    save_qat_checkpoint,
)
from wal_tat.qat.distill import (  # noqa: E402
    cosine_lr_multiplier,
    shift_labels_right,
    transformer_block_modules,
)

IGNORE_INDEX = -100


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--precision", default="t3", choices=("t3", "b1", "fp"))
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--init-from", type=Path, required=True,
                   help="чекпоинт, с которого продолжаем; калибровка при этом не нужна")
    p.add_argument("--data", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch", type=int, default=32,
                   help="НА КАРТУ; эффективный батч умножается на число карт")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--scale-lr-mult", type=float, default=0.1)
    p.add_argument("--fp-lr-mult", type=float, default=0.25)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--min-lr-ratio", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--feature-weight", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--compile", action="store_true",
                   help="torch.compile(dynamic=True): −16%% времени, но ~6 мин на старте")
    p.add_argument("--no-fused-adam", action="store_true")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=2500)
    p.add_argument("--valid-every", type=int, default=2500)
    p.add_argument("--valid-batches", type=int, default=16)
    return p.parse_args(argv)


def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def log(message: str) -> None:
    if is_main():
        print(message, flush=True)


class FeatureTap:
    """Выходы блоков трансформера, снятые хуками — для feature-matching."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.outputs: Dict[str, torch.Tensor] = {}
        self._handles = []
        for name, module in transformer_block_modules(model).items():
            self._handles.append(
                module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            self.outputs[name] = output[0] if isinstance(output, tuple) else output
        return hook

    def clear(self) -> None:
        self.outputs.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


def feature_mse(student: Dict[str, torch.Tensor],
                teacher: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    shared = [k for k in student if k in teacher]
    if not shared:
        return None
    total = None
    for key in shared:
        a, b = student[key].float(), teacher[key].float()
        if a.shape != b.shape:
            continue
        term = F.mse_loss(a, b)
        total = term if total is None else total + term
    return None if total is None else total / max(len(shared), 1)


def parameter_groups(model: torch.nn.Module, args) -> List[dict]:
    """Те же три группы, что в однокарточном рецепте."""
    from wal_tat.qat.quant import QuantLinear

    latent, scales, seen = [], [], set()
    for module in model.modules():
        if not isinstance(module, QuantLinear):
            continue
        if module.weight.requires_grad and id(module.weight) not in seen:
            latent.append(module.weight)
            seen.add(id(module.weight))
        if module.group_scale is not None and module.group_scale.requires_grad \
                and id(module.group_scale) not in seen:
            scales.append(module.group_scale)
            seen.add(id(module.group_scale))
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    groups = []
    if latent:
        groups.append({"name": "latent", "params": latent, "lr": args.lr})
    if scales:
        groups.append({"name": "scale", "params": scales,
                       "lr": args.lr * args.scale_lr_mult})
    if rest:
        groups.append({"name": "fp16", "params": rest,
                       "lr": args.lr * args.fp_lr_mult})
    return groups


def main(argv=None) -> int:
    args = parse_args(argv)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + local_rank)

    out = args.out.expanduser().resolve()
    if is_main():
        if out.exists() and any(out.iterdir()) and not args.overwrite:
            raise SystemExit(f"{out} не пуст; передайте --overwrite")
        out.mkdir(parents=True, exist_ok=True)

    log(f"[ddp] карт: {world}, батч на карту {args.batch}, "
        f"эффективный батч {args.batch * world}")

    # -- модели ------------------------------------------------------------ #
    processor = WhisperProcessor.from_pretrained(args.model)
    teacher = (WhisperForConditionalGeneration
               .from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval())
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student = WhisperForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32).to(device)

    # Калибровка не нужна: масштабы приходят из чекпоинта и перезаписывают
    # любую инициализацию.
    qat_manifest = convert_model_to_qat(student, precision=args.precision,
                                        group_size=args.group_size,
                                        learnable_scales=True)
    manifest, extra = load_qat_checkpoint(student, args.init_from)
    log(f"[init] веса из {args.init_from} (шаг {extra.get('step', '?')}), "
        f"матриц {len(qat_modules(student))}, precision {manifest.precision}")

    student_tap = FeatureTap(student) if args.feature_weight > 0 else None
    teacher_tap = FeatureTap(teacher) if args.feature_weight > 0 else None

    model = student
    if world > 1:
        model = DistributedDataParallel(student, device_ids=[local_rank],
                                        find_unused_parameters=False)
    if args.compile:
        model = torch.compile(model, dynamic=True)
        log("[opt] torch.compile(dynamic=True) включён")

    # -- данные ------------------------------------------------------------ #
    parts = [WhisperFeatureShards(path, subset="train") for path in args.data]
    train_set = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    hours = sum(getattr(p, "total_hours", 0.0) for p in parts)
    log(f"[data] окон {len(train_set):,}, часов {hours:.0f}")
    collator = parts[0].default_collator()   # это метод-фабрика, а не сам коллатор
    sampler = DistributedSampler(train_set, shuffle=True, seed=args.seed,
                                 drop_last=True) if world > 1 else None
    loader = DataLoader(train_set, batch_size=args.batch, sampler=sampler,
                        shuffle=(sampler is None), num_workers=args.workers,
                        pin_memory=True, drop_last=True, collate_fn=collator)

    # -- оптимизатор -------------------------------------------------------- #
    fused = not args.no_fused_adam
    optimizer = torch.optim.AdamW(parameter_groups(student, args),
                                  betas=(0.9, 0.95), weight_decay=0.0, fused=fused)
    log(f"[opt] AdamW fused={fused}, групп {len(optimizer.param_groups)}")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: cosine_lr_multiplier(s, total_steps=args.steps,
                                   warmup_steps=args.warmup,
                                   min_ratio=args.min_lr_ratio))

    start_id = student.config.decoder_start_token_id
    pad_id = student.config.pad_token_id
    step, epoch, t0 = 0, 0, time.time()
    history: List[dict] = []

    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
        for batch in loader:
            if step >= args.steps:
                break
            features = batch["input_features"].to(device, torch.float32, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            decoder_in = shift_labels_right(labels,
                                            decoder_start_token_id=start_id,
                                            pad_token_id=pad_id)
            if student_tap is not None:
                student_tap.clear()
                teacher_tap.clear()

            with torch.autocast("cuda", torch.bfloat16):
                with torch.no_grad():
                    teacher_logits = teacher(input_features=features.to(torch.bfloat16),
                                             decoder_input_ids=decoder_in).logits
                student_logits = model(input_features=features,
                                       decoder_input_ids=decoder_in).logits
                keep = labels.reshape(-1).ne(IGNORE_INDEX)
                index = keep.nonzero(as_tuple=True)[0]
                vocabulary = student_logits.shape[-1]
                s_flat = student_logits.reshape(-1, vocabulary).index_select(0, index).float()
                t_flat = teacher_logits.reshape(-1, vocabulary).index_select(0, index).float()
                targets = labels.reshape(-1).index_select(0, index)

                loss = s_flat.new_zeros(())
                kl_value = ce_value = feature_value = 0.0
                if args.kl_weight > 0:
                    kl = F.kl_div(F.log_softmax(s_flat / args.temperature, -1),
                                  F.log_softmax(t_flat / args.temperature, -1),
                                  reduction="batchmean", log_target=True) * (args.temperature ** 2)
                    kl_value = float(kl.detach())
                    loss = loss + args.kl_weight * kl
                if args.ce_weight > 0:
                    ce = F.cross_entropy(s_flat, targets)
                    ce_value = float(ce.detach())
                    loss = loss + args.ce_weight * ce
                if student_tap is not None:
                    fl = feature_mse(student_tap.outputs, teacher_tap.outputs)
                    if fl is not None:
                        feature_value = float(fl.detach())
                        loss = loss + args.feature_weight * fl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            if is_main() and (step % args.log_every == 0 or step == 1):
                zero = plus = total = 0
                with torch.no_grad():
                    for module in list(qat_modules(student).values())[:8]:
                        codes, _ = module.export_codes()
                        flat = codes.reshape(-1)
                        zero += int((flat == 0).sum()); total += flat.numel()
                elapsed = time.time() - t0
                print(f"[step {step:6d}] loss {float(loss):.4f} (kl {kl_value:.4f} "
                      f"ce {ce_value:.4f} feat {feature_value:.4f}) "
                      f"lr {scheduler.get_last_lr()[0]:.2e} gnorm {float(grad_norm):.2f} "
                      f"zero {100*zero/max(total,1):.1f}% {elapsed/step:.3f}s/шаг", flush=True)

            if is_main() and args.save_every and step % args.save_every == 0:
                path = out / f"step{step:06d}.pt"
                save_qat_checkpoint(student, path, manifest=qat_manifest,
                                    extra={"step": step})
                print(f"[save] {path}", flush=True)
                history.append({"step": step, "loss": float(loss),
                                "kl": kl_value, "ce": ce_value})
                (out / "history.jsonl").write_text(
                    "\n".join(json.dumps(h) for h in history), encoding="utf-8")

    if world > 1:
        dist.barrier()
    if is_main():
        save_qat_checkpoint(student, out / "final.pt", manifest=qat_manifest,
                            extra={"step": step})
        print(f"[train] {step} шагов за {(time.time()-t0)/60:.1f} мин, "
              f"пик {torch.cuda.max_memory_allocated()/2**30:.1f} ГиБ", flush=True)
        print("[DONE]", flush=True)
    if student_tap is not None:
        student_tap.close(); teacher_tap.close()
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
