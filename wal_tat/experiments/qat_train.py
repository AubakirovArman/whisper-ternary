"""Train a fully quantised Whisper by distillation from its own fp16 teacher.

The run is a single pass through five stages, all in one process so the numbers
it prints refer to one and the same set of weights:

1. load the frozen teacher and an fp32 student from the *same* checkpoint;
2. calibrate diagonal activation second moments on a few teacher-forced
   batches and convert **every** target ``nn.Linear`` at once;
3. distil (:class:`wal_tat.qat.distill.QATDistiller`) -- teacher forced, no
   decoding, all matrices quantised from step zero;
4. export the deployable packed artifact and verify the code alphabet;
5. score WER on the evaluation split twice -- once through the QAT graph and
   once through a plain model rebuilt from the packed artifact -- and compare
   against a frozen full-precision baseline.

Example
-------
    CUDA_VISIBLE_DEVICES=2 python experiments/qat_train.py \
        --model tiny --precision t3 --steps 3000 --batch 48 --lr 2e-4 \
        --data cache/qat/features/whisper-small-train-clean-100 \
        --out results/qat/tiny_t3_v1 \
        --baseline results/qat/whisper-tiny_bf16_dev-clean.json
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, Mapping, Optional

import torch
from torch.utils.data import ConcatDataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from wal_tat import atomic_write_json
from wal_tat.binary import unpack_binary_codes
from wal_tat.binary_packing import read_packed_binary_matrix, unpack_binary_matrix
from wal_tat.packing import (
    read_packed_matrix,
    unpack_partial_matrix,
    unpack_ternary_codes,
)
from wal_tat.qat import (
    DEFAULT_TARGETS,
    QATManifest,
    WhisperFeatureShards,
    build_dataloader,
    collect_input_second_moments,
    convert_model_to_qat,
    evaluate_wer,
    export_packed_artifact,
    load_librispeech,
    load_qat_checkpoint,
    qat_modules,
    save_qat_checkpoint,
)
from wal_tat.qat.distill import AUTOCAST_DTYPES, DistillConfig, QATDistiller

MODEL_ALIASES: Mapping[str, str] = {
    "tiny": "openai/whisper-tiny",
    "small": "openai/whisper-small",
    "base": "openai/whisper-base",
    "medium": "openai/whisper-medium",
}

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

#: Matrices per Whisper transformer block, used to assert full coverage.
_ENCODER_MATRICES = 6  # q,k,v,out of self-attention + fc1,fc2
_DECODER_MATRICES = 10  # the same plus the four cross-attention projections


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="tiny", help="tiny|small|<hf id or path>")
    parser.add_argument("--precision", default="t3", choices=("t3", "b1", "fp"),
                        help="t3/b1 квантуют; fp — контрольная ветка без квантования, тот же путь кода")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--scale-lr-mult", type=float, default=0.1)
    parser.add_argument("--fp-lr-mult", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.1)
    parser.add_argument("--feature-weight", type=float, default=0.0)
    parser.add_argument("--derived-scales", action="store_true",
                        help="Use absmean scales instead of learned ones.")
    parser.add_argument("--freeze-non-quantized", action="store_true",
                        help="Train only latent weights and group scales.")
    parser.add_argument("--autocast", default="bf16", choices=sorted(AUTOCAST_DTYPES))
    parser.add_argument("--teacher-dtype", default="bf16", choices=sorted(DTYPES))
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--data", type=Path, required=True, nargs="+",
                        help="Feature cache(s) built by experiments/qat_build_data.py. "
                             "Several caches are pooled into one corpus, so the "
                             "LibriSpeech train splits can be trained on together.")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--calibration-batches", type=int, default=4)

    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=0)

    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--init-from", type=Path, default=None,
                        help="стартовать с готового QAT-чекпоинта (дообучение); "
                             "оптимизатор и расписание начинаются заново")
    parser.add_argument("--split", default="dev-clean")
    parser.add_argument("--eval-n", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--baseline", type=Path, default=None,
                        help="Frozen full-precision result to divide by.")
    parser.add_argument("--skip-wer", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


def _log(message: str) -> None:
    print(message, flush=True)


def expected_matrix_count(config) -> int:
    """Number of target matrices implied by the architecture."""
    return (
        config.encoder_layers * _ENCODER_MATRICES
        + config.decoder_layers * _DECODER_MATRICES
    )


# --------------------------------------------------------------------------- #
# Artifact verification
# --------------------------------------------------------------------------- #


def audit_packed_artifact(directory: Path, precision: str) -> Dict[str, Any]:
    """Re-read every packed matrix and prove the codes are really low-bit.

    Reads the files back from disk rather than trusting the in-memory model:
    the alphabet, the code histogram and the absence of any full-precision
    fallback group are all measured on the bytes that would ship.
    """
    summary = json.loads((directory / "manifest.json").read_text())
    alphabet: set[int] = set()
    histogram: Dict[int, int] = {}
    total_codes = 0
    fallback_weights = 0
    uncommitted_groups = 0
    for record in summary["files"]:
        path = directory / str(record["file"])
        if precision == "t3":
            packed = read_packed_matrix(path)
            codes = unpack_ternary_codes(
                packed.codes_packed, packed.committed_groups * packed.group_size
            )
            uncommitted_groups += packed.total_groups - packed.committed_groups
            fallback_weights += int(packed.fallback_bf16.numel())
        else:
            packed = read_packed_binary_matrix(path)
            codes = unpack_binary_codes(
                packed.codes_packed, packed.total_groups * packed.group_size
            )
        values, counts = torch.unique(codes, return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            alphabet.add(int(value))
            histogram[int(value)] = histogram.get(int(value), 0) + int(count)
        total_codes += int(codes.numel())
    allowed = {-1, 0, 1} if precision == "t3" else {-1, 1}
    return {
        "matrices": len(summary["files"]),
        "artifact_bytes": summary["artifact_bytes"],
        "total_codes": total_codes,
        "alphabet": sorted(alphabet),
        "alphabet_is_valid": alphabet.issubset(allowed),
        "code_histogram": {str(k): v for k, v in sorted(histogram.items())},
        "code_fractions": {
            str(k): v / total_codes for k, v in sorted(histogram.items())
        },
        "uncommitted_groups": uncommitted_groups,
        "fallback_weights": fallback_weights,
    }


def materialize_from_artifact(
    model_id: str,
    student: torch.nn.Module,
    directory: Path,
    manifest: QATManifest,
    *,
    dtype: torch.dtype,
    device: torch.device,
    local_files_only: bool = False,
) -> WhisperForConditionalGeneration:
    """Rebuild a plain dense model from the packed artifact + trained fp state.

    Every quantised matrix is replaced by the matrix that
    :func:`wal_tat.packing.unpack_partial_matrix` reconstructs from disk; every
    parameter that stayed in the baseline dtype (layer norms, biases,
    embeddings, the convolutional frontend, the tied head) is copied from the
    trained student.  Nothing of the QAT machinery survives, so a WER measured
    on the result is a WER of the deployable weights.
    """
    fresh = WhisperForConditionalGeneration.from_pretrained(
        model_id, local_files_only=local_files_only, low_cpu_mem_usage=True
    )
    state = {
        key: value.detach().clone()
        for key, value in student.state_dict().items()
        if not key.endswith(".group_scale")
    }
    for entry in manifest.entries:
        suffix = ".waltq2" if manifest.precision == "t3" else ".waltb1"
        path = directory / f"{entry.name}{suffix}"
        if manifest.precision == "t3":
            weight = unpack_partial_matrix(read_packed_matrix(path))
        else:
            weight = unpack_binary_matrix(read_packed_binary_matrix(path))
        key = f"{entry.name}.weight"
        if state[key].shape != weight.shape:
            raise ValueError(f"{key}: artifact shape {weight.shape} != {state[key].shape}")
        state[key] = weight
    missing, unexpected = fresh.load_state_dict(
        {key: value.to(torch.float32) for key, value in state.items()}, strict=False
    )
    unexpected = [key for key in unexpected if not key.endswith(".group_scale")]
    if unexpected:
        raise ValueError(f"unexpected keys when rebuilding: {unexpected[:5]}")
    tied = set(getattr(fresh, "_tied_weights_keys", []) or [])
    missing = [key for key in missing if key not in tied]
    if missing:
        raise ValueError(f"missing keys when rebuilding: {missing[:5]}")
    return fresh.to(device=device, dtype=dtype).eval()


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def run_wer(
    model,
    processor,
    split,
    args: argparse.Namespace,
    *,
    label: str,
    autocast: Optional[torch.dtype] = None,
) -> Dict[str, Any]:
    started = time.perf_counter()

    def progress(state: Mapping[str, Any]) -> None:
        if state["decoded"] % (args.eval_batch_size * 40) == 0:
            _log(f"[wer:{label}] {state['decoded']}/{state['total']}")

    context = (
        torch.autocast(device_type="cuda", dtype=autocast)
        if autocast is not None
        else nullcontext()
    )
    with context:
        result = evaluate_wer(
            model,
            processor,
            split,
            batch_size=args.eval_batch_size,
            max_new_tokens=args.max_new_tokens,
            compute_nll=True,
            progress=progress,
        )
    _log(
        f"[wer:{label}] {100 * result['wer']:.4f}% "
        f"(S {result['S']} D {result['D']} I {result['I']} N {result['N']}, "
        f"nll {result['nll']:.4f}) in {time.perf_counter() - started:.0f}s"
    )
    return result


def main(argv: Optional[list[str]] = None) -> int:
    # Предохранитель для карты, где живут чужие процессы: жёсткий потолок доли
    # видеопамяти, чтобы ошибка в конфиге не уронила соседей по GPU.
    fraction = os.environ.get("WALTAT_GPU_MEMORY_FRACTION")
    if fraction and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(fraction))

    args = parse_args(argv)
    out = args.out.expanduser().resolve()
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out} is not empty; pass --overwrite to reuse it")
    out.mkdir(parents=True, exist_ok=True)

    model_id = MODEL_ALIASES.get(args.model, args.model)
    device = torch.device(args.device)
    teacher_dtype = DTYPES[args.teacher_dtype]
    started = time.time()
    torch.manual_seed(args.seed)

    # -- models ----------------------------------------------------------- #
    _log(f"[load] {model_id} (teacher {args.teacher_dtype}, student fp32)")
    processor = WhisperProcessor.from_pretrained(
        model_id, local_files_only=args.local_files_only
    )
    teacher = WhisperForConditionalGeneration.from_pretrained(
        model_id, local_files_only=args.local_files_only, low_cpu_mem_usage=True,
        dtype=teacher_dtype,
    ).to(device=device, dtype=teacher_dtype).eval()
    student = WhisperForConditionalGeneration.from_pretrained(
        model_id, local_files_only=args.local_files_only, low_cpu_mem_usage=True,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.float32)
    dense_parameters = int(sum(p.numel() for p in student.parameters()))

    # -- data ------------------------------------------------------------- #
    train_parts = [WhisperFeatureShards(path, subset="train") for path in args.data]
    val_parts = [WhisperFeatureShards(path, subset="validation") for path in args.data]
    # Pooled caches must share a tokenizer and mel geometry or the batches they
    # produce are not interchangeable; a mismatch would train on silently
    # misaligned features instead of failing.
    signatures = {
        (part.model_id, part.n_mels, part.n_frames,
         int(part.manifest["labels"]["decoder_start_token_id"]))
        for part in train_parts
    }
    if len(signatures) != 1:
        raise SystemExit(f"[data] caches are not poolable: {sorted(signatures)}")
    for path, part in zip(args.data, train_parts):
        _log(f"[data] {path} -> train {part!r}")
    train_set = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    val_set = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
    _log(f"[data] pooled train n={len(train_set)} "
         f"hours={sum(p.total_hours for p in train_parts):.2f} | "
         f"valid n={len(val_set)} "
         f"hours={sum(p.total_hours for p in val_parts):.2f}")
    # ConcatDataset has no default_collator; take it from a part (all identical).
    collator = train_parts[0].default_collator()
    train_loader = build_dataloader(
        train_set, batch_size=args.batch, shuffle=True, seed=args.seed,
        num_workers=args.workers, drop_last=True, collator=collator,
    )
    val_loader = build_dataloader(
        val_set, batch_size=args.batch, shuffle=False,
        num_workers=max(args.workers // 2, 1), collator=collator,
    )

    # -- calibration + conversion ----------------------------------------- #
    calibration = [
        batch for _, batch in zip(range(args.calibration_batches), train_loader)
    ]
    _log(f"[calib] {len(calibration)} batches x {args.batch} utterances")

    def run_calibration() -> None:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for batch in calibration:
                student(
                    input_features=batch["input_features"].to(device),
                    labels=batch["labels"].to(device),
                )

    moments = collect_input_second_moments(student, run_calibration)
    _log(f"[calib] collected {len(moments)} activation moments")

    manifest = convert_model_to_qat(
        student,
        precision=args.precision,
        group_size=args.group_size,
        learnable_scales=not args.derived_scales,
        input_second_moments=moments,
    )
    expected = expected_matrix_count(student.config)
    _log(f"[convert] {manifest.summary()}")
    _log(f"[convert] untouched linears: {manifest.untouched_linear}")
    if manifest.num_matrices != expected:
        raise SystemExit(
            f"converted {manifest.num_matrices} matrices, architecture implies "
            f"{expected}: coverage of the target set is not 100%"
        )
    if args.init_from is not None:
        # Продолжение с готового чекпоинта: модель уже сконвертирована выше,
        # поэтому load_qat_checkpoint только подставит веса (внутри стоит
        # `if convert and not qat_modules(model)`).  Оптимизатор и расписание
        # при этом начинаются заново — это дообучение, а не возобновление.
        prior_manifest, prior_extra = load_qat_checkpoint(student, args.init_from)
        if prior_manifest.precision != args.precision:
            raise SystemExit(
                f"чекпоинт имеет precision={prior_manifest.precision!r}, "
                f"а запуск идёт с {args.precision!r}"
            )
        _log(f"[init] веса взяты из {args.init_from} "
             f"(шаг {prior_extra.get('step', '?')}); "
             f"оптимизатор и расписание стартуют заново")
    del calibration, moments
    torch.cuda.empty_cache()

    # -- training ---------------------------------------------------------- #
    config = DistillConfig(
        steps=args.steps,
        batch_size=args.batch,
        grad_accum=args.grad_accum,
        lr=args.lr,
        scale_lr_mult=args.scale_lr_mult,
        fp_lr_mult=args.fp_lr_mult,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup,
        min_lr_ratio=args.min_lr_ratio,
        grad_clip=args.grad_clip,
        temperature=args.temperature,
        kl_weight=args.kl_weight,
        ce_weight=args.ce_weight,
        feature_weight=args.feature_weight,
        autocast=args.autocast,
        train_non_quantized=not args.freeze_non_quantized,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        save_every=args.save_every,
    )
    distiller = QATDistiller(
        student,
        teacher,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        manifest=manifest,
        checkpoint_dir=out,
        log=_log,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    report = distiller.run()
    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    _log(
        f"[train] {report.steps_completed} steps in "
        f"{report.elapsed_seconds / 60:.1f} min, peak {peak_bytes / 2**30:.1f} GiB"
    )

    checkpoint = save_qat_checkpoint(
        student, out / "final.pt", manifest=manifest,
        extra={"step": report.steps_completed, "config": config.to_dict()},
    )
    _log(f"[save] {checkpoint}")

    # -- export + audit ----------------------------------------------------- #
    packed_dir = out / "packed"
    artifact = export_packed_artifact(student, packed_dir, manifest=manifest)
    audit = audit_packed_artifact(packed_dir, args.precision)
    _log(
        f"[export] {audit['matrices']} files, {audit['artifact_bytes'] / 2**20:.2f} MiB, "
        f"alphabet {audit['alphabet']} valid={audit['alphabet_is_valid']}, "
        f"fractions {audit['code_fractions']}"
    )
    if not audit["alphabet_is_valid"]:
        raise SystemExit(f"packed codes contain values outside the codebook: {audit['alphabet']}")
    if audit["uncommitted_groups"] or audit["fallback_weights"]:
        raise SystemExit("packed artifact contains full-precision fallback groups")

    # -- quality ------------------------------------------------------------ #
    quality: Dict[str, Any] = {}
    if not args.skip_wer:
        split = load_librispeech(args.split, n=args.eval_n)
        _log(f"[wer] {len(split)} utterances of {args.split}")
        quality["qat_graph"] = run_wer(
            student, processor, split, args,
            label="qat", autocast=AUTOCAST_DTYPES[args.autocast],
        )
        del train_loader, val_loader
        torch.cuda.empty_cache()
        reloaded = materialize_from_artifact(
            model_id, student, packed_dir, manifest,
            dtype=teacher_dtype, device=device,
            local_files_only=args.local_files_only,
        )
        quality["reloaded_artifact"] = run_wer(
            reloaded, processor, split, args, label="reload", autocast=None,
        )

    baseline = None
    if args.baseline is not None:
        baseline = json.loads(Path(args.baseline).read_text())

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "wal-tat-qat-train",
        "model": model_id,
        "precision": args.precision,
        "dense_parameters": dense_parameters,
        "manifest": manifest.to_dict(),
        "expected_target_matrices": expected,
        "coverage_is_complete": manifest.num_matrices == expected,
        "artifact": {
            "directory": str(packed_dir),
            "matrices": artifact["matrices"],
            "bytes": artifact["artifact_bytes"],
            "audit": audit,
        },
        "training": report.to_dict(),
        "peak_cuda_bytes": peak_bytes,
        "quality": quality,
        "baseline": None,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "command": " ".join(sys.argv),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if baseline is not None and quality:
        payload["baseline"] = {
            "file": str(args.baseline),
            "model": baseline["model"],
            "wer": baseline["wer"],
            "nll": baseline.get("nll"),
        }
        for key, result in quality.items():
            payload["baseline"][f"relative_wer_{key}"] = (
                result["wer"] / baseline["wer"] if baseline["wer"] else None
            )
    atomic_write_json(out / "report.json", payload)

    _log("")
    _log("=== summary ===")
    _log(f"matrices        {manifest.num_matrices} / {expected} target linears")
    _log(f"weights         {manifest.quantized_weights:,} "
         f"({100 * manifest.parameter_coverage:.2f}% of all parameters)")
    _log(f"model bpw       {manifest.effective_bpw:.3f} "
         f"({manifest.compression_ratio:.2f}x vs bf16)")
    if report.initial_validation and report.final_validation:
        _log(f"val loss        {report.initial_validation['loss']:.4f} -> "
             f"{report.final_validation['loss']:.4f}")
    for key, result in quality.items():
        line = f"WER {key:<18} {100 * result['wer']:.4f}%"
        if baseline is not None and baseline["wer"]:
            line += f"  relative {result['wer'] / baseline['wer']:.4f}"
        _log(line)
    if baseline is not None:
        _log(f"baseline WER    {100 * baseline['wer']:.4f}%")
    _log(f"[write] {out / 'report.json'}")
    _log("[DONE]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
