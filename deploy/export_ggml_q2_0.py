"""Экспорт тернарной QAT-модели в ggml-формат whisper.cpp с типом Q2_0.

Почему Q2_0, а не TQ2_0
-----------------------
Оба тернарных формата уже есть в ggml, который whisper.cpp тянет подмодулем, —
вместе с CPU-ядрами (скалярным и NEON).  Различаются они размером группы, и это
решается замером::

    наша модель, группа 128       2.7145 %
    Q2_0,        группа  64       2.7145 %   потерь НЕТ
    TQ2_0,       группа 256       3.3372 %   +0.62 пп — отвергнут

Группа 64 делит каждую нашу 128-группу пополам, обе половины наследуют тот же
масштаб — реконструкция ``код * масштаб`` не меняется ни на бит.  Группа 256
склеивала бы две группы с разными масштабами под один, и это уже потеря.

Раскладка совпадает побитово
----------------------------
``block_q2_0`` в ggml: ``ggml_half d`` плюс ``uint8 qs[16]`` на 64 веса,
байт ``j/4``, сдвиг ``(j%4)*2``, хранится ``код + 1``.  Это ровно та же
раскладка, что в :func:`wal_tat.kernels.pack_codes_rowwise` — мы пришли к ней
независимо.  Алфавит Q2_0 шире нашего (``{-1,0,+1,+2}``), код ``11`` мы просто
не используем.

Веса берутся РАЗВЁРНУТЫЕ, а не латентные
----------------------------------------
Латентные fp32-веса QAT непригодны как мастер-копия: за 80k шагов STE они
уезжают от рабочих в 3.8 раза по норме, 60 % сидят за порогами округления.
Проекция латентных весов свежими масштабами даёт 1000 % WER — проверено.
Мастер-представление это **пара** (латент, выученные масштабы), а как матрица
весов осмыслен только их продукт ``codes * scales``.

Что делает скрипт
-----------------
1. читает базовую ggml-модель whisper.cpp (F16), полученную их же конвертером;
2. для 192 матриц энкодера и декодера подставляет наши коды и масштабы,
   записывая блоки Q2_0 напрямую, без пересчёта масштабов средствами ggml;
3. остальные тензоры (эмбеддинги, свёртки, layernorm) копирует как есть;
4. сверяет каждую подменённую матрицу: распаковывает записанные блоки тем же
   правилом, что ``dequantize_row_q2_0``, и сравнивает с исходной матрицей.
"""

import os as _os
import pathlib as _pl
# корень репозитория: от расположения файла, можно переопределить WALTAT_ROOT
_REPO = _pl.Path(_os.environ.get('WALTAT_ROOT',
                                 _pl.Path(__file__).resolve().parents[1]))
_REPO_STR = str(_REPO)

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(f"{_REPO_STR}/wal_tat")
sys.path.insert(0, str(ROOT / "src"))

from transformers import WhisperForConditionalGeneration  # noqa: E402
from wal_tat.qat import load_qat_checkpoint, qat_modules  # noqa: E402

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q2_0 = 42
QK2_0 = 64

#: Соответствие имён: PyTorch-модуль -> тензор в ggml-модели whisper.cpp.
_PT_TO_GGML = [
    ("model.encoder.layers.{i}.self_attn.q_proj", "encoder.blocks.{i}.attn.query.weight"),
    ("model.encoder.layers.{i}.self_attn.k_proj", "encoder.blocks.{i}.attn.key.weight"),
    ("model.encoder.layers.{i}.self_attn.v_proj", "encoder.blocks.{i}.attn.value.weight"),
    ("model.encoder.layers.{i}.self_attn.out_proj", "encoder.blocks.{i}.attn.out.weight"),
    ("model.encoder.layers.{i}.fc1", "encoder.blocks.{i}.mlp.0.weight"),
    ("model.encoder.layers.{i}.fc2", "encoder.blocks.{i}.mlp.2.weight"),
    ("model.decoder.layers.{i}.self_attn.q_proj", "decoder.blocks.{i}.attn.query.weight"),
    ("model.decoder.layers.{i}.self_attn.k_proj", "decoder.blocks.{i}.attn.key.weight"),
    ("model.decoder.layers.{i}.self_attn.v_proj", "decoder.blocks.{i}.attn.value.weight"),
    ("model.decoder.layers.{i}.self_attn.out_proj", "decoder.blocks.{i}.attn.out.weight"),
    ("model.decoder.layers.{i}.encoder_attn.q_proj", "decoder.blocks.{i}.cross_attn.query.weight"),
    ("model.decoder.layers.{i}.encoder_attn.k_proj", "decoder.blocks.{i}.cross_attn.key.weight"),
    ("model.decoder.layers.{i}.encoder_attn.v_proj", "decoder.blocks.{i}.cross_attn.value.weight"),
    ("model.decoder.layers.{i}.encoder_attn.out_proj", "decoder.blocks.{i}.cross_attn.out.weight"),
    ("model.decoder.layers.{i}.fc1", "decoder.blocks.{i}.mlp.0.weight"),
    ("model.decoder.layers.{i}.fc2", "decoder.blocks.{i}.mlp.2.weight"),
]


def deployed_weights(checkpoint: Path, model_id: str) -> dict[str, np.ndarray]:
    """Развёрнутые веса: ``codes * scales`` по каждой тернарной матрице."""
    model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=torch.float32)
    load_qat_checkpoint(model, checkpoint)
    out: dict[str, np.ndarray] = {}
    for name, module in qat_modules(model).items():
        codes, scales = module.export_codes()          # [out, groups, 128], [out, groups]
        product = codes.float() * scales.float().unsqueeze(-1)
        dense = product.reshape(product.shape[0], -1)[:, : module.in_features]
        out[name] = dense.numpy().astype(np.float32)
    return out


def ternary_blocks(dense: np.ndarray) -> np.ndarray:
    """Матрица -> байты блоков Q2_0, по строкам.

    Масштаб блока — максимум по модулю; для тернарной строки это и есть
    выученный масштаб, а коды восстанавливаются точно.  Блок целиком из нулей
    получает нулевой масштаб — так же, как это делает сам ggml.
    """
    rows, cols = dense.shape
    if cols % QK2_0:
        raise ValueError(f"{cols} не делится на {QK2_0}")
    grouped = dense.reshape(rows, cols // QK2_0, QK2_0)
    scale = np.abs(grouped).max(axis=-1)                                  # [rows, blocks]
    inv = np.where(scale > 0, 1.0 / np.maximum(scale, 1e-30), 0.0)
    codes = np.rint(grouped * inv[..., None]).astype(np.int32) + 1        # {-1,0,1} -> {0,1,2}
    codes = np.clip(codes, 0, 3).astype(np.uint8)
    lanes = codes.reshape(rows, -1, QK2_0 // 4, 4)
    packed = (lanes[..., 0] | (lanes[..., 1] << 2)
              | (lanes[..., 2] << 4) | (lanes[..., 3] << 6)).astype(np.uint8)
    half = scale.astype(np.float16).view(np.uint8).reshape(rows, -1, 2)
    return np.concatenate([half, packed], axis=-1).reshape(-1)            # [rows*blocks*18]


def unpack_blocks(raw: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Обратная операция — точная копия ``dequantize_row_q2_0`` для сверки."""
    blocks = cols // QK2_0
    view = raw.reshape(rows, blocks, 2 + QK2_0 // 4)
    scale = view[:, :, :2].copy().view(np.float16).astype(np.float32).reshape(rows, blocks)
    qs = view[:, :, 2:]
    lanes = np.stack([(qs >> shift) & 0x03 for shift in (0, 2, 4, 6)], axis=-1)
    codes = lanes.reshape(rows, blocks, QK2_0).astype(np.int32) - 1
    return (codes * scale[..., None]).reshape(rows, cols).astype(np.float32)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True,
                   help="ggml-модель whisper.cpp в F16 (convert-h5-to-ggml.py)")
    p.add_argument("--checkpoint", type=Path,
                   default=ROOT / "results/qat/small_t3_repair_v3/step010000.pt")
    p.add_argument("--model-id", default="openai/whisper-small")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--layers", type=int, default=12)
    args = p.parse_args(argv)

    weights = deployed_weights(args.checkpoint, args.model_id)
    mapping = {}
    for pt, gg in _PT_TO_GGML:
        for i in range(args.layers):
            key = pt.format(i=i)
            if key in weights:
                mapping[gg.format(i=i)] = weights[key]
    print(f"[map] тернарных матриц в чекпоинте {len(weights)}, "
          f"сопоставлено с ggml-именами {len(mapping)}", flush=True)
    if len(mapping) != len(weights):
        missing = set(weights) - {pt.format(i=i) for pt, _ in _PT_TO_GGML
                                  for i in range(args.layers)}
        raise SystemExit(f"не сопоставлены: {sorted(missing)[:5]}")

    src = open(args.base, "rb")
    dst = open(args.out, "wb")

    magic = src.read(4)
    if struct.unpack("<i", magic)[0] != 0x67676d6c:
        raise SystemExit("не ggml-файл")
    dst.write(magic)
    # гиперпараметры: ОДИННАДЦАТЬ int32 (whisper.cpp, whisper_model_load):
    # n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer,
    # n_text_ctx, n_text_state, n_text_head, n_text_layer, n_mels, ftype
    head = list(struct.unpack("<11i", src.read(44)))
    qnt_factor = 1000                                    # GGML_QNT_VERSION_FACTOR
    head[-1] = 2 * qnt_factor + 28   # GGML_QNT_VERSION=2, GGML_FTYPE_MOSTLY_Q2_0=28
    dst.write(struct.pack("<11i", *head))
    print(f"[hdr] ftype -> {head[-1]}", flush=True)

    # мел-фильтры и словарь копируются без изменений
    n_mel, n_fft = struct.unpack("<2i", src.read(8))
    dst.write(struct.pack("<2i", n_mel, n_fft))
    dst.write(src.read(n_mel * n_fft * 4))
    n_vocab = struct.unpack("<i", src.read(4))[0]
    dst.write(struct.pack("<i", n_vocab))
    for _ in range(n_vocab):
        ln = struct.unpack("<i", src.read(4))[0]
        dst.write(struct.pack("<i", ln))
        dst.write(src.read(ln))

    replaced, copied, worst = 0, 0, 0.0
    while True:
        header = src.read(12)
        if len(header) < 12:
            break
        n_dims, name_len, ttype = struct.unpack("<3i", header)
        ne = list(struct.unpack(f"<{n_dims}i", src.read(4 * n_dims)))
        name = src.read(name_len).decode()
        count = int(np.prod(ne))
        raw = src.read(count * (4 if ttype == GGML_TYPE_F32 else 2))

        if name in mapping:
            dense = mapping[name]                       # [out, in] в PyTorch-порядке
            if (ne[1], ne[0]) != dense.shape:
                raise SystemExit(f"{name}: ggml {ne} против чекпоинта {dense.shape}")
            blocks = ternary_blocks(dense)
            check = unpack_blocks(blocks, *dense.shape)
            worst = max(worst, float(np.abs(check - dense).max()))
            dst.write(struct.pack("<3i", n_dims, name_len, GGML_TYPE_Q2_0))
            dst.write(struct.pack(f"<{n_dims}i", *ne))
            dst.write(name.encode())
            dst.write(blocks.tobytes())
            replaced += 1
        else:
            dst.write(header)
            dst.write(struct.pack(f"<{n_dims}i", *ne))
            dst.write(name.encode())
            dst.write(raw)
            copied += 1

    src.close()
    dst.close()
    size = args.out.stat().st_size / 2 ** 20
    print(f"[done] заменено {replaced}, скопировано {copied}, файл {size:.1f} МиБ")
    print(f"[assert] макс |распаковка - исходная матрица| = {worst:.3e} "
          f"{'— ТОЧНО' if worst < 1e-6 else '— РАСХОЖДЕНИЕ!'}")
    if replaced != len(mapping):
        raise SystemExit(f"заменено {replaced}, ожидалось {len(mapping)}")
    print("[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
