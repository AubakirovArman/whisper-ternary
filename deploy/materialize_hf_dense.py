"""Материализовать QAT-чекпоинт как обычную HF-модель (плотные веса).

Зачем: экспортёр в ggml раньше брал неквантованные тензоры из СТОКОВОЙ
ggml-small.bin, а они у нас обучались 80k шагов (fp_lr_mult=0.25) — эмбеддинги,
layernorm'ы, свёртки, смещения.  Получалась химера из двух несовместимых
половин: наши тернарные матрицы + стоковая обвязка -> «почти правильный, но
изуродованный» вывод в whisper.cpp.

Правильный путь: полная наша модель -> их же convert-h5-to-ggml.py (он знает
все 287 маппингов имён и транспонирования) -> наш экспортёр подменяет только
192 матрицы точными блоками Q2_0.
"""

import os as _os
import pathlib as _pl
# корень репозитория: от расположения файла, можно переопределить WALTAT_ROOT
_REPO = _pl.Path(_os.environ.get('WALTAT_ROOT',
                                 _pl.Path(__file__).resolve().parents[1]))
_REPO_STR = str(_REPO)

import sys

import torch

sys.path.insert(0, f"{_REPO_STR}/wal_tat/src")
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from wal_tat.qat import load_qat_checkpoint, qat_modules

CK = (f"{_REPO_STR}/wal_tat/results/qat/"
      "small_t3_repair_v3/step010000.pt")
OUT = "/tmp/wal_hf_dense"

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small", dtype=torch.float32)
load_qat_checkpoint(model, CK)

# QuantLinear -> обычный Linear с развёрнутым весом (коды * выученные масштабы)
for name, mod in list(qat_modules(model).items()):
    codes, scales = mod.export_codes()
    W = codes.float() * scales.float().unsqueeze(-1)
    dense = W.reshape(W.shape[0], -1)[:, : mod.in_features].contiguous()
    lin = torch.nn.Linear(mod.in_features, dense.shape[0],
                          bias=mod.bias is not None)
    lin.weight.data = dense
    if mod.bias is not None:
        lin.bias.data = mod.bias.detach().clone()
    parent = model.get_submodule(name.rsplit(".", 1)[0])
    setattr(parent, name.rsplit(".", 1)[1], lin)

leftover = len(qat_modules(model))
assert leftover == 0, f"осталось QuantLinear: {leftover}"

model.save_pretrained(OUT, safe_serialization=True)
WhisperProcessor.from_pretrained("openai/whisper-small").save_pretrained(OUT)
print(f"[done] полная плотная модель: {OUT}")
