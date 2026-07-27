#!/bin/bash
# Ручная проверка WER тернарной модели.  Протокол закреплён внутри check_wer.py
# (батч 16, лимит 440 токенов) — менять его флагами нельзя, это часть измерения.
#
#   bash wal_tat/check_wer.sh                      # весь dev-clean, текущий прогон
#   bash wal_tat/check_wer.sh --n 500              # быстро, но БЕЗ длинных записей
#   bash wal_tat/check_wer.sh --run small_t3_full_v1   # прогон №1 (якорь)
#   bash wal_tat/check_wer.sh --ckpt путь/к/checkpoint.pt
#   bash wal_tat/check_wer.sh --history            # история всех замеров
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"/wal_tat/src \
./.venv/bin/python wal_tat/check_wer.py "$@" 2>&1 \
  | grep -vE "Loading weights|it/s\]|max_new_tokens|logits processor|clean_up_tokenization|^$"
