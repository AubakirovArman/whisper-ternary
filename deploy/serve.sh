#!/bin/bash
# Поднять тернарный whisper-small службой на CPU.
#   -ng          только процессор
#   -ml 100000   снять принудительную нарезку сегментов по 60 символов:
#                умолчание сервера рвёт слова посередине ("fl uttered")
#   без -nt      режим с таймстемпами: в нём модель ставит финальные точки
BIN=${WCPP:-/tmp/wcpp}/build/bin/whisper-server
MODEL=$(dirname "$0")/ggml-small-wal-ternary-q2_0.bin
exec "$BIN" -m "$MODEL" --host 127.0.0.1 --port 8178 -t "${THREADS:-16}" -ng -ml 100000 "$@"
