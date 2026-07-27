"""Проверка тернарной модели, поднятой службой: 20 аудио через HTTP /inference.

Берём разброс по длительности, считаем WER на этой двадцатке и сверяем ответы
службы с тем, что даёт CLI на тех же файлах — они обязаны совпасть дословно.
"""

import os as _os
import pathlib as _pl
# корень репозитория: от расположения файла, можно переопределить WALTAT_ROOT
_REPO = _pl.Path(_os.environ.get('WALTAT_ROOT',
                                 _pl.Path(__file__).resolve().parents[1]))
_REPO_STR = str(_REPO)

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, f"{_REPO_STR}/wal_tat/src")
from transformers import WhisperProcessor
from wal_tat.qat.evaluate import build_english_normalizer, corpus_word_error_counts

URL = "http://127.0.0.1:8178/inference"
WAVS = Path("/tmp/wcpp/other_wavs")
MODEL = "/tmp/wcpp/models/ggml-small-wal-q2_0-v2.bin"

rows = [l.split("\t") for l in Path("/tmp/wcpp/other_refs.tsv").read_text().splitlines()]
rows = [(r, float(d), t) for r, d, t in rows]
rows.sort(key=lambda x: x[1])
pick = [rows[round(i * (len(rows) - 1) / 19)] for i in range(20)]

print(f"  длительности: {pick[0][1]:.1f} — {pick[-1][1]:.1f} с\n")
print(f"  {'#':<6}{'сек':>6}{'мс':>8}{'RTF':>7}  результат")

refs, hyps, times, mismatch, errors = [], [], [], [], []
for rid, dur, ref in pick:
    wav = WAVS / f"{rid}.wav"
    t0 = time.perf_counter()
    try:
        with open(wav, "rb") as fh:
            r = requests.post(URL, files={"file": (wav.name, fh, "audio/wav")},
                              data={"temperature": "0.0", "response_format": "json",
                                    "language": "en"}, timeout=300)
        r.raise_for_status()
        hyp = " ".join(r.json()["text"].split())
    except Exception as exc:
        errors.append((rid, f"{type(exc).__name__}: {exc}"))
        print(f"  {rid:<6}{dur:6.1f}   ОШИБКА {type(exc).__name__}")
        continue
    ms = (time.perf_counter() - t0) * 1000

    refs.append(ref); hyps.append(hyp); times.append((ms, dur))
    print(f"  {rid:<6}{dur:6.1f}{ms:8.0f}{ms/1000/dur:7.3f}  {hyp[:50]}")

    cli = subprocess.run(
        ["/tmp/wcpp/build/bin/whisper-cli", "-m", MODEL, "-f", str(wav),
         "-t", "16", "-ng", "-np", "-l", "en", "-bs", "1"],
        capture_output=True, text=True, timeout=300)
    cli_txt = " ".join(re.sub(r"\[[^\]]*\]", " ", cli.stdout).split())
    if cli_txt != hyp:
        mismatch.append((rid, hyp, cli_txt))

proc = WhisperProcessor.from_pretrained("openai/whisper-small")
t, _ = corpus_word_error_counts(refs, hyps, normalizer=build_english_normalizer(proc))
wer = 100 * (t.substitutions + t.deletions + t.insertions) / t.reference_words

tot_ms = sum(m for m, _ in times); tot_s = sum(d for _, d in times)
print(f"\n  успешных запросов:   {len(refs)}/20   ошибок: {len(errors)}")
print(f"  WER на этих 20:      {wer:.4f}%   (S={t.substitutions} D={t.deletions} "
      f"I={t.insertions}, слов {t.reference_words})")
print(f"  аудио суммарно:      {tot_s:.1f} с, обработано за {tot_ms/1000:.1f} с  "
      f"→ RTF {tot_ms/1000/tot_s:.3f}  (быстрее реального времени в {tot_s/(tot_ms/1000):.1f}×)")
print(f"  служба против CLI:   {'СОВПАЛО дословно на всех' if not mismatch else f'РАСХОЖДЕНИЙ {len(mismatch)}'}")
for rid, a, b in mismatch[:3]:
    print(f"     {rid}\n       служба: {a}\n       CLI   : {b}")

Path("/tmp/api_check_result.json").write_text(json.dumps({
    "wer_pct": round(wer, 4), "ok": len(refs), "errors": len(errors),
    "words": t.reference_words, "S": t.substitutions, "D": t.deletions, "I": t.insertions,
    "audio_s": round(tot_s, 1), "wall_s": round(tot_ms / 1000, 1),
    "rtf": round(tot_ms / 1000 / tot_s, 4), "service_vs_cli_identical": not mismatch,
}, ensure_ascii=False, indent=1))
