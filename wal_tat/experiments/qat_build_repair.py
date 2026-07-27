"""Набор для целевой починки двух измеренных дефектов сразу.

Дефект 1 — стена нулевого паддинга
----------------------------------
Whisper всегда считает мел из 3000 кадров: если звук короче 30 с, хвост
добивается тишиной.  Замер на шаге 60 000::

    вход 29.99 с → 1 кадр тишины → 82 слова,  0 повторов
    вход 30.00 с → 0 кадров      → 92 слова, 51 повтор

Разница — 10 миллисекунд.  В упакованном корпусе окон ровно на 30.000000 с
оказалось 133 из 146 564 (0.091%) — модель их почти не видела.

Дефект 2 — штатный long-form
----------------------------
За 951 час обучения таймстемпов в метках было РОВНО НОЛЬ.  Модель перестала
ставить промежуточные таймстемпы, и последовательный алгоритм Whisper выродился
в тупую нарезку по 30 с — число в число::

    HF long-form         15.435%   96 вставок   51 повтор
    нарезка 30с (стена)  15.435%   96 вставок   51 повтор

Учитель на тех же записях режет их на сегменты по 5–9 с и даёт 3.891%.

Что делает этот сборщик
-----------------------
Каждое окно получает **настоящие таймстемпы**: границы исходных реплик известны
точно в момент упаковки, поэтому размечать ничего не надо и учителя гонять не
надо.  Доля окон растягивается по времени ровно до 480 000 отсчётов —
это и есть нулевой паддинг, которого модель не видела.

Метка выглядит так::

    <|startoftranscript|><|en|><|transcribe|>
    <|0.00|> первая реплика <|8.20|><|8.20|> вторая реплика <|15.40|>
    <|endoftext|>

Обратите внимание: ``<|notimestamps|>`` при этом ОТСУТСТВУЕТ — именно поэтому
такие окна учат модель тому режиму, в котором работает штатный long-form.

Набор идёт ДОБАВКОЙ к основному корпусу, а не заменой: обучать только на
30-секундных окнах значило бы повторить ошибку упаковки в зеркальном виде и
потерять короткие записи, на которых сейчас всё хорошо.
"""

import os as _os
import pathlib as _pl
# корень репозитория: от расположения файла, можно переопределить WALTAT_ROOT
_REPO = _pl.Path(_os.environ.get('WALTAT_ROOT',
                                 _pl.Path(__file__).resolve().parents[2]))
_REPO_STR = str(_REPO)

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import WhisperProcessor

ROOT = Path(f"{_REPO_STR}/wal_tat")
sys.path.insert(0, str(ROOT / "src"))

RATE = 16000
WINDOW_SAMPLES = RATE * 30
TS_BASE = 50364          # <|0.00|>
TS_STEP = 0.02           # секунд на один токен таймстемпа
TS_MAX = 1500            # <|30.00|>
SHARD_SIZE = 256


def _wave(rec) -> np.ndarray:
    a = rec["audio"]
    if isinstance(a, dict) and a.get("bytes"):
        data, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        if sr != RATE:
            raise ValueError(f"sampling rate {sr}")
        return np.asarray(data, dtype=np.float32)
    return np.asarray(a["array"], dtype=np.float32)


def _ts(seconds: float) -> int:
    """Токен таймстемпа: сетка 0.02 с, зажата в [0, 30] секунд."""
    return TS_BASE + int(min(max(round(seconds / TS_STEP), 0), TS_MAX))


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _stretch(wave: np.ndarray) -> np.ndarray:
    """Растянуть ровно до WINDOW_SAMPLES; расхождение < одного отсчёта."""
    if wave.size >= WINDOW_SAMPLES:
        return np.ascontiguousarray(wave[:WINDOW_SAMPLES], dtype=np.float32)
    out = resample_poly(wave, WINDOW_SAMPLES, wave.size).astype(np.float32)
    if out.size > WINDOW_SAMPLES:
        out = out[:WINDOW_SAMPLES]
    elif out.size < WINDOW_SAMPLES:
        out = np.pad(out, (0, WINDOW_SAMPLES - out.size), mode="edge")
    return np.ascontiguousarray(out, dtype=np.float32)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, nargs="+", default=[
        ROOT / "cache/qat/librispeech_asr/clean/train.360",
        ROOT / "cache/qat/librispeech_asr/clean/train.100",
    ])
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "cache/qat/features/whisper-small-repair-ts")
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--windows", type=int, default=35000)
    p.add_argument("--stretch-every", type=int, default=3,
                   help="каждое N-е окно растягивается ровно до 30.000 с")
    p.add_argument("--min-seconds", type=float, default=20.0)
    p.add_argument("--max-seconds", type=float, default=29.5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    out = args.output_dir
    (out / "shards").mkdir(parents=True, exist_ok=True)
    proc = WhisperProcessor.from_pretrained(args.model)
    tok, extractor = proc.tokenizer, proc.feature_extractor
    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
    lang = tok.convert_tokens_to_ids("<|en|>")
    task = tok.convert_tokens_to_ids("<|transcribe|>")
    eot = tok.eos_token_id

    print(f"[cfg] цель {args.windows} окон, каждое {args.stretch_every}-е "
          f"растянуто до 30.000 с, ВСЕ с таймстемпами", flush=True)

    ready, buf, shards, index, made, t0 = [], [], [], 0, 0, time.time()
    stretched = 0
    open_g = None

    def close():
        nonlocal open_g
        g, open_g = open_g, None
        if g is None:
            return None
        secs = sum(w.size for w in g["waves"]) / RATE
        return g if args.min_seconds <= secs <= args.max_seconds else None

    def write_shard():
        nonlocal buf, index
        if not buf:
            return
        feats = np.stack([b.pop("_feat") for b in buf]).astype(np.float16)
        np.save(out / "shards" / f"features_{index:05d}.npy", feats)
        (out / "shards" / f"shard_{index:05d}.json").write_text(
            json.dumps({"items": buf}, ensure_ascii=False), encoding="utf-8")
        shards.append({"index": index, "count": len(buf)})
        index += 1
        buf = []

    def emit(batch):
        """Собрать метки с таймстемпами и посчитать признаки."""
        nonlocal made, stretched
        waves, metas = [], []
        for g in batch:
            joined = np.concatenate(g["waves"])
            do_stretch = (made + len(metas)) % args.stretch_every == 0
            if do_stretch:
                wave = _stretch(joined)
                scale = WINDOW_SAMPLES / joined.size
            else:
                wave = joined
                scale = 1.0
            waves.append(wave)
            # границы участников: масштабируются вместе со звуком
            labels = [sot, lang, task]
            cursor = 0.0
            for wv, txt in zip(g["waves"], g["texts"]):
                start = cursor
                cursor += (wv.size / RATE) * scale
                piece = _norm(txt)
                if not piece:
                    continue
                labels.append(_ts(start))
                labels.extend(tok(" " + piece, add_special_tokens=False).input_ids)
                labels.append(_ts(cursor))
            labels.append(eot)
            metas.append({"g": g, "labels": labels, "wave": wave,
                          "stretched": bool(do_stretch),
                          "seconds": wave.size / RATE})
            if do_stretch:
                stretched += 1
        enc = extractor(waves, sampling_rate=RATE, padding="max_length",
                        return_tensors="pt", device=args.device)["input_features"]
        enc = enc.to(torch.float16).cpu().numpy()
        for meta, feat in zip(metas, enc):
            g = meta["g"]
            buf.append({
                "utterance_id": g["ids"][0] + ("_ts30" if meta["stretched"] else "_ts"),
                "text": " " + " ".join(_norm(t) for t in g["texts"] if t.strip()),
                "labels": meta["labels"],
                "speaker_id": g["spk"], "chapter_id": g["ch"],
                "duration_seconds": round(meta["seconds"], 4),
                "num_samples": int(meta["wave"].size),
                "num_source_utterances": len(g["ids"]),
                "source_utterance_ids": g["ids"],
                "has_timestamps": True, "exact_window": meta["stretched"],
                "_feat": feat})
            made += 1

    for src in args.sources:
        if made >= args.windows:
            break
        for path in sorted(src.glob("*.parquet")):
            if made >= args.windows:
                break
            table = pq.read_table(path, columns=["id", "audio", "text",
                                                 "speaker_id", "chapter_id"])
            for rec in table.to_pylist():
                if made >= args.windows:
                    break
                idx = int(str(rec["id"]).rsplit("-", 1)[-1])
                wave = _wave(rec)
                done = None
                g = open_g
                if g is not None:
                    same = (g["spk"] == rec["speaker_id"]
                            and g["ch"] == rec["chapter_id"]
                            and idx == g["last"] + 1)
                    fits = (sum(w.size for w in g["waves"]) + wave.size
                            <= int(args.max_seconds * RATE))
                    if not same or not fits:
                        done = close()
                if open_g is None:
                    open_g = {"spk": rec["speaker_id"], "ch": rec["chapter_id"],
                              "last": idx, "waves": [wave],
                              "texts": [rec["text"]], "ids": [rec["id"]]}
                else:
                    open_g["waves"].append(wave)
                    open_g["texts"].append(rec["text"])
                    open_g["ids"].append(rec["id"])
                    open_g["last"] = idx
                if done is not None:
                    ready.append(done)
                if len(ready) >= args.batch:
                    emit(ready)
                    ready = []
                    if len(buf) >= SHARD_SIZE:
                        write_shard()
                    if made % 2000 < args.batch:
                        print(f"[pack] {made}/{args.windows}  "
                              f"растянутых {stretched}  ({time.time()-t0:.0f} с)",
                              flush=True)
    if ready:
        emit(ready)
    write_shard()

    lens, secs, nts = [], [], 0
    for s in shards:
        items = json.loads((out / "shards" / f"shard_{s['index']:05d}.json")
                           .read_text(encoding="utf-8"))["items"]
        for i in items:
            lens.append(len(i["labels"]))
            secs.append(i["duration_seconds"])
            nts += sum(1 for t in i["labels"] if t >= TS_BASE)
    (out / "manifest.json").write_text(json.dumps({
        "format_version": 1, "model_id": args.model, "split": "train",
        "num_utterances": made, "shard_size": SHARD_SIZE, "shards": shards,
        "total_audio_seconds": float(sum(secs)),
        "features": {"dtype": "float16", "shape": [80, 3000]},
        "labels": {"language": "en", "task": "transcribe", "timestamps": True},
        "repair": {"exact_window_count": stretched,
                   "timestamps_per_window_mean": nts / max(made, 1),
                   "fixes": ["стена нулевого паддинга (вход ровно 30.000 с)",
                             "штатный long-form (промежуточные таймстемпы)"]},
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    ex = sum(1 for s in secs if abs(s - 30.0) < 1e-4)
    print(f"\n[done] {made} окон = {sum(secs)/3600:.1f} ч за {time.time()-t0:.0f} с")
    print(f"[assert] ровно 30.000000 с: {ex} окон ({100*ex/made:.1f}%)")
    print(f"[assert] таймстемпов на окно: {nts/made:.1f} (по 2 на реплику)")
    print(f"[labels] медиана {np.median(lens):.0f}  p99 {np.percentile(lens,99):.0f}  макс {max(lens)}")
    print(f"[durations] медиана {np.median(secs):.2f}с  мин {min(secs):.2f}  макс {max(secs):.2f}")
    print("[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
