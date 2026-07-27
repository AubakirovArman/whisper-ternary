#!/usr/bin/env python
"""Ручная проверка качества тернарной модели (WAL-TAT QAT).

ПРОТОКОЛ ЗАКРЕПЛЁН.  batch_size и max_new_tokens — часть измерения, не удобства:
  * batch_size меняет WER на ±7 ошибок (±0.013 п.п.) из-за недетерминизма
    батчевых matmul.  Это ШУМОВОЙ ПОРОГ харнесса: разницы < 0.02 п.п. сигналом
    не считаются.  Измерено: батч 16 → 3.4298%, батч 8 → 3.4224%, батч 1 → 3.4168%.
  * max_new_tokens=440, а не 128.  Старый лимит 128 обрезал выводы и МАСКИРОВАЛ
    разнос на длинных записях — модель выглядела лучше, чем есть.

ГЛАВНАЯ ЦИФРА — отношение на подмножестве ≤30 с.  Граница 30 с задана окном
Whisper (архитектурой), а не точкой отказа модели.  Резать по точке отказа
нельзя: это подгонка.  Разбивка по длительности идёт РЯДОМ, как диагностика.

Что делает:
  1. Берёт чекпоинт (по умолчанию — best.pt текущего прогона).
  2. Считает WER на dev-clean под закреплённым протоколом.
  3. Считает эталон bf16 на том же подмножестве (кэш по протоколу).
  4. Проверяет, что модель ДЕЙСТВИТЕЛЬНО тернарная (алфавит {-1,0,+1}).
  5. Печатает таблицу по корзинам длительности — здесь видно порог обобщения.
  6. Сравнивает с якорем прогона №1 (шаг 30000), если шаг совпал.

Примеры:
    python wal_tat/check_wer.py                    # весь dev-clean, текущий прогон
    python wal_tat/check_wer.py --n 500            # быстро (~2 мин), но БЕЗ длинных записей
    python wal_tat/check_wer.py --ckpt путь/до/checkpoint.pt
    python wal_tat/check_wer.py --run small_t3_full_v1   # старый прогон
    python wal_tat/check_wer.py --history          # только история замеров
"""

import os as _os
import pathlib as _pl
# корень репозитория: от расположения файла, можно переопределить WALTAT_ROOT
_REPO = _pl.Path(_os.environ.get('WALTAT_ROOT',
                                 _pl.Path(__file__).resolve().parents[1]))
_REPO_STR = str(_REPO)

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(f"{_REPO_STR}/wal_tat")
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results/qat"
HISTORY = RESULTS / "manual_wer_history.jsonl"
MODEL_ID = "openai/whisper-small"
DEFAULT_RUN = "small_t3_packed_v2"

# ── закреплённый протокол ─────────────────────────────────────────────────
BATCH_SIZE = 16
MAX_NEW_TOKENS = 440
NOISE_FLOOR_PP = 0.013

# ── якорь для сравнения: прогон №1 (неупакованные данные), чекпойнт 30000 ──
ANCHOR = {
    "run": "small_t3_full_v1",
    "step": 30000,
    "data": "неупакованный корпус (медиана 13.9 с, окон ≥25 с — 0.0%)",
    "le30_wer_pct": 4.2275,      # тернарь — воспроизводится до знака
    "le30_relative": 1.2366,     # против КЭШИРОВАННОГО эталона 3.4187%
    "baseline_pct": 3.4187,      # кэш фиксирует эталон: сравнения между прогонами точны
    "gt30_wer_pct": 148.25,
}

# 28-30 отделена от 25-28 намеренно: в упакованном корпусе окон ≥28 с только
# 29.3%, и если слабость останется — она будет именно в этом верхнем хвосте.
BUCKETS = [(0, 10), (10, 20), (20, 25), (25, 28), (28, 30), (30, 10**9)]


def human(seconds: float) -> str:
    return f"{seconds:.0f}с" if seconds < 90 else f"{seconds / 60:.1f}мин"


def agg(rows: list) -> dict:
    """S/D/I/N и WER по набору per-utterance записей."""
    S = sum(int(u.get("S", 0)) for u in rows)
    D = sum(int(u.get("D", 0)) for u in rows)
    I = sum(int(u.get("I", 0)) for u in rows)
    N = sum(int(u.get("N", 0)) for u in rows)
    return {"S": S, "D": D, "I": I, "N": N, "n": len(rows),
            "wer_pct": 100 * (S + D + I) / N if N else float("nan")}


def by_duration(per_utt: list) -> dict:
    """Разбивка по корзинам длительности — здесь виден порог обобщения энкодера."""
    out = {}
    for lo, hi in BUCKETS:
        sel = [u for u in per_utt if lo <= float(u.get("duration_seconds", 0)) < hi]
        if sel:
            out[f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"] = agg(sel)
    return out


def le30(per_utt: list) -> dict:
    """Главное подмножество: ≤30 с — родное окно Whisper."""
    return agg([u for u in per_utt if float(u.get("duration_seconds", 0)) <= 30.0])


def gt30(per_utt: list) -> dict:
    return agg([u for u in per_utt if float(u.get("duration_seconds", 0)) > 30.0])


def baseline(split: str, n: int, device: str) -> dict:
    """Эталон bf16 под ТЕМ ЖЕ протоколом (кэш ключуется протоколом)."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from wal_tat.qat.evaluate import evaluate_wer

    tag = f"{MODEL_ID.split('/')[-1]}_{split}_n{n}_b{BATCH_SIZE}_t{MAX_NEW_TOKENS}"
    cache = RESULTS / f"baseline_{tag}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        d["cached"] = True
        return d

    print(f"  … считаю эталон bf16 на {n} записях под тем же протоколом "
          f"(один раз, потом из кэша)", flush=True)
    proc = WhisperProcessor.from_pretrained(MODEL_ID)
    model = (WhisperForConditionalGeneration
             .from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval())
    r = evaluate_wer(model, proc, split, n=n, batch_size=BATCH_SIZE,
                     device=device, max_new_tokens=MAX_NEW_TOKENS)
    per = r.get("per_utterance", []) or []
    out = {"full": {k: r[k] for k in ("S", "D", "I", "N")},
           "full_wer_pct": r["wer"] * 100,
           "le30": le30(per), "gt30": gt30(per), "buckets": by_duration(per),
           "utterances": n, "protocol": {"batch_size": BATCH_SIZE,
                                         "max_new_tokens": MAX_NEW_TOKENS}}
    cache.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    del model
    torch.cuda.empty_cache()
    out["cached"] = False
    return out


def ternary_report(model) -> dict:
    """Проверка, что веса действительно тернарные, + распределение кодов."""
    import torch
    from wal_tat.qat.convert import qat_modules

    alphabet: set[int] = set()
    zeros = plus = minus = total = n_mat = 0
    precisions: set[str] = set()
    for _, mod in qat_modules(model).items():
        precisions.add(getattr(mod.quantizer, "name", "?"))
        # Контрольная ветка 'fp' не имеет кодов: export_codes вернёт сами веса,
        # и приведение их к int дало бы ЛОЖНОЕ «алфавит [-1,0,1]» на
        # полноточной модели.  Такой штамп однажды уедет в историю замеров.
        if "fp" in precisions:
            return {"matrices": len(qat_modules(model)), "weights": 0,
                    "alphabet": [], "is_ternary": False, "precision": "fp",
                    "zero_pct": 0.0, "plus_pct": 0.0, "minus_pct": 0.0}
        codes, _ = mod.export_codes()
        c = codes.reshape(-1).to(torch.int64)
        alphabet.update(torch.unique(c).tolist())
        zeros += int((c == 0).sum())
        plus += int((c == 1).sum())
        minus += int((c == -1).sum())
        total += c.numel()
        n_mat += 1
    return {"matrices": n_mat, "weights": total, "alphabet": sorted(alphabet),
            "is_ternary": sorted(alphabet) in ([-1, 0, 1], [-1, 1]),
            "precision": "/".join(sorted(precisions)),
            "zero_pct": 100 * zeros / max(total, 1),
            "plus_pct": 100 * plus / max(total, 1),
            "minus_pct": 100 * minus / max(total, 1)}


def checkpoint_step(ckpt: Path, info) -> tuple:
    """Реальный шаг берём ИЗ САМОГО чекпоинта, а не из состояния обучения —
    иначе замер подписывается чужим номером шага (эта ошибка уже случалась)."""
    import torch
    step = (info or {}).get("step") if isinstance(info, dict) else None
    val = None
    try:
        meta = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
        extra = meta.get("extra", {}) or {}
        step = extra.get("step", step)
        val = (extra.get("validation") or {}).get("loss")
        del meta
    except Exception:
        pass
    return step, val


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, default=None, help="чекпоинт (по умолчанию best.pt прогона)")
    p.add_argument("--run", default=DEFAULT_RUN, help=f"имя прогона (по умолч. {DEFAULT_RUN})")
    p.add_argument("--n", type=int, default=2703,
                   help="сколько записей (2703 = весь dev-clean; при n<2600 ДЛИННЫХ ЗАПИСЕЙ НЕ БУДЕТ)")
    p.add_argument("--split", default="dev-clean")
    p.add_argument("--device", default="cuda")
    p.add_argument("--history", action="store_true", help="только показать историю")
    args = p.parse_args(argv)

    if args.history:
        if not HISTORY.exists():
            print("Замеров ещё не было.")
            return 0
        print(f"{'когда':<17} {'прогон':<22} {'шаг':>7} {'n':>6} {'≤30с WER':>9} {'отн.':>7} {'>30с':>8}")
        stale = 0
        for line in HISTORY.read_text().splitlines():
            r = json.loads(line)
            if "protocol" not in r:
                # Замеры до закрепления протокола: лимит 128 токенов обрезал вывод
                # и маскировал разнос на длинных записях; подмножество — полный
                # набор, а не ≤30 с.  Сравнивать с новыми НЕЛЬЗЯ.
                stale += 1
                print(f"{r['time'][:16]:<17} {'† старый протокол':<22} "
                      f"{str(r.get('step', '?')):>7} {r['n']:>6} "
                      f"{r.get('wer_pct', float('nan')):>9.3f} "
                      f"{r.get('relative', float('nan')):>7.3f} "
                      f"{'—':>8}")
                continue
            g = r.get("gt30_wer_pct")
            print(f"{r['time'][:16]:<17} {str(r.get('run', '?'))[:22]:<22} "
                  f"{str(r.get('step', '?')):>7} {r['n']:>6} "
                  f"{r.get('le30_wer_pct', float('nan')):>9.3f} "
                  f"{r.get('le30_relative', float('nan')):>7.3f} "
                  f"{(f'{g:.1f}%' if g is not None else '—'):>8}")
        if stale:
            print(f"\n  † {stale} замер(ов) сняты до закрепления протокола: лимит 128 токенов "
                  f"(обрезал вывод\n    и маскировал разнос на длинных записях), подмножество — "
                  f"полный набор, а не ≤30 с.\n    С новыми строками НЕ СРАВНИВАЮТСЯ.")
        return 0

    run_dir = RESULTS / args.run
    ckpt = args.ckpt or (run_dir / "best.pt")
    if not ckpt.exists():
        print(f"❌ Чекпоинт не найден: {ckpt}")
        print(f"   Первый сохраняется на шаге 10000 — подожди и повтори.")
        return 1

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from wal_tat.qat.convert import load_qat_checkpoint
    from wal_tat.qat.evaluate import evaluate_wer

    t0 = time.time()
    mtime = datetime.fromtimestamp(ckpt.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    print("═" * 74)
    print("  ПРОВЕРКА КАЧЕСТВА ТЕРНАРНОЙ МОДЕЛИ")
    print("═" * 74)
    print(f"  прогон:    {args.run}")
    print(f"  чекпоинт:  {ckpt.name}  ({ckpt.stat().st_size / 2**30:.2f} ГБ, {mtime})")
    print(f"  протокол:  {args.split}, n={args.n}, батч {BATCH_SIZE}, "
          f"лимит {MAX_NEW_TOKENS} токенов, greedy")
    if args.n < 2600:
        print(f"  ⚠ при n={args.n} записей >25 с в наборе НЕТ — "
              f"главный дефект будет невиден, бери n=2703")
    print("─" * 74)

    proc = WhisperProcessor.from_pretrained(MODEL_ID)
    model = (WhisperForConditionalGeneration
             .from_pretrained(MODEL_ID, dtype=torch.float32).to(args.device).eval())
    info = load_qat_checkpoint(model, ckpt)
    step, ckpt_val = checkpoint_step(ckpt, info)

    live = None
    for log in (Path("/tmp/qat_small_t3_v2.log"), Path("/tmp/qat_small_t3.log")):
        if log.exists():
            import re
            hits = re.findall(r"^\[step +(\d+)\]", log.read_text(errors="ignore"), re.M)
            if hits:
                live = int(hits[-1])
                break
    if live is not None and step is not None and live > step:
        print(f"  ⚠ обучение сейчас на шаге {live}, чекпоинт — шага {step}")

    if HISTORY.exists():
        for line in HISTORY.read_text().splitlines():
            r = json.loads(line)
            if r.get("run") == args.run and r.get("step") == step and r.get("n") == args.n:
                print(f"  ⚠ этот чекпоинт (шаг {step}) уже мерили — результат будет тем же")

    tern = ternary_report(model)
    if tern.get("precision") == "fp":
        print(f"  ⚪ КОНТРОЛЬНАЯ ВЕТКА: квантования НЕТ (precision='fp', "
              f"{tern['matrices']} матриц в полной точности)")
        print(f"     нужна как знаменатель: цена тернаризации = WER(t3) − WER(fp) на том же шаге")
    else:
        print(f"  {'✅' if tern['is_ternary'] else '❌'} тернарность: алфавит {tern['alphabet']} "
              f"по {tern['weights']:,} весам в {tern['matrices']} матрицах")
        print(f"     коды:  0 → {tern['zero_pct']:.1f}%   +1 → {tern['plus_pct']:.1f}%   "
              f"−1 → {tern['minus_pct']:.1f}%")
    print("─" * 74)

    r = evaluate_wer(model, proc, args.split, n=args.n, batch_size=BATCH_SIZE,
                     device=args.device, max_new_tokens=MAX_NEW_TOKENS)
    per = r.get("per_utterance", []) or []
    t_le30, t_gt30, t_buk = le30(per), gt30(per), by_duration(per)

    del model
    torch.cuda.empty_cache()
    base = baseline(args.split, args.n, args.device)
    b_le30, b_gt30, b_buk = base["le30"], base["gt30"], base["buckets"]

    rel = t_le30["wer_pct"] / b_le30["wer_pct"] if b_le30["wer_pct"] else float("nan")
    quality = 100 / rel if rel == rel and rel else float("nan")

    # ── ГЛАВНАЯ ЦИФРА ────────────────────────────────────────────────────
    print(f"\n  ГЛАВНОЕ — подмножество ≤30 с (родное окно Whisper, {t_le30['n']} записей):")
    print(f"  {'':24} {'WER':>9}   {'S':>6} {'D':>5} {'I':>6}")
    print(f"  {'эталон bf16':24} {b_le30['wer_pct']:>8.4f}%   "
          f"{b_le30['S']:>6} {b_le30['D']:>5} {b_le30['I']:>6}")
    print(f"  {'ТЕРНАРЬ (1.58 бит)':24} {t_le30['wer_pct']:>8.4f}%   "
          f"{t_le30['S']:>6} {t_le30['D']:>5} {t_le30['I']:>6}")
    print(f"  ОТНОСИТЕЛЬНЫЙ WER:  {rel:.4f}×  ± {NOISE_FLOOR_PP / b_le30['wer_pct']:.4f} "
          f"(шум eval)      цель ≈ 1.20×")
    print(f"  СОХРАНЕНО КАЧЕСТВА: {quality:.1f}%                        цель ≈ 80–85%")

    # ── ДИАГНОСТИКА: где порог обобщения ─────────────────────────────────
    print(f"\n  ДИАГНОСТИКА — по длительности входа (порог обобщения энкодера):")
    print(f"  {'корзина':>10} {'зап':>5} {'bf16':>9} {'тернарь':>9} {'отн.':>7} {'вставок':>8}")
    for k in t_buk:
        t, b = t_buk[k], b_buk.get(k)
        if not b or not b["N"]:
            continue
        rr = t["wer_pct"] / b["wer_pct"] if b["wer_pct"] else float("nan")
        flag = "  ← разнос" if rr > 3 else ("  ← слабо" if rr > 1.8 else "")
        print(f"  {k + 'с':>10} {t['n']:>5} {b['wer_pct']:>8.2f}% {t['wer_pct']:>8.2f}% "
              f"{rr:>7.2f} {t['I']:>8}{flag}")

    # ── ДЛИННЫЕ ЗАПИСИ ───────────────────────────────────────────────────
    if t_gt30["n"]:
        print(f"\n  ЗАПИСИ >30 с ({t_gt30['n']} шт) — здесь аудио ОБРЕЗАНО окном, "
              f"обе цифры искусственно плохие:")
        print(f"     эталон bf16 {b_gt30['wer_pct']:.2f}%   тернарь {t_gt30['wer_pct']:.2f}%")
        print(f"     ⚠ сравнивать их между собой НЕЛЬЗЯ: учитель в этом режиме тоже сломан")
        print(f"     настоящий long-form — нарезкой по 29 с (НЕ 30: ровно на 30.000 с "
              f"вход без паддинга,")
        print(f"     которого не было в обучении, и модель начинает повторяться). "
              f"См. longform_root_cause.json")

    vtxt = f", val-loss {ckpt_val:.4f}" if ckpt_val is not None else ""
    print(f"\n  ЧЕКПОИНТ ШАГА: {step if step is not None else '?'}{vtxt}")

    # ── СРАВНЕНИЕ С ЯКОРЕМ ───────────────────────────────────────────────
    if step == ANCHOR["step"] and args.run != ANCHOR["run"] and args.n == 2703:
        d = t_le30["wer_pct"] - ANCHOR["le30_wer_pct"]
        verdict = ("НИЧЕГО НЕ ИЗМЕНИЛОСЬ (в пределах шума)" if abs(d) < 0.02
                   else ("ЛУЧШЕ" if d < 0 else "ХУЖЕ"))
        print(f"\n  СРАВНЕНИЕ НА ОДНОМ ШАГЕ с прогоном №1 ({ANCHOR['run']}, шаг {ANCHOR['step']}):")
        print(f"     данные №1: {ANCHOR['data']}")
        print(f"     ≤30 с: {ANCHOR['le30_wer_pct']:.4f}% → {t_le30['wer_pct']:.4f}%  "
              f"({d:+.4f} п.п.)  {verdict}")
        print(f"     >30 с: {ANCHOR['gt30_wer_pct']:.2f}% → {t_gt30['wer_pct']:.2f}%")
        print(f"     между прогонами менялись ТОЛЬКО ДАННЫЕ — код, расписание, "
              f"гиперпараметры совпадают")

    print(f"\n  заняло {human(time.time() - t0)}")
    print("═" * 74)

    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a") as f:
        f.write(json.dumps({
            "time": datetime.now().isoformat(timespec="seconds"),
            "run": args.run, "ckpt": str(ckpt), "step": step,
            "split": args.split, "n": args.n,
            "protocol": {"batch_size": BATCH_SIZE, "max_new_tokens": MAX_NEW_TOKENS},
            "le30_wer_pct": t_le30["wer_pct"], "le30_baseline_pct": b_le30["wer_pct"],
            "le30_relative": rel, "quality_pct": quality,
            "gt30_wer_pct": t_gt30["wer_pct"] if t_gt30["n"] else None,
            "full_wer_pct": r["wer"] * 100,
            "buckets": t_buk, "ternary": tern["is_ternary"], "zero_pct": tern["zero_pct"],
        }, ensure_ascii=False) + "\n")
    print(f"  записано в {HISTORY.name}   (история:  --history)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
