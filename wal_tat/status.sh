#!/bin/bash
# WAL-TAT ternary QAT — статус прогона.  Запуск:  bash wal_tat/status.sh
#
# Прогон №2: код, расписание и гиперпараметры те же, что в №1 — меняются
# ТОЛЬКО ДАННЫЕ (упакованные 30-секундные окна вместо сырых реплик).
# Это контролируемый опыт, поэтому сверка идёт на одинаковом шаге 30000.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"/wal_tat
LOG=${LOG:-/tmp/qat_small_t3_v2.log}
RUN=${RUN:-small_t3_packed_v2}
OUT=$ROOT/results/qat/$RUN
TOTAL=120000
ANCHOR=30000          # шаг, на котором у прогона №1 есть чекпойнт для сравнения

echo "══════════════════════════════════════════════════════════════════"
echo "  WHISPER-SMALL → 100% ТЕРНАРЬ (T3, 1.58 бит)   прогон: $RUN"
echo "══════════════════════════════════════════════════════════════════"

N=$(pgrep -fc qat_train.py)
if [ "$N" -gt 0 ]; then echo "  Статус:   🟢 ИДЁТ ОБУЧЕНИЕ"
elif grep -q '\[DONE\]' "$LOG" 2>/dev/null; then echo "  Статус:   ✅ ЗАВЕРШЕНО"
else echo "  Статус:   🔴 НЕ ЗАПУЩЕНО / УПАЛО"; fi

S=$(grep -oP '^\[step +\K[0-9]+' "$LOG" 2>/dev/null | tail -1)
SEC=$(grep -oP '^\[step.*\s\K[0-9.]+(?=s$)' "$LOG" 2>/dev/null | tail -1)
if [ -n "$S" ]; then
  PCT=$(awk "BEGIN{printf \"%.1f\", $S*100/$TOTAL}")
  ETA=$(awk "BEGIN{printf \"%.1f\", ($TOTAL-$S)*${SEC:-0.52}/3600}")
  BAR=$(awk "BEGIN{n=int($S*46/$TOTAL); for(i=0;i<n;i++)printf \"█\"; for(i=n;i<46;i++)printf \"░\"}")
  echo "  Шаг:      $S / $TOTAL   ($PCT%)     ${SEC}с/шаг"
  echo "  [$BAR]"
  echo "  Осталось: ~${ETA} ч"
  if [ "$S" -lt "$ANCHOR" ]; then
    A=$(awk "BEGIN{printf \"%.1f\", ($ANCHOR-$S)*${SEC:-0.52}/3600}")
    echo "  До сверки с прогоном №1 (шаг $ANCHOR): ~${A} ч"
  fi
fi

echo "──────────────────────────────────────────────────────────────────"
echo "  ДАННЫЕ — в этом и состоит единственное отличие от прогона №1:"
SUM=$ROOT/cache/qat/features/whisper-small-packed30-955h.summary.json
[ -f "$SUM" ] && python3 -c "
import json;d=json.load(open('$SUM'))
g=d['group_seconds'];l=d['label_tokens']
print(f\"    окон {d['num_train']:,} из {d['num_source_utterances']:,} реплик, {d['total_hours']:.0f} ч\")
print(f\"    длительность окна: медиана {g['median']}с  p90 {g['p90']}с  макс {g['max']}с\")
print(f\"    заполненность {100*d['audio_fill_ratio']:.1f}%   метка: медиана {l['median']:.0f}  p99 {l['p99']:.0f}  макс {l['max']:.0f} ток\")
print(f\"    было в №1:  медиана 13.9с,  окон >=25с — 0.0%  (порог отказа ~25с)\")" 2>/dev/null

echo "──────────────────────────────────────────────────────────────────"
echo "  ПОСЛЕДНИЕ ШАГИ (loss / доля нулей / переворотов кодов):"
grep '^\[step' "$LOG" 2>/dev/null | tail -4 | sed 's/^/    /'

V=$(grep '^\[valid' "$LOG" 2>/dev/null | tail -5)
if [ -n "$V" ]; then
  echo "──────────────────────────────────────────────────────────────────"
  echo "  VALIDATION (отложенные дикторы):"
  echo "$V" | sed 's/^/    /'
fi

WERF=$(ls -t $OUT/wer_step*.json 2>/dev/null | head -4)
if [ -n "$WERF" ]; then
  echo "──────────────────────────────────────────────────────────────────"
  echo "  WER ПО ХОДУ ОБУЧЕНИЯ (быстрый eval; полная проверка — check_wer.sh):"
  for f in $WERF; do python3 -c "
import json;d=json.load(open('$f'))
print(f\"    шаг {d.get('step','?'):>7}:  WER {d.get('wer',0)*100:.3f}%   (n={d.get('utterances','?')})\")" 2>/dev/null; done
fi

echo "──────────────────────────────────────────────────────────────────"
echo "  ЯКОРЬ — прогон №1 (неупакованные данные), чекпойнт шага 30000:"
echo "    ≤30 с: 4.2275%  (отн. 1.2366× ± 0.004)   >30 с: 148.25%"
echo "    сохранено качества: 80.9%   эталон bf16 закэширован на 3.4187%"
echo "    протокол: батч 16, лимит 440 токенов, greedy;  шум eval ±0.013 п.п."
echo "──────────────────────────────────────────────────────────────────"
echo "  GPU 2:"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader -i 2 2>/dev/null | sed 's/^/    /'
echo "  Чекпоинты:"; ls -lh $OUT/*.pt 2>/dev/null | awk '{print "    "$9" — "$5" ("$6" "$7" "$8")"}'
echo "══════════════════════════════════════════════════════════════════"
echo "  полная проверка:  bash wal_tat/check_wer.sh"
echo "  история замеров:  bash wal_tat/check_wer.sh --history"
echo "  живой лог:        tail -f $LOG"
