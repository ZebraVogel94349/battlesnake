# V25: gezielter Abschlusslauf für zehn Stunden

## Ergebnis des Final-Sprints

Der Lauf von Update 75.250 bis 77.250 ist vollständig und fehlerfrei
durchgelaufen, hat aber keine Policy befördert. Update 75.250 blieb deshalb
korrekt als geschützter Champion erhalten.

| Update | direkt gegen 75.250 | feste Suite | schwächster Anchor | Heuristiken |
|-------:|--------------------:|------------:|-------------------:|------------:|
| 75.500 | 0,506 | 0,511 | 0,475 | 0,720 |
| 76.250 | **0,508** | 0,520 | 0,475 | 0,707 |
| 76.750 | 0,502 | **0,539** | **0,490** | **0,741** |
| 77.000 | 0,494 | 0,532 | 0,496 | 0,719 |
| 77.250 | 0,494 | 0,514 | 0,477 | 0,731 |

Update 76.750 ist der beste Generalist: Gegen die feste Suite liegt er im
Mittel `+0,016` über Update 75.250 und gegen die Heuristiken ungefähr `+0,022`
über dem Recovery-Stand. Die Gewinne verteilen sich auf fast alle V24-Anker.
Nur drei gemessene Defizite bleiben:

- `source_87500`: `0,590` statt `0,622`;
- `v24_champion_69900`: `0,490` statt `0,518`;
- `v24_early_56251`: `0,548` statt `0,568`.

Die späteren Updates verloren wieder direkten Score. Deshalb wird weder vom
letzten Update 77.250 noch nochmals von 75.250 gestartet.

## Targeted Finish

`train_v25_elite_pressure.sh` startet einen neuen Branch
`v25_targeted_finish` von Update 76.750. Update 76.750 ist zugleich die feste
Actor-Referenz; Update 75.250 bleibt der geschützte League-Champion und damit
die Messlatte für jede Beförderung.

Konfiguration:

- 3.250 Updates, Zielupdate 80.000;
- erwartete Laufzeit etwa neun bis zehn Stunden;
- frischer Adam-Optimizer;
- Cosine-Lernrate `9,0e-7 -> 2,5e-7`;
- Actor-Referenz-L2 `0,025` zur Erhaltung der Generalistengewinne;
- PFSP-Ziele entsprechen nun den tatsächlich gemessenen Scores von Update
  75.250. Dadurch erhalten genau die drei Regressionen zusätzlichen Druck;
- 30 Prozent gleichmäßige Anchor-Masse und 30 Prozent Mindestmasse für den
  geschützten Champion;
- 28 Prozent echte Duelle, davon 45 Prozent direkt gegen Update 75.250;
- weitere gezielte Duelle gegen V24-Champion, frühen V24-Anker und
  `source_87500`;
- Promotion ab `0,510` direkt gegen Update 75.250, mindestens `+0,003` im
  Mittel der festen Suite und höchstens `-0,015` auf einem einzelnen Anchor.
- höchstens 12 resumierbare Varianten und 32 Poolmodelle; dadurch bleibt der
  zusätzliche Speicherbedarf des Laufs begrenzt.

Der niedrigere Lernimpuls als im Final-Sprint soll die bereits beobachteten
Oszillationen reduzieren. Der Actor-Anker schützt diesmal Update 76.750 und
nicht 75.250, damit dessen breite Gewinne nicht erneut herausoptimiert werden.

## Upload und Start

Auf den Server laden:

```text
train_cuda.py
train_v25_elite_pressure.sh
```

Dann prüfen:

```bash
cd /tmp/hisss/bs-blackout-starter
pgrep -af '[t]rain_cuda.py'
./train_v25_elite_pressure.sh --check
```

Bei einem frischen Lauf muss die Prüfung melden:

```text
[v25] validated complete source update=76750, V25 league baseline update=75250, protected V24 champion update=69900, ...
[v25] branching targeted finish from update 76750 with fresh Adam state
[v25] phase schedule: update 76750 -> 80000
```

Start:

```bash
nohup ./train_v25_elite_pressure.sh \
  > train_v25_targeted_finish.log 2>&1 &

echo $! > v25_targeted_finish.pid
tail -f train_v25_targeted_finish.log
```

Beim Start muss außerdem erscheinen:

```text
[actor-reference] loaded Torch checkpoint update=76750 coef=0.025 initial_l2=0.000000
[league] ... champion ... update=75250
```

Eine akzeptierte Verbesserung benötigt beide Zeilen:

```text
[league-general] ... passed=1
[league] PROMOTED update=... direct_score=... guard_score=...
```

Der geschützte Endstand liegt hier:

```text
ppo_bs_lstm_cuda_v25_targeted_finish_champion.zip
```

Für den Einsatz immer die Datei mit `_champion` verwenden. Falls der Lauf
keinen Kandidaten durch alle Gates bringt, enthält sie weiterhin den sicheren
Stand Update 75.250 und nicht den schwächeren letzten Learner.
