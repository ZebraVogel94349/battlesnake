# Effektive Fortsetzung nach V21

## Diagnose

Die heruntergeladene V21-Fortsetzung rechnet schnell, lernt aber nicht in die
richtige Richtung:

- Der Phasenlauf startete von Update 78.500 mit einer geplanten Länge von
  40.000 Updates. Der Scheduler verwendete trotzdem den absoluten Zähler. Weil
  78.500 bereits größer als 40.000 war, starteten Lernrate und
  Entropieregularisierung sofort an ihren Endwerten: `5e-6` und `0.0015` statt
  `5e-5` und `0.003`.
- Der spätere manuelle Lauf mit `1e-5` erzeugte zwar größere PPO-Updates
  (`KL` etwa `0.0032` statt `0.0014`), aber keinen messbaren Fortschritt.
- Zwischen Update 81.500 und 87.500 erreichten die Kandidaten im Mittel nur
  `0.465` gegen den unveränderten Champion. Keine der 13 Prüfungen führte zu
  einer Promotion. Gegen den Champion lagen die festen Evaluationen meist nur
  bei 39–46 %, gegen `v18_final` meist bei 37–45 %.
- `pool-anchor-uniform-floor=0.30` traf zusammen mit nur vier aktiven neuronalen
  Gegnern einen ungünstigen Sonderfall: Nach Champion, Snake-25-Klon und
  neuestem Snapshot blieb nur ein rotierender Anchor übrig. Dieser eine Anchor
  bekam jeweils die ganzen 30 % Mindestgewicht, während der durch Nash als
  schwerste erkannte Gegner nur zeitweise aktiv war.

## Neue Fortsetzung

`train_v22_effective.sh` lässt V21 vollständig unangetastet und schreibt in
eigene `v22_effective`-Pfade. Der Lauf:

- startet mit frischem Optimizer und frischem Nash-Zustand vom unveränderten
  Champion bei Update 51.250; dieser ist nach den vorhandenen
  Mehrfach-Evaluationen klar stärker als alle späten V21-Checkpoints;
- verwendet einen phasenlokalen LR-/Entropieplan, der auch nach Unterbrechungen
  korrekt an derselben Stelle weiterläuft;
- hält sechs statt vier neuronale Gegner aktiv, entfernt den konzentrierenden
  Anchor-Floor und reserviert mindestens 20 % der Verteilung für den Champion;
- lässt den restlichen neuronalen Anteil durch aktuelle Nash-Ergebnisse auf die
  wirklich schweren Gegner verteilen;
- stoppt PPO-Minibatches bei zu großer KL und behält den bisherigen Champion,
  solange der Kandidat ihn nicht mit mindestens `0.525` über neun gepaarte
  Tests aus vier-, zwei-und-zwei- sowie echten Duell-Layouts schlägt;
- nutzt auf dem RTX-4090-Profil CUDA Graphs nur für den inference-only Rollout.
  Die getrennten Actor-CNN-, Critic-CNN- und PPO-Loss-Aufrufe behalten
  Inductor-Fusion, verwenden wegen ihres gemeinsamen Backwards aber keine CUDA
  Graphs. Dadurch entfällt die Warnung aus `cudagraph_trees.py`, ohne den
  kompletten Compiler oder den schnellen Rollout abzuschalten.

## Start auf dem GPU-Server

Zuerst nur die Eingaben prüfen:

```bash
cd bs-blackout-starter
./train_v22_effective.sh --check
```

Dann starten:

```bash
nohup ./train_v22_effective.sh > train_v22_effective.log 2>&1 &
echo $! > v22_effective.pid
```

Nach einer Unterbrechung wird mit exakt demselben Befehl automatisch der
neueste vollständige V22-Checkpoint gewählt. Checkpoints entstehen alle 100
Updates. Ein expliziter Pfad kann bei Bedarf mit
`V22_RESUME_PATH=/pfad/zum/checkpoint.pt` gewählt werden.

Der erste echte Update-Eintrag eines frischen Laufs soll ungefähr
`lr=2.50e-05` und `ent_coef=0.0035` zeigen. Außerdem muss davor stehen:

```text
[schedule] phase_origin_update=51250 phase_completed_updates=0
```

Wichtiger als der schwankende Rollout-Reward sind diese Zeilen:

```bash
grep -E '^\[league\]|^\[eval\].*(league_champion|v18_final|v19_final)' \
  train_v22_effective.log | tail -n 40
```

Eine erfolgreiche Verbesserung wird als `[league] PROMOTED ...` protokolliert.
Der jederzeit sicher einsetzbare Export ist
`ppo_bs_lstm_cuda_v22_effective_champion.zip`; er wird nur nach bestandener
Promotion ersetzt. `models/ppo_bs_lstm_cuda_v22_effective_latest.pt` ist der
vollständige Trainingszustand und kann zwischen Prüfungen vorübergehend
schlechter sein.
