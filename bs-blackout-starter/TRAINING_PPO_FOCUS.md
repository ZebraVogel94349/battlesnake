# V23: PPO-Fokus mit Retention

V22 reservierte rund 35 % seiner Gegnerverteilung fest für `best`, `hunter`,
`snake25`, `snake25_duelist` und `random`. Dieser Anteil blieb auch dann
erhalten, wenn die Winrates bereits sehr hoch waren. PPO optimiert den
gewichteten Erwartungswert: Viele einfache, weiterhin leicht verbesserbare
Partien können deshalb kleinere, aber wichtigere Verschlechterungen gegen
neuronale Gegner überstimmen.

V23 startet bewusst vom vollständigen V21-Checkpoint bei Update 87.500. Dessen
bekannte Ausgangswerte sind ungefähr 95,7 % gegen Hunter, 88,7 % gegen Snake-25
und 82,0 % im Snake-25-Duell. Diese Fähigkeiten werden nicht aufgegeben:

- Heuristiken bleiben mit zusammen 4 % als Rehearsal-Daten im Training.
- Nur 10 % statt 25 % der Spiele sind echte Duelle.
- Promotions müssen weiterhin mindestens 90 % gegen Hunter, 83 % gegen
  Snake-25 und 76 % gegen den Duellisten erreichen.
- Der Snake-25-Behavioral-Clone bleibt zusätzlich als neuronaler Anchor aktiv.

Die übrigen 96 % Grundgewicht gehören neuronalen Gegnern. V18, V19, V20, der
alte Champion und vier repräsentative V21-Snapshots sind feste Anchors. PFSP
verstärkt Anchors unterhalb ihres Zielwerts; Nash verteilt den verbleibenden
Anteil anhand aktueller Evaluationen. Elf aktive Modelle verhindern, dass ein
schwerer Anchor wegen Rotation vorübergehend ganz aus dem Training fällt.

Die Evaluation verwendet bei jedem Messpunkt dieselben gepaarten Seeds. V21
und V22 verschoben den Seed-Block bei jeder Evaluation; deren Kurven enthalten
daher zusätzliches Suite-Rauschen und können kurzfristig wie Regression
aussehen. Die Trainingsumgebungen bleiben weiterhin zufällig und unabhängig.

## Start

```bash
cd bs-blackout-starter
./train_v23_ppo_focus.sh --check

nohup ./train_v23_ppo_focus.sh > train_v23_ppo_focus.log 2>&1 &
echo $! > v23_ppo_focus.pid
```

Nach Unterbrechungen setzt derselbe Befehl automatisch den neuesten
vollständigen V23-Checkpoint fort. Der geschützte Export ist
`ppo_bs_lstm_cuda_v23_ppo_focus_champion.zip`.

Die festen Evaluationen sind entscheidend, nicht der gemischte Rollout-Reward:

```bash
grep -E '^\[pfsp\]|^\[nash\]|^\[league\]|^\[eval\].*(v18|v19|v20|history|hunter|snake25)' \
  train_v23_ppo_focus.log | tail -n 80
```
