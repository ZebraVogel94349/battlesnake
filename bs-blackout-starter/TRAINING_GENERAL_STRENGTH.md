# V24: allgemeine Spielstärke

## Befund aus V23

V23 lief technisch stabil, hat aber keine allgemeine Verbesserung erzeugt.
Zwischen Update 87.750 und 90.000 lag der mittlere Match-Score gegen die
neuronalen Fixanker nur zwischen 0,460 und 0,484. Der geschützte Champion
wurde kein einziges Mal ersetzt.

Einzelne Ausreißer waren nicht allgemein besser. Beispielsweise erreichte
Update 89.750 gegen Update 87.500 einen Score von 0,541, hatte über die
neuronalen Fixanker aber nur 0,476 und bestand den direkten Mehrlayout-Test
gegen den Champion mit 0,498 nicht. Das ist das typische Bild einer
zyklischen Gegenstrategie.

Die Ursachen waren:

1. Nash und PFSP bewerteten jeweils eine benannte Policy. Im Training wurden
   die drei Gegnersitze anschließend unabhängig gezogen. Die so entstehenden
   Mischlineups entsprachen weder der Nash-Spalte noch den homogenen
   Evaluationen gegen V18, V19, V20 und Snapshots.
2. Der vollständige Adam-Zustand von Update 87.500 wurde verworfen und die
   Lernrate gleichzeitig von `1e-5` auf `1.5e-5` erhöht.
3. Die Entropie stieg trotz reifer Ausgangspolicy von ungefähr 0,27 auf 0,34.
4. Nash konzentrierte sich stark auf `old_champion`, V18 und V19, während V20
   und manche Snapshots nach dem Solve nur ungefähr ein Prozent
   Trainingsgewicht bekamen. `league_champion` war dabei nicht derselbe
   starke Champion, sondern das schwächere Startmodell von Update 87.500.
5. Update 87.500 war nicht das stärkste allgemein bekannte Modell: Über viele
   V21-Evaluationen erzielte es gegen den geschützten V21-Champion nur ungefähr
   0,40 bis 0,46 Match-Score.

## Was V24 anders macht

- 85 Prozent der Vier-Spieler-Episoden verwenden eine einzige gezogene
  Gegnerpolicy auf allen drei Gegnersitzen. Die übrigen 15 Prozent behalten
  gemischte Lineups zur Robustheit.
- Der Learner startet deshalb vom stärkeren geschützten V21-Champion. Update
  87.500 bleibt als permanenter Retention-Gegner erhalten, damit dessen
  spätere Strategien nicht aus der Trainingsverteilung verschwinden.
- Für den Champion existiert kein passender vollständiger Optimizer-Payload.
  Der notwendige frische Optimizer beginnt deshalb sehr konservativ bei
  `3e-6`; PPO-Clip ist 0,08 und Ziel-KL 0,0020.
- Der Actor hat eine eigene Gradientenbegrenzung von 0,25.
- Der Entropiekoeffizient fällt von 0,0003 auf 0,00005.
- Update 87.500, V18, V19, V20 und vier V21-Snapshots sind permanente
  Retention-Anker. Snake25-BC bleibt ebenfalls im Pool.
- 32 Prozent Gesamtmasse sind gleichmäßig für die normalen Fixanker
  reserviert; 20 Prozent sind für den jeweils besten Champion reserviert.
  Der Rest wird adaptiv über Nash und PFSP verteilt.
- Hunter, Snake25 und Snake25-Duelist haben zusammen nur drei Prozent
  garantierte Grundmasse. Bei einem echten Einbruch darf Nash ihnen mehr
  zuweisen.
- Der alte starke V21-Champion ist nun sowohl Learner-Ausgangspunkt als auch
  anfänglicher deploybarer Champion und wird nicht zusätzlich unter einem
  zweiten Ankerlabel geführt.
- Ein Kandidat muss den Champion über drei Seeds und die Layouts `copies`,
  `solo-pair` und `true-duel` schlagen und gleichzeitig alle neuronalen und
  heuristischen Mindestwerte halten.
- Zusätzlich wird der Champion einmal auf exakt derselben festen neuronalen
  Benchmark-Suite gemessen. Ein Kandidat braucht dort mindestens +0,005
  mittlere Verbesserung und darf gegen keinen einzelnen Anker mehr als 0,025
  verlieren. Ein bloßer zyklischer Head-to-Head-Counter kann dadurch nicht
  mehr zum allgemein besten Modell erklärt werden.

## Start auf dem Server

Den alten V23-Prozess zuerst beenden und kontrollieren, dass kein
`train_cuda.py` mehr daraus läuft. Danach:

```bash
cd bs-blackout-starter
./train_v24_general_strength.sh --check

nohup ./train_v24_general_strength.sh \
  > train_v24_general_strength.log 2>&1 &

echo $! > v24_general_strength.pid
```

Dasselbe Skript setzt nach einer Unterbrechung automatisch den neuesten
vollständigen V24-Checkpoint fort.

Falls der Abbruch unmittelbar nach einer Champion-Beförderung und vor dem
nächsten periodischen Trainingscheckpoint passiert, darf der geschützte
Champion neuer als der Resume-Checkpoint sein. Das ist beabsichtigt: Der
Learner wird mit dem letzten vollständigen Optimizerzustand zurückgesetzt,
während der neuere Champion als Deploymentmodell und Trainingsgegner erhalten
bleibt. Ein zum Resume-Checkpoint zu neuer Nash-Zustand wird automatisch auf
den jüngsten passenden Historienstand zurückgerollt.

Vor einer Fortsetzung zuerst kontrollieren, dass kein alter Prozess mehr läuft.
Das Log wird beim Resume angehängt, damit die bisherige Historie erhalten
bleibt:

```bash
cd bs-blackout-starter
pgrep -af '[t]rain_cuda.py' || true
./train_v24_general_strength.sh --check

nohup ./train_v24_general_strength.sh \
  >> train_v24_general_strength.log 2>&1 &

echo $! > v24_general_strength.pid
```

Die V24-Updatezähler beginnen wieder bei 51.250, weil sie die Provenienz des
tatsächlichen Learner-Ausgangsmodells abbilden. Das ist beabsichtigt; Update
87.500 bleibt als `source_87500` im Pool und wird nicht überschrieben.

## Kontrolle

```bash
tail -f train_v24_general_strength.log
```

Bei 2.048 Environments sollte `copy_lineups` meistens ungefähr im Bereich
1.500 bis 1.650 liegen. Das bestätigt, dass die korrigierte Lineup-Verteilung
wirklich aktiv ist. KL sollte überwiegend unter etwa 0,0030 bleiben und die
Entropie nicht mehr systematisch nach oben laufen.

Eine echte Modellverbesserung ist erst mit einer Zeile wie dieser bestätigt:

```text
[league] PROMOTED update=... direct_score=... guard_score=...
```

Unmittelbar davor muss außerdem `league-general ... passed=1` stehen.

Der jeweils beste geschützte Export liegt danach hier:

```text
ppo_bs_lstm_cuda_v24_general_strength_champion.zip
```

Ein fallender Learner-Einzelwert ist nicht automatisch ein Rückschritt des
deploybaren Modells: Solange kein Kandidat alle Tests besteht, bleibt der
vorherige Champion unverändert erhalten.
