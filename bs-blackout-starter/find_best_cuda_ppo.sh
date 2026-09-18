#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
COMPARE_SCRIPT="${COMPARE_SCRIPT:-$SCRIPT_DIR/compare_ppo_cuda.py}"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v7_variants}"
REFERENCE_MODEL="${REFERENCE_MODEL:-$SCRIPT_DIR/ppo_bs_lstm_cuda_steps_000655360000_u00005000.zip}"

for path in "$PYTHON" "$COMPARE_SCRIPT" "$REFERENCE_MODEL"; do
    if [[ ! -f "$path" ]]; then
        echo "Datei nicht gefunden: $path" >&2
        exit 1
    fi
done

shopt -s nullglob
models=("$MODEL_DIR"/*.zip)
if (( ${#models[@]} == 0 )); then
    echo "Keine .zip-Modelle gefunden in: $MODEL_DIR" >&2
    exit 1
fi

best_model=""
best_rate="-1"

for model in "${models[@]}"; do
    echo
    echo "Vergleiche $(basename -- "$model") mit $(basename -- "$REFERENCE_MODEL")"

    # Das Variantenmodell ist immer model_a, daher ist a_win_rate die gesuchte Rate.
    output="$($PYTHON "$COMPARE_SCRIPT" "$model" "$REFERENCE_MODEL" "$@")"
    printf '%s\n' "$output"

    rate="$(awk '/: wins=.* win_rate=/ && !seen { sub(/^.*win_rate=/, ""); print $1; seen=1 }' <<< "$output")"
    if [[ -z "$rate" ]]; then
        echo "Konnte Win-Rate nicht auslesen für: $model" >&2
        exit 1
    fi

    if awk -v current="$rate" -v best="$best_rate" 'BEGIN { exit !(current > best) }'; then
        best_rate="$rate"
        best_model="$model"
    fi
done

echo
echo "Bestes Modell: $best_model"
echo "Win-Rate gegen Referenz: $best_rate"
