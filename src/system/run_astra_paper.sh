#!/bin/bash
# Reprodução estilo artigo do ASTRA-FL sobre PUMA (CNN + partição ASTRA-FL, SEM concept drift).
# Mantido estritamente separado de `astra_drift_*` (resultados em `src/results/astra_paper_*`).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

ALPHA="${DIRICHLET_ALPHA:-0.1}"
P_SEED="${PARTITION_SEED:-42}"
ROUNDS="${ROUNDS:-50}"
CLIENTS="${CLIENTS_PER_ROUND:-5}"
MAX_CLIENTS="${MAX_CLIENTS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"
SIMULATIONS="${SIMULATIONS:-1}"
EXP_NAME="${EXP_NAME:-astra_paper_cifar10_a${ALPHA}_s${P_SEED}}"

./run.sh \
    --partition-source astra \
    -d Cifar10 \
    --dirichlet-alpha "$ALPHA" \
    --partition-seed "$P_SEED" \
    --model cnn \
    --astra \
    --rounds "$ROUNDS" \
    --clients "$CLIENTS" \
    --max-clients "$MAX_CLIENTS" \
    --batch-size "$BATCH_SIZE" \
    --prune 0 \
    --ala 1 \
    --no-quant \
    --global-eval official \
    --deterministic-selection \
    --train-augment astra \
    --seed "$SEED" \
    --simulations "$SIMULATIONS" \
    --exp-name "$EXP_NAME" \
    --auto-next \
    "$@"

