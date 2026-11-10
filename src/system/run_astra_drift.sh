#!/bin/bash
# Experimento ASTRA-FL + Concept Drift do PUMA (resultados em `src/results/astra_drift_*`).
# Separado da reprodução do artigo (`run_astra_paper.sh`). Registra G-orig, G-drift e C-mix.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

ALPHA="${DIRICHLET_ALPHA:-0.1}"
P_SEED="${PARTITION_SEED:-42}"
DRIFT_SEED="${DRIFT_SEED:-1234}"
DRIFT_TRANSFORM="${DRIFT_TRANSFORM:-label_swap}"
DRIFT_TYPE="${DRIFT_TYPE:-abrupt}"
DRIFT_ROUND="${DRIFT_ROUND:-25}"
DRIFT_LENGTH="${DRIFT_LENGTH:-5}"
DRIFT_FREQ="${DRIFT_FREQ:-0}"
DRIFT_CLIENTS="${DRIFT_CLIENTS:-0-9}"
ROUNDS="${ROUNDS:-50}"
CLIENTS="${CLIENTS_PER_ROUND:-5}"
MAX_CLIENTS="${MAX_CLIENTS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"
SIMULATIONS="${SIMULATIONS:-1}"

DC_TAG="${DRIFT_CLIENTS//,/_}"
EXP_NAME="${EXP_NAME:-astra_drift_cifar10_a${ALPHA}_${DRIFT_TRANSFORM}_${DRIFT_TYPE}_dc${DC_TAG}}"

./run.sh \
    --partition-source astra \
    -d Cifar10 \
    --dirichlet-alpha "$ALPHA" \
    --partition-seed "$P_SEED" \
    --drift \
    --drift-transform "$DRIFT_TRANSFORM" \
    --drift-type "$DRIFT_TYPE" \
    --drift-round "$DRIFT_ROUND" \
    --drift-length "$DRIFT_LENGTH" \
    --drift-freq "$DRIFT_FREQ" \
    --drift-clients "$DRIFT_CLIENTS" \
    --drift-seed "$DRIFT_SEED" \
    --model cnn \
    --astra \
    --rounds "$ROUNDS" \
    --clients "$CLIENTS" \
    --max-clients "$MAX_CLIENTS" \
    --batch-size "$BATCH_SIZE" \
    --prune 0 \
    --ala 1 \
    --no-quant \
    --global-eval both \
    --deterministic-selection \
    --blend-gradual-eval \
    --train-augment astra \
    --seed "$SEED" \
    --simulations "$SIMULATIONS" \
    --exp-name "$EXP_NAME" \
    --auto-next \
    "$@"

