#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

DATASET="${DATASET:-Cifar100}"
ALPHA="${ALPHA:-0.1}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
PARTITION_SEED="${PARTITION_SEED:-42}"
DRIFT_TRANSFORM="${DRIFT_TRANSFORM:-label_swap}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
CLIENTS_PER_ROUND="${CLIENTS_PER_ROUND:-5}"
MAX_CLIENTS="${MAX_CLIENTS:-20}"
SEED="${SEED:-42}"
ROUNDS="${ROUNDS:-50}"
SIMULATIONS="${SIMULATIONS:-1}"

DS_TAG="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')"
EXP_ABRUPT="${EXP_ABRUPT:-clip_astra_${DS_TAG}_a${ALPHA}_e${LOCAL_EPOCHS}_abrupt_drift}"
EXP_GRADUAL="${EXP_GRADUAL:-clip_astra_${DS_TAG}_a${ALPHA}_e${LOCAL_EPOCHS}_gradual_drift}"

echo "================================================================="
echo " INICIANDO BATERIA GRADUAL ($EXP_GRADUAL)"
echo " Data/Hora: $(date)"
echo "================================================================="

./run.sh \
    --partition-source astra \
    --dirichlet-alpha "$ALPHA" \
    --partition-seed "$PARTITION_SEED" \
    --drift-transform "$DRIFT_TRANSFORM" \
    -d "$DATASET" \
    --model clip \
    --clients "$CLIENTS_PER_ROUND" \
    --max-clients "$MAX_CLIENTS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --local-epochs "$LOCAL_EPOCHS" \
    --rounds "$ROUNDS" \
    --simulations "$SIMULATIONS" \
    --seed "$SEED" \
    --drift \
    --drift-type gradual \
    --drift-freq 5 \
    --drift-clients all \
    --global-eval both \
    --deterministic-selection \
    --blend-gradual-eval \
    --exp-name "$EXP_GRADUAL" \
    --auto-next

echo "[INFO] Gerando gráficos para $EXP_GRADUAL..."
uv run python plot/plot_astra_benchmark.py --exp-name "$EXP_GRADUAL" || true

echo "[INFO] Gerando gráficos diagnósticos comparativos (Abrupto vs Gradual)..."
uv run python plot/plot_astra_diagnostics.py \
    --abrupt-exp "$EXP_ABRUPT" \
    --gradual-exp "$EXP_GRADUAL" || true

echo "================================================================="
echo " BATERIA GRADUAL CONCLUÍDA COM SUCESSO!"
echo " Data/Hora: $(date)"
echo "================================================================="
