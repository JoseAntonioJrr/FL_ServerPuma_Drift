#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

DATASET="${DATASET:-Cifar100}"
ALPHA="${ALPHA:-0.1}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
PARTITION_SEED="${PARTITION_SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
CLIENTS_PER_ROUND="${CLIENTS_PER_ROUND:-5}"
MAX_CLIENTS="${MAX_CLIENTS:-20}"
SEED="${SEED:-42}"
ROUNDS="${ROUNDS:-50}"
SIMULATIONS="${SIMULATIONS:-1}"

DS_TAG="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')"
EXP_NODRIFT="${EXP_NODRIFT:-clip_astra_${DS_TAG}_a${ALPHA}_e${LOCAL_EPOCHS}_nodrift}"

echo "================================================================="
echo " BENCHMARK LoRA-CLIP ESTACIONÁRIO (SEM DRIFT) — ASTRA-FL"
echo " Dataset: $DATASET | alpha=$ALPHA | local_epochs=$LOCAL_EPOCHS"
echo " Simulações: $SIMULATIONS | Rodadas: $ROUNDS | Clientes: $CLIENTS_PER_ROUND/$MAX_CLIENTS"
echo " Experimento: $EXP_NODRIFT"
echo " Data/Hora de Início: $(date)"
echo "================================================================="

./run.sh \
    --partition-source astra \
    --dirichlet-alpha "$ALPHA" \
    --partition-seed "$PARTITION_SEED" \
    --drift-transform none \
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
    --global-eval both \
    --deterministic-selection \
    --exp-name "$EXP_NODRIFT" \
    --auto-next

echo "[INFO] Gerando gráficos para $EXP_NODRIFT..."
uv run python plot/plot_astra_benchmark.py --exp-name "$EXP_NODRIFT" || true

echo "================================================================="
echo " EXPERIMENTO ESTACIONÁRIO CONCLUÍDO COM SUCESSO!"
echo " Data/Hora de Término: $(date)"
echo "================================================================="
