#!/bin/bash
# Benchmark ASTRA-FL sobre LoRA-CLIP + PUMA concept drift (zero data-leakage).
# Usa --astra (proximal + KD com teacher congelado) no modelo CLIP, mantendo
# o particionamento ASTRA-FL, Dirichlet alpha configurável e local_epochs configurável.
#
# Uso:
#   ./run_clip_astra_drift.sh [SIMULATIONS] [ROUNDS]
#   DATASET=Cifar100 ALPHA=0.1 LOCAL_EPOCHS=1 ./run_clip_astra_drift.sh 1 50
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

SIMULATIONS="${1:-1}"
ROUNDS="${2:-50}"
DATASET="${DATASET:-Cifar10}"
ALPHA="${ALPHA:-0.1}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
PARTITION_SEED="${PARTITION_SEED:-42}"
DRIFT_TRANSFORM="${DRIFT_TRANSFORM:-label_swap}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
CLIENTS_PER_ROUND="${CLIENTS_PER_ROUND:-5}"
MAX_CLIENTS="${MAX_CLIENTS:-20}"
SEED="${SEED:-42}"
ASTRA_MU="${ASTRA_MU:-0.01}"
ASTRA_ALPHA="${ASTRA_ALPHA:-0.2}"
ASTRA_TEMPERATURE="${ASTRA_TEMPERATURE:-3.0}"
ASTRA_BOOT_ROUNDS="${ASTRA_BOOT_ROUNDS:-10}"
ASTRA_INTERVAL="${ASTRA_INTERVAL:-2}"

DS_TAG="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')"
EXP_ABRUPT="${EXP_ABRUPT:-clip_astrafl_${DS_TAG}_a${ALPHA}_e${LOCAL_EPOCHS}_abrupt_drift}"
EXP_GRADUAL="${EXP_GRADUAL:-clip_astrafl_${DS_TAG}_a${ALPHA}_e${LOCAL_EPOCHS}_gradual_drift}"

echo "================================================================="
echo " BENCHMARK ASTRA-FL + LoRA-CLIP + CONCEPT DRIFT (zero leakage)"
echo " Dataset: $DATASET | alpha=$ALPHA | local_epochs=$LOCAL_EPOCHS"
echo " Simulações: $SIMULATIONS | Rodadas: $ROUNDS | Clientes: $CLIENTS_PER_ROUND/$MAX_CLIENTS"
echo " Data/Hora de Início: $(date)"
echo "================================================================="

# -----------------------------------------------------------------
# [1/2] CENÁRIO ABRUPTO (Periódico a cada 5 rounds)
# -----------------------------------------------------------------
echo ""
echo "================================================================="
echo " [1/2] INICIANDO BATERIA ABRUPTA ($EXP_ABRUPT)"
echo " Data/Hora: $(date)"
echo "================================================================="

./run.sh \
    --partition-source astra \
    --dirichlet-alpha "$ALPHA" \
    --partition-seed "$PARTITION_SEED" \
    --drift-transform "$DRIFT_TRANSFORM" \
    -d "$DATASET" \
    --model clip \
    --astra \
    --astra-mu "$ASTRA_MU" \
    --astra-alpha "$ASTRA_ALPHA" \
    --astra-temperature "$ASTRA_TEMPERATURE" \
    --astra-boot-rounds "$ASTRA_BOOT_ROUNDS" \
    --astra-interval "$ASTRA_INTERVAL" \
    --astra-local-epochs "$LOCAL_EPOCHS" \
    --clients "$CLIENTS_PER_ROUND" \
    --max-clients "$MAX_CLIENTS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --local-epochs "$LOCAL_EPOCHS" \
    --rounds "$ROUNDS" \
    --simulations "$SIMULATIONS" \
    --seed "$SEED" \
    --drift \
    --drift-type abrupt \
    --drift-freq 5 \
    --drift-clients all \
    --global-eval both \
    --deterministic-selection \
    --exp-name "$EXP_ABRUPT" \
    --auto-next

echo "[INFO] Gerando gráficos para $EXP_ABRUPT..."
uv run python plot/plot_astra_benchmark.py --exp-name "$EXP_ABRUPT" || true

# -----------------------------------------------------------------
# [2/2] CENÁRIO GRADUAL (Periódico a cada 5 rounds com blend)
# -----------------------------------------------------------------
echo ""
echo "================================================================="
echo " [2/2] INICIANDO BATERIA GRADUAL ($EXP_GRADUAL)"
echo " Data/Hora: $(date)"
echo "================================================================="

./run.sh \
    --partition-source astra \
    --dirichlet-alpha "$ALPHA" \
    --partition-seed "$PARTITION_SEED" \
    --drift-transform "$DRIFT_TRANSFORM" \
    -d "$DATASET" \
    --model clip \
    --astra \
    --astra-mu "$ASTRA_MU" \
    --astra-alpha "$ASTRA_ALPHA" \
    --astra-temperature "$ASTRA_TEMPERATURE" \
    --astra-boot-rounds "$ASTRA_BOOT_ROUNDS" \
    --astra-interval "$ASTRA_INTERVAL" \
    --astra-local-epochs "$LOCAL_EPOCHS" \
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

echo ""
echo "================================================================="
echo " TODAS AS SIMULAÇÕES FORAM CONCLUÍDAS COM SUCESSO!"
echo " Data/Hora de Término: $(date)"
echo "================================================================="
