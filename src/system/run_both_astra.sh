#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================="
echo " [1/2] INICIANDO BATERIA GRADUAL (astra_allgradual_drift)"
echo "================================================================="
./run.sh \
    --model clip \
    --dataset Cifar10_drift \
    --clients 5 \
    --max-clients 20 \
    --batch-size 8 \
    --grad-accum 4 \
    --rounds 50 \
    --simulations 5 \
    --seed 42 \
    --drift \
    --drift-type gradual \
    --drift-freq 5 \
    --drift-clients all \
    --exp-name astra_allgradual_drift \
    --auto-next

echo "[INFO] Gerando gráficos de benchmark para astra_allgradual_drift..."
uv run python plot/plot_astra_benchmark.py --exp-name astra_allgradual_drift || true

echo "================================================================="
echo " [2/2] INICIANDO BATERIA ABRUPTA (astra_allabrupt_drift)"
echo "================================================================="
./run.sh \
    --model clip \
    --dataset Cifar10_drift \
    --clients 5 \
    --max-clients 20 \
    --batch-size 8 \
    --grad-accum 4 \
    --rounds 50 \
    --simulations 5 \
    --seed 42 \
    --drift \
    --drift-type abrupt \
    --drift-freq 5 \
    --drift-clients all \
    --exp-name astra_allabrupt_drift \
    --auto-next

echo "[INFO] Gerando gráficos de benchmark para astra_allabrupt_drift..."
uv run python plot/plot_astra_benchmark.py --exp-name astra_allabrupt_drift || true

echo "================================================================="
echo " TODAS AS SIMULAÇÕES FORAM CONCLUÍDAS COM SUCESSO!"
echo "================================================================="
