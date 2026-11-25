#!/bin/bash
# LEGADO (Benchmark SBRC 2026 com LoRA-CLIP e partição PUMA — NÃO executa ASTRA-FL).
# Renomeado de run_abrupt_astra.sh (astra_allabrupt_drift -> clip_allabrupt_drift) para evitar confusão com ASTRA-FL.
# Para ASTRA-FL real use `run_astra_paper.sh` ou `run_astra_drift.sh`.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================="
echo " INICIANDO BATERIA ABRUPTA CLIP LEGADO (clip_allabrupt_drift)"
echo " Data/Hora: $(date)"
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
    --exp-name clip_allabrupt_drift \
    --auto-next

echo "================================================================="
echo " BATERIA ABRUPTA CONCLUÍDA COM SUCESSO!"
echo " Data/Hora: $(date)"
echo "================================================================="

echo "[INFO] Gerando gráficos de benchmark para clip_allabrupt_drift..."
uv run python plot/plot_astra_benchmark.py --exp-name clip_allabrupt_drift || true

