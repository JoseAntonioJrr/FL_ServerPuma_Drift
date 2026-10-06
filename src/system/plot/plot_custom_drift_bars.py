#!/usr/bin/env python3
"""
plot_custom_drift_bars.py

Gera os gráficos no formato solicitado pelo usuário:
1. Gráfico de Acurácia Global no estilo 01_acuracia_global.pdf + indicação de início/janela de drift (como em concept_drift_analysis.pdf).
2. Gráfico de Consumo de RAM (RSS MB) em GRÁFICO DE BARRAS por rodada, com destaque visual na janela de drift.
"""

import os
import sys
import glob
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configurações de estilo inspiradas em 01_acuracia_global.pdf (fontes grandes e legíveis para artigo)
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'figure.titlesize': 22,
    'grid.alpha': 0.4,
    'grid.linestyle': '--'
})

def load_metrics(exp_dir: Path):
    server_files = sorted(exp_dir.glob("server_*.h5"))
    client_files = sorted(exp_dir.glob("client_*.h5"))

    if not server_files or not client_files:
        return None

    # Server Global Acc
    with h5py.File(server_files[0], 'r') as hf:
        global_acc = np.array(hf['rs_test_acc'])
        if np.nanmax(global_acc) <= 1.01:
            global_acc *= 100.0

    # Client RAM RSS
    clients_ram = []
    for cf in client_files:
        with h5py.File(cf, 'r') as hf:
            if 'rs_ram_rss_mb' in hf:
                clients_ram.append(np.array(hf['rs_ram_rss_mb']))

    avg_ram = np.mean(clients_ram, axis=0) if clients_ram else None
    return global_acc, avg_ram


def plot_accuracy_style(global_acc_10, global_acc_100, drift_start=20, drift_length=10, out_dir=Path('.')):
    """Plot de Acurácia no estilo 01_acuracia_global.pdf com indicação de Drift."""
    rounds = np.arange(1, len(global_acc_10) + 1)
    drift_end = drift_start + drift_length - 1

    fig, ax = plt.subplots(figsize=(12, 7))

    # Janela de Drift Shaded e Linha Vertical
    ax.axvspan(drift_start, drift_end, color='#ff7f0e', alpha=0.18, label='Janela Drift Gradual (R20-R29)')
    ax.axvline(x=drift_start, color='#d62728', linestyle='--', lw=2.5, label='Início do Drift (Round 20)')

    # Linhas de acurácia com marcadores como em 01_acuracia_global
    ax.plot(rounds, global_acc_10, color='#1f77b4', marker='o', markevery=3, ms=7, lw=2.5, label='CIFAR-10 (10 Clientes, CNN)')
    ax.plot(rounds, global_acc_100, color='#2ca02c', marker='s', markevery=3, ms=7, lw=2.5, label='CIFAR-100 (10 Clientes, CNN)')

    ax.set_xlabel('Rodadas de Comunicação (Rounds)', fontweight='bold', labelpad=10)
    ax.set_ylabel('Acurácia Global (%)', fontweight='bold', labelpad=10)
    ax.set_ylim(0, 102)
    ax.grid(True)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()

    out_png = out_dir / "01_acuracia_global_drift.png"
    out_pdf = out_dir / "01_acuracia_global_drift.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Acurácia estilo 01_acuracia_global salva em:\n  🖼️ {out_png}\n  📄 {out_pdf}")


def plot_ram_bars(avg_ram_10, avg_ram_100, drift_start=20, drift_length=10, out_dir=Path('.')):
    """Plot de Consumo de RAM em GRÁFICO DE BARRAS por rodada com destaque no drift."""
    rounds = np.arange(1, len(avg_ram_10) + 1)
    drift_end = drift_start + drift_length - 1

    fig, ax = plt.subplots(figsize=(14, 7))

    width = 0.4
    x1 = rounds - width/2
    x2 = rounds + width/2

    # Definir cores para barras pré/pós drift vs barras durante o drift
    colors_c10 = ['#ff7f0e' if (drift_start <= r <= drift_end) else '#1f77b4' for r in rounds]
    colors_c100 = ['#d62728' if (drift_start <= r <= drift_end) else '#2ca02c' for r in rounds]

    bars1 = ax.bar(x1, avg_ram_10, width=width, color=colors_c10, edgecolor='none', alpha=0.85, label='CIFAR-10 RAM RSS (MB)')
    bars2 = ax.bar(x2, avg_ram_100, width=width, color=colors_c100, edgecolor='none', alpha=0.85, label='CIFAR-100 RAM RSS (MB)')

    # Linha vertical indicando o drift
    ax.axvline(x=drift_start - 0.5, color='#d62728', linestyle='--', lw=2.2, label=f'Início do Drift (Round {drift_start})')
    ax.axvspan(drift_start - 0.5, drift_end + 0.5, color='#ff7f0e', alpha=0.12, label='Janela Transição Drift Gradual (R20-R29)')

    ax.set_xlabel('Rodadas de Comunicação (Rounds)', fontweight='bold', labelpad=10)
    ax.set_ylabel('Consumo de RAM RSS por Cliente (MB)', fontweight='bold', labelpad=10)

    # Ajustar eixo Y para focar no delta de consumo
    min_ram = min(np.min(avg_ram_10), np.min(avg_ram_100))
    max_ram = max(np.max(avg_ram_10), np.max(avg_ram_100))
    ax.set_ylim(min_ram - 200, max_ram + 300)

    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    # Legenda customizada
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='CIFAR-10 (Normal)'),
        Patch(facecolor='#2ca02c', label='CIFAR-100 (Normal)'),
        Patch(facecolor='#ff7f0e', label='CIFAR-10 (Durante Drift)'),
        Patch(facecolor='#d62728', label='CIFAR-100 (Durante Drift)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()

    out_png = out_dir / "02_consumo_ram_barras.png"
    out_pdf = out_dir / "02_consumo_ram_barras.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 RAM em barras salva em:\n  🖼️ {out_png}\n  📄 {out_pdf}")


def main():
    base_results = Path('/home/jose/FL_ServerPuma/src/results')
    exp_c10 = base_results / 'fl_puma_Cifar10_drift_cnn_lora_prune1_ala1_paca12_drift_gradual_r20'
    exp_c100 = base_results / 'fl_puma_Cifar100_drift_cnn_lora_prune1_ala1_paca12_drift_gradual_r20'

    acc_10, ram_10 = load_metrics(exp_c10)
    acc_100, ram_100 = load_metrics(exp_c100)

    output_dir = base_results
    plot_accuracy_style(acc_10, acc_100, out_dir=output_dir)
    plot_ram_bars(ram_10, ram_100, out_dir=output_dir)

if __name__ == '__main__':
    main()
