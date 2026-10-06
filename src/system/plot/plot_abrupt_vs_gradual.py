#!/usr/bin/env python3
"""
plot_abrupt_vs_gradual.py

Gera os gráficos para o experimento de Concept Drift Abrupto e o comparativo Abrupto vs Gradual:
1. 03_acuracia_abrupto.pdf / .png: Acurácia Global do Servidor para Drift Abrupto (CIFAR-10 e CIFAR-100).
2. 04_consumo_ram_abrupto_barras.pdf / .png: Consumo de RAM em barras para Drift Abrupto.
3. 05_comparativo_abrupto_vs_gradual.pdf / .png: Comparação da Acurácia Global (Gradual vs Abrupto).
"""

import os
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Estilo acadêmico
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 14,
    'figure.titlesize': 22,
    'grid.alpha': 0.4,
    'grid.linestyle': '--'
})

def load_metrics(exp_dir: Path):
    server_files = sorted(exp_dir.glob("server_*.h5"))
    client_files = sorted(exp_dir.glob("client_*.h5"))

    if not server_files or not client_files:
        return None, None

    with h5py.File(server_files[0], 'r') as hf:
        global_acc = np.array(hf['rs_test_acc'])
        if np.nanmax(global_acc) <= 1.01:
            global_acc *= 100.0

    clients_ram = []
    for cf in client_files:
        with h5py.File(cf, 'r') as hf:
            if 'rs_ram_rss_mb' in hf:
                clients_ram.append(np.array(hf['rs_ram_rss_mb']))

    avg_ram = np.mean(clients_ram, axis=0) if clients_ram else None
    return global_acc, avg_ram

def plot_abrupt_accuracy(acc_10, acc_100, drift_round=20, out_dir=Path('.')):
    rounds = np.arange(1, len(acc_10) + 1)
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.axvline(x=drift_round, color='#d62728', linestyle='--', lw=2.5, label=f'Início do Drift Abrupto (Round {drift_round})')

    ax.plot(rounds, acc_10, color='#1f77b4', marker='o', markevery=3, ms=7, lw=2.5, label='CIFAR-10 (10 Clientes, Abrupto)')
    ax.plot(rounds, acc_100, color='#2ca02c', marker='s', markevery=3, ms=7, lw=2.5, label='CIFAR-100 (10 Clientes, Abrupto)')

    ax.set_xlabel('Rodadas de Comunicação (Rounds)', fontweight='bold', labelpad=10)
    ax.set_ylabel('Acurácia Global (%)', fontweight='bold', labelpad=10)
    ax.set_ylim(0, 102)
    ax.grid(True)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_dir / "03_acuracia_abrupto.png", dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / "03_acuracia_abrupto.pdf", bbox_inches='tight')
    plt.close(fig)
    print("📊 Gráfico de Acurácia Abrupta salvo.")

def plot_abrupt_ram_bars(ram_10, ram_100, drift_round=20, out_dir=Path('.')):
    rounds = np.arange(1, len(ram_10) + 1)
    fig, ax = plt.subplots(figsize=(14, 7))

    width = 0.4
    x1 = rounds - width/2
    x2 = rounds + width/2

    colors_c10 = ['#d62728' if r == drift_round else '#1f77b4' for r in rounds]
    colors_c100 = ['#ff7f0e' if r == drift_round else '#2ca02c' for r in rounds]

    ax.bar(x1, ram_10, width=width, color=colors_c10, alpha=0.85, label='CIFAR-10 RAM RSS (MB)')
    ax.bar(x2, ram_100, width=width, color=colors_c100, alpha=0.85, label='CIFAR-100 RAM RSS (MB)')

    ax.axvline(x=drift_round - 0.5, color='#d62728', linestyle='--', lw=2.2, label=f'Troca Abrupta de Dados (Round {drift_round})')

    ax.set_xlabel('Rodadas de Comunicação (Rounds)', fontweight='bold', labelpad=10)
    ax.set_ylabel('Consumo de RAM RSS por Cliente (MB)', fontweight='bold', labelpad=10)

    min_ram = min(np.min(ram_10), np.min(ram_100))
    max_ram = max(np.max(ram_10), np.max(ram_100))
    ax.set_ylim(min_ram - 200, max_ram + 300)

    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    legend_elements = [
        Patch(facecolor='#1f77b4', label='CIFAR-10 (RAM Constante)'),
        Patch(facecolor='#2ca02c', label='CIFAR-100 (RAM Constante)'),
        Patch(facecolor='#d62728', label=f'CIFAR-10 (Round {drift_round} Troca Abrupta)'),
        Patch(facecolor='#ff7f0e', label=f'CIFAR-100 (Round {drift_round} Troca Abrupta)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_dir / "04_consumo_ram_abrupto_barras.png", dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / "04_consumo_ram_abrupto_barras.pdf", bbox_inches='tight')
    plt.close(fig)
    print("📊 Gráfico de RAM Abrupta em Barras salvo.")

def plot_comparison(g_acc_10, a_acc_10, g_acc_100, a_acc_100, drift_round=20, out_dir=Path('.')):
    rounds = np.arange(1, len(g_acc_10) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    # CIFAR-10 Subplot
    ax1.axvspan(drift_round, drift_round+9, color='#ff7f0e', alpha=0.15, label='Janela Drift Gradual (R20-R29)')
    ax1.axvline(x=drift_round, color='#d62728', linestyle='--', lw=2, label='Início do Drift (Round 20)')
    ax1.plot(rounds, g_acc_10, color='#1f77b4', lw=2.5, marker='o', markevery=4, label='Drift Gradual')
    ax1.plot(rounds, a_acc_10, color='#d62728', lw=2.5, linestyle='-.', marker='^', markevery=4, label='Drift Abrupto')
    ax1.set_title('CIFAR-10: Gradual vs Abrupto', fontweight='bold', fontsize=18)
    ax1.set_xlabel('Rodadas de Comunicação (Rounds)', fontweight='bold')
    ax1.set_ylabel('Acurácia Global (%)', fontweight='bold')
    ax1.grid(True)
    ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

    # CIFAR-100 Subplot
    ax2.axvspan(drift_round, drift_round+9, color='#ff7f0e', alpha=0.15, label='Janela Drift Gradual (R20-R29)')
    ax2.axvline(x=drift_round, color='#d62728', linestyle='--', lw=2, label='Início do Drift (Round 20)')
    ax2.plot(rounds, g_acc_100, color='#2ca02c', lw=2.5, marker='s', markevery=4, label='Drift Gradual')
    ax2.plot(rounds, a_acc_100, color='#9467bd', lw=2.5, linestyle='-.', marker='d', markevery=4, label='Drift Abrupto')
    ax2.set_title('CIFAR-100: Gradual vs Abrupto', fontweight='bold', fontsize=18)
    ax2.set_xlabel('Rodadas de Comunicação (Rounds)', fontweight='bold')
    ax2.grid(True)
    ax2.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_dir / "05_comparativo_abrupto_vs_gradual.png", dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / "05_comparativo_abrupto_vs_gradual.pdf", bbox_inches='tight')
    plt.close(fig)
    print("📊 Gráfico Comparativo Gradual vs Abrupto salvo.")

def main():
    base_results = Path('/home/jose/FL_ServerPuma/src/results')
    
    g_c10_dir = base_results / 'fl_puma_Cifar10_drift_cnn_lora_prune1_ala1_paca12_drift_gradual_r20'
    g_c100_dir = base_results / 'fl_puma_Cifar100_drift_cnn_lora_prune1_ala1_paca12_drift_gradual_r20'

    a_c10_dir = base_results / 'fl_puma_Cifar10_drift_cnn_lora_prune1_ala1_paca12_drift_abrupt_r20'
    a_c100_dir = base_results / 'fl_puma_Cifar100_drift_cnn_lora_prune1_ala1_paca12_drift_abrupt_r20'

    g_acc_10, g_ram_10 = load_metrics(g_c10_dir)
    g_acc_100, g_ram_100 = load_metrics(g_c100_dir)

    a_acc_10, a_ram_10 = load_metrics(a_c10_dir)
    a_acc_100, a_ram_100 = load_metrics(a_c100_dir)

    plot_abrupt_accuracy(a_acc_10, a_acc_100, out_dir=base_results)
    plot_abrupt_ram_bars(a_ram_10, a_ram_100, out_dir=base_results)
    plot_comparison(g_acc_10, a_acc_10, g_acc_100, a_acc_100, out_dir=base_results)

if __name__ == '__main__':
    main()
