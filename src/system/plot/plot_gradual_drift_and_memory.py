#!/usr/bin/env python3
"""
plot_gradual_drift_and_memory.py

Gera gráficos de qualidade de publicação (PNG 300 DPI e PDF) para a análise de
Concept Drift Gradual (Round 20-29) e Telemetria de Memória RAM (psutil RSS MB).
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

# Estilo visual limpo e profissional
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'grid.alpha': 0.4,
    'grid.linestyle': '--'
})

def load_exp_metrics(exp_dir: Path):
    server_files = sorted(exp_dir.glob("server_*.h5"))
    client_files = sorted(exp_dir.glob("client_*.h5"))

    if not server_files or not client_files:
        print(f"❌ Nenhum arquivo HDF5 encontrado em {exp_dir}")
        return None

    # Carrega dados do Servidor
    with h5py.File(server_files[0], 'r') as hf:
        global_acc = np.array(hf['rs_test_acc'])
        if np.nanmax(global_acc) <= 1.01:
            global_acc *= 100.0

    # Carrega dados dos Clientes (Acurácia local e RAM)
    clients_ram = []
    clients_acc = []

    for cf in client_files:
        with h5py.File(cf, 'r') as hf:
            if 'rs_ram_rss_mb' in hf:
                clients_ram.append(np.array(hf['rs_ram_rss_mb']))
            if 'rs_global_acc' in hf:
                c_acc = np.array(hf['rs_global_acc'])
                if np.nanmax(c_acc) <= 1.01:
                    c_acc *= 100.0
                clients_acc.append(c_acc)

    avg_ram = np.mean(clients_ram, axis=0) if clients_ram else None
    std_ram = np.std(clients_ram, axis=0) if clients_ram else None

    return {
        'exp_dir': exp_dir,
        'global_acc': global_acc,
        'clients_ram': clients_ram,
        'avg_ram': avg_ram,
        'std_ram': std_ram,
        'clients_acc': clients_acc
    }

def generate_individual_plot(metrics, dataset_name, drift_start=20, drift_length=10, out_dir=Path('.')):
    global_acc = metrics['global_acc']
    avg_ram = metrics['avg_ram']
    rounds = np.arange(1, len(global_acc) + 1)
    drift_end = drift_start + drift_length - 1

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Sombreamento da Janela de Drift Gradual
    ax1.axvspan(drift_start, drift_end, color='#ff7f0e', alpha=0.15, label=f'Janela Drift Gradual (R{drift_start}-R{drift_end})')
    ax1.axvline(x=drift_start, color='#d62728', linestyle='--', lw=2, label=f'Início Drift (Round {drift_start})')

    # Eixo 1: Acurácia Global
    color_acc = '#1f77b4'
    ax1.set_xlabel('Rodada de Comunicação (Round)', fontweight='bold')
    ax1.set_ylabel('Acurácia Global (%)', color=color_acc, fontweight='bold')
    line1 = ax1.plot(rounds, global_acc, color=color_acc, marker='o', ms=4, lw=2.2, label='Acurácia Global (%)')
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_ylim(0, max(100, float(np.max(global_acc)) + 5))
    ax1.grid(True)

    # Eixo 2 (Twin): Consumo de RAM (MB)
    ax2 = ax1.twinx()
    color_ram = '#2ca02c'
    ax2.set_ylabel('Consumo de RAM por Cliente (RSS MB)', color=color_ram, fontweight='bold')
    if avg_ram is not None:
        line2 = ax2.plot(rounds, avg_ram, color=color_ram, marker='s', ms=4, lw=2, linestyle='-.', label='RAM Média (MB)')
        ax2.tick_params(axis='y', labelcolor=color_ram)
        ram_min, ram_max = np.min(avg_ram), np.max(avg_ram)
        ax2.set_ylim(ram_min - 100, ram_max + 250)

    # Título e Legendas Combinadas
    plt.title(f'Desempenho FL & Consumo de Memória RAM sob Concept Drift Gradual\n[{dataset_name} — 10 Clientes | CNN]', fontweight='bold', pad=12)
    
    # Unir legendas dos dois eixos
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels() if avg_ram is not None else ([], [])
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()

    fn_stem = f"gradual_drift_ram_{dataset_name.lower().replace('-', '')}"
    out_png = out_dir / f"{fn_stem}.png"
    out_pdf = out_dir / f"{fn_stem}.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Gráfico salvo ({dataset_name}): {out_png}")


def generate_comparison_plot(c10_metrics, c100_metrics, drift_start=20, drift_length=10, out_dir=Path('.')):
    rounds_10 = np.arange(1, len(c10_metrics['global_acc']) + 1)
    rounds_100 = np.arange(1, len(c100_metrics['global_acc']) + 1)
    drift_end = drift_start + drift_length - 1

    fig, (ax_acc, ax_ram) = plt.subplots(1, 2, figsize=(16, 6))

    # --- SUBPLOT 1: Comparativo de Acurácia Global ---
    ax_acc.axvspan(drift_start, drift_end, color='#ff7f0e', alpha=0.15, label='Janela Drift Gradual (R20-R29)')
    ax_acc.axvline(x=drift_start, color='#d62728', linestyle='--', lw=1.8, label='Início do Drift (R20)')

    ax_acc.plot(rounds_10, c10_metrics['global_acc'], color='#1f77b4', marker='o', ms=4, lw=2.2, label='CIFAR-10 Global Acc (%)')
    ax_acc.plot(rounds_100, c100_metrics['global_acc'], color='#9467bd', marker='^', ms=4, lw=2.2, label='CIFAR-100 Global Acc (%)')

    ax_acc.set_xlabel('Rodada de Comunicação (Round)', fontweight='bold')
    ax_acc.set_ylabel('Acurácia Global (%)', fontweight='bold')
    ax_acc.set_title('(A) Resiliência e Recuperação da Acurácia Global', fontweight='bold')
    ax_acc.grid(True)
    ax_acc.set_ylim(0, 100)
    ax_acc.legend(loc='lower left', frameon=True)

    # --- SUBPLOT 2: Comparativo de Consumo de RAM ---
    ax_ram.axvspan(drift_start, drift_end, color='#ff7f0e', alpha=0.15, label='Janela Drift Gradual (R20-R29)')
    ax_ram.axvline(x=drift_start, color='#d62728', linestyle='--', lw=1.8, label='Início do Drift (R20)')

    if c10_metrics['avg_ram'] is not None:
        ax_ram.plot(rounds_10, c10_metrics['avg_ram'], color='#2ca02c', marker='s', ms=4, lw=2, label='CIFAR-10 RAM (RSS MB)')
    if c100_metrics['avg_ram'] is not None:
        ax_ram.plot(rounds_100, c100_metrics['avg_ram'], color='#e377c2', marker='D', ms=4, lw=2, label='CIFAR-100 RAM (RSS MB)')

    ax_ram.set_xlabel('Rodada de Comunicação (Round)', fontweight='bold')
    ax_ram.set_ylabel('Consumo Médio de RAM por Cliente (MB)', fontweight='bold')
    ax_ram.set_title('(B) Telemetria de Memória RAM (psutil RSS)', fontweight='bold')
    ax_ram.grid(True)
    ax_ram.legend(loc='upper right', frameon=True)

    plt.suptitle('Análise Comparativa: Federated Concept Drift Gradual — CIFAR-10 vs CIFAR-100 (10 Clientes, CNN)', fontweight='bold', fontsize=15)
    plt.tight_layout()

    out_png = out_dir / "gradual_drift_comparison_cifar10_vs_cifar100.png"
    out_pdf = out_dir / "gradual_drift_comparison_cifar10_vs_cifar100.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Gráfico comparativo salvo: {out_png}")


def main():
    base_results = Path('/home/jose/FL_ServerPuma/src/results')
    
    exp_c10 = base_results / 'fl_puma_Cifar10_drift_cnn_lora_prune1_ala1_paca12_drift_gradual_r20'
    exp_c100 = base_results / 'fl_puma_Cifar100_drift_cnn_lora_prune1_ala1_paca12_drift_gradual_r20'

    print("📊 Carregando dados dos experimentos...")
    metrics_c10 = load_exp_metrics(exp_c10)
    metrics_c100 = load_exp_metrics(exp_c100)

    if not metrics_c10 or not metrics_c100:
        print("❌ Falha ao carregar métricas.")
        sys.exit(1)

    output_dir = base_results
    generate_individual_plot(metrics_c10, "CIFAR-10", out_dir=output_dir)
    generate_individual_plot(metrics_c100, "CIFAR-100", out_dir=output_dir)
    generate_comparison_plot(metrics_c10, metrics_c100, out_dir=output_dir)
    print("✨ Todos os gráficos foram gerados com sucesso!")

if __name__ == '__main__':
    main()
