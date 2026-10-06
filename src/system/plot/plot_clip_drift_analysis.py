#!/usr/bin/env python3
"""
plot_clip_drift_analysis.py

Gera visualizações acadêmicas de alta resolução (PNG + PDF) para o benchmark do 
modelo `lora_clip` (CLIP ViT-B/32 + LoRA) sob Concept Drift Abrupto e Gradual:

1. 06_clip_abrupto_vs_gradual.png / .pdf: Comparativo da Acurácia Global entre Drift Abrupto e Gradual com CLIP.
2. 07_clip_vs_cnn_drift_comparison.png / .pdf: Comparação de Desempenho e Recuperação Pós-Drift (CLIP vs CNN Baseline).
3. 08_clip_clientes_e_ram.png / .pdf: Acurácia por Cliente (Afetado vs Estável) e Telemetria de RAM RSS.
"""

import os
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuração de Estilo Acadêmico / IEEE
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.titlesize': 20,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'figure.autolayout': True
})

def load_exp_data(exp_dir: Path):
    server_files = sorted(exp_dir.glob("server_*.h5"))
    client_files = sorted(exp_dir.glob("client_*.h5"))

    if not server_files:
        return None

    data = {}
    with h5py.File(server_files[0], 'r') as hf:
        acc = np.array(hf['rs_test_acc'])
        if np.nanmax(acc) <= 1.01:
            acc *= 100.0
        data['server_acc'] = acc
        data['server_loss'] = np.array(hf['rs_test_loss'])

    data['clients'] = {}
    for cf in client_files:
        c_name = cf.stem.split('_')[0] + '_' + cf.stem.split('_')[1]
        with h5py.File(cf, 'r') as hf:
            c_dict = {}
            c_dict['global_acc'] = np.array(hf['rs_global_acc'])
            c_dict['local_acc'] = np.array(hf['rs_local_acc'])
            c_dict['train_acc'] = np.array(hf['rs_train_acc'])
            if 'rs_ram_rss_mb' in hf:
                c_dict['ram_rss'] = np.array(hf['rs_ram_rss_mb'])
            data['clients'][c_name] = c_dict

    return data

def plot_clip_abrupt_vs_gradual(abrupt_data, gradual_data, out_dir: Path):
    fig, ax = plt.subplots(figsize=(11, 6))

    acc_abrupt = abrupt_data['server_acc']
    acc_gradual = gradual_data['server_acc']

    rounds_abrupt = np.arange(1, len(acc_abrupt) + 1)
    rounds_gradual = np.arange(1, len(acc_gradual) + 1)

    # Destaque das fases de Drift
    ax.axvline(x=5, color='#d62728', linestyle='--', lw=2, label='Início do Concept Drift (Round 5)')
    ax.axvspan(5, 8, color='#ff7f0e', alpha=0.15, label='Janela de Drift Gradual (Rounds 5-8)')

    # Curvas de Acurácia Global
    ax.plot(rounds_abrupt, acc_abrupt, color='#d62728', lw=2.8, marker='^', ms=8, label='CLIP LoRA - Drift Abrupto')
    ax.plot(rounds_gradual, acc_gradual, color='#1f77b4', lw=2.8, marker='o', ms=8, label='CLIP LoRA - Drift Gradual')

    # Anotação de recuperação
    ax.annotate(f'Drop Abrupto\n({acc_abrupt[4]:.1f}%)', xy=(5, acc_abrupt[4]), xytext=(3.5, 55),
                arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.5, headwidth=8),
                fontweight='bold', color='#d62728', fontsize=12)

    ax.annotate(f'Recuperação Rápida\n({acc_abrupt[7]:.1f}%)', xy=(8, acc_abrupt[7]), xytext=(8.2, 65),
                arrowprops=dict(facecolor='#2ca02c', shrink=0.08, width=1.5, headwidth=8),
                fontweight='bold', color='#2ca02c', fontsize=12)

    ax.set_xlabel('Rodadas de Comunicação Federada (Rounds)', fontweight='bold')
    ax.set_ylabel('Acurácia Global do Servidor (%)', fontweight='bold')
    ax.set_ylim(40, 100)
    ax.set_xticks(np.arange(1, 13))
    ax.grid(True)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95)
    ax.set_title('Federated Concept Drift: CLIP ViT-B/32 + LoRA (CIFAR-10)', fontweight='bold', pad=12)

    fig.savefig(out_dir / "06_clip_abrupto_vs_gradual.png", dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / "06_clip_abrupto_vs_gradual.pdf", bbox_inches='tight')
    plt.close(fig)
    print("✅ Gráfico 06_clip_abrupto_vs_gradual gerado com sucesso!")

def plot_clip_vs_cnn_comparison(clip_abrupt, clip_gradual, cnn_abrupt, cnn_gradual, out_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Subplot 1: Abrupt Drift (CLIP vs CNN)
    acc_clip_a = clip_abrupt['server_acc']
    rounds_c_a = np.arange(1, len(acc_clip_a) + 1)
    
    ax1.axvline(x=5, color='#d62728', linestyle='--', lw=2, label='Drift Round 5')
    ax1.plot(rounds_c_a, acc_clip_a, color='#1f77b4', lw=2.8, marker='o', ms=7, label='CLIP ViT-B/32 + LoRA')
    
    if cnn_abrupt is not None:
        acc_cnn_a = cnn_abrupt['server_acc']
        rounds_cnn = np.arange(1, len(acc_cnn_a) + 1)
        ax1.plot(rounds_cnn[:10], acc_cnn_a[:10], color='#ff7f0e', lw=2.5, linestyle='--', marker='s', ms=7, label='CNN SimpleModel Baseline')

    ax1.set_title('Acurácia em Drift Abrupto', fontweight='bold')
    ax1.set_xlabel('Rounds', fontweight='bold')
    ax1.set_ylabel('Acurácia Global (%)', fontweight='bold')
    ax1.set_xticks(np.arange(1, 11))
    ax1.grid(True)
    ax1.legend(loc='center left', frameon=True, facecolor='white')

    # Subplot 2: Gradual Drift (CLIP vs CNN)
    acc_clip_g = clip_gradual['server_acc']
    rounds_c_g = np.arange(1, len(acc_clip_g) + 1)

    ax2.axvspan(5, 8, color='#ff7f0e', alpha=0.15, label='Janela Drift (R5-R8)')
    ax2.axvline(x=5, color='#d62728', linestyle='--', lw=2, label='Drift Round 5')
    ax2.plot(rounds_c_g, acc_clip_g, color='#2ca02c', lw=2.8, marker='^', ms=7, label='CLIP ViT-B/32 + LoRA')

    if cnn_gradual is not None:
        acc_cnn_g = cnn_gradual['server_acc']
        rounds_cnn_g = np.arange(1, len(acc_cnn_g) + 1)
        ax2.plot(rounds_cnn_g[:12], acc_cnn_g[:12], color='#d62728', lw=2.5, linestyle='--', marker='d', ms=7, label='CNN SimpleModel Baseline')

    ax2.set_title('Acurácia em Drift Gradual', fontweight='bold')
    ax2.set_xlabel('Rounds', fontweight='bold')
    ax2.set_xticks(np.arange(1, 13))
    ax2.grid(True)
    ax2.legend(loc='center left', frameon=True, facecolor='white')

    fig.savefig(out_dir / "07_clip_vs_cnn_drift_comparison.png", dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / "07_clip_vs_cnn_drift_comparison.pdf", bbox_inches='tight')
    plt.close(fig)
    print("✅ Gráfico 07_clip_vs_cnn_drift_comparison gerado com sucesso!")

def plot_clip_clients_and_ram(gradual_data, out_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    c0 = gradual_data['clients'].get('client_0', None)
    c1 = gradual_data['clients'].get('client_1', None)

    if c0 is None or c1 is None:
        print("⚠️ Dados de clientes não encontrados em gradual_data.")
        return

    rounds = np.arange(1, len(c0['global_acc']) + 1)

    # Subplot 1: Acurácia por Cliente (Afetado vs Estável)
    ax1.axvspan(5, 8, color='#ff7f0e', alpha=0.15, label='Janela Drift Gradual')
    ax1.plot(rounds, c0['global_acc'], color='#d62728', lw=2.8, marker='o', label='Cliente 0 (Afetado por Drift)')
    ax1.plot(rounds, c1['global_acc'], color='#2ca02c', lw=2.8, marker='s', label='Cliente 1 (Estável - Sem Drift)')
    ax1.set_title('Acurácia por Cliente (CLIP Gradual Drift)', fontweight='bold')
    ax1.set_xlabel('Rounds', fontweight='bold')
    ax1.set_ylabel('Acurácia Global (%)', fontweight='bold')
    ax1.set_xticks(np.arange(1, 13))
    ax1.grid(True)
    ax1.legend(loc='lower right', frameon=True, facecolor='white')

    # Subplot 2: Consumo de RAM RSS por Cliente (MB)
    ram0 = c0.get('ram_rss', None)
    ram1 = c1.get('ram_rss', None)

    if ram0 is not None and ram1 is not None:
        width = 0.38
        ax2.bar(rounds - width/2, ram0, width=width, color='#d62728', alpha=0.85, label='Cliente 0 RAM RSS (MB)')
        ax2.bar(rounds + width/2, ram1, width=width, color='#2ca02c', alpha=0.85, label='Cliente 1 RAM RSS (MB)')
        ax2.axvspan(4.5, 8.5, color='#ff7f0e', alpha=0.15, label='Overhead de Memory Blend (Phase Blend)')
        ax2.set_title('Telemetria de Consumo de RAM (RSS MB)', fontweight='bold')
        ax2.set_xlabel('Rounds', fontweight='bold')
        ax2.set_ylabel('Consumo de RAM RSS (MB)', fontweight='bold')
        ax2.set_xticks(np.arange(1, 13))
        ax2.set_ylim(1800, 3000)
        ax2.grid(True, axis='y')
        ax2.legend(loc='upper right', frameon=True, facecolor='white')

    fig.savefig(out_dir / "08_clip_clientes_e_ram.png", dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / "08_clip_clientes_e_ram.pdf", bbox_inches='tight')
    plt.close(fig)
    print("✅ Gráfico 08_clip_clientes_e_ram gerado com sucesso!")

def main():
    base_results = Path('/home/jose/FL_ServerPuma/src/results')

    abrupt_clip_dir = base_results / 'fl_puma_Cifar10_drift_clip_lora_prune1_ala1_paca12_drift_abrupt_r5'
    gradual_clip_dir = base_results / 'fl_puma_Cifar10_drift_clip_lora_prune1_ala1_paca12_drift_gradual_r5'

    abrupt_cnn_dir = base_results / 'fl_puma_Cifar10_drift_cnn_lora_prune1_ala1_paca12_drift_abrupt_r10'
    gradual_cnn_dir = base_results / 'fl_puma_Cifar10_drift_cnn_lora_prune1_ala1_paca12_drift_gradual_r10'

    clip_abrupt = load_exp_data(abrupt_clip_dir)
    clip_gradual = load_exp_data(gradual_clip_dir)

    cnn_abrupt = load_exp_data(abrupt_cnn_dir) if abrupt_cnn_dir.exists() else None
    cnn_gradual = load_exp_data(gradual_cnn_dir) if gradual_cnn_dir.exists() else None

    if clip_abrupt and clip_gradual:
        plot_clip_abrupt_vs_gradual(clip_abrupt, clip_gradual, base_results)
        plot_clip_vs_cnn_comparison(clip_abrupt, clip_gradual, cnn_abrupt, cnn_gradual, base_results)
        plot_clip_clients_and_ram(clip_gradual, base_results)
    else:
        print("❌ Erro ao carregar dados dos experimentos de CLIP.")

if __name__ == '__main__':
    main()
