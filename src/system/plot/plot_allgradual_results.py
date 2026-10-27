#!/usr/bin/env python3
"""
plot_allgradual_results.py — Análise e visualização dos resultados de astra_allgradual_drift.
Gera gráficos científicos em alta resolução (300 DPI) em:
  /home/jose/FL_ServerPuma/src/results/astra_allgradual_drift/
"""
import os
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configuração estética padronizada
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'savefig.dpi': 300,
    'grid.alpha': 0.35,
    'grid.linestyle': '--'
})

RESULTS_DIR = Path("/home/jose/FL_ServerPuma/src/results")
GRADUAL_DIR = RESULTS_DIR / "astra_allgradual_drift"
OLD_GRADUAL_DIR = RESULTS_DIR / "astra_periodic_gradual_drift"
OLD_ABRUPT_DIR = RESULTS_DIR / "astra_periodic_drift"

L = 5  # drift_freq


def alpha_schedule(n_rounds, L=L):
    """Schedule gradual alpha(t) implementado no client/server."""
    a = []
    for t in range(1, n_rounds + 1):
        p, s = (t - 1) // L, (t - 1) % L
        r = s / (L - 1) if L > 1 else 0.0
        a.append(r if p % 2 == 0 else 1.0 - r)
    return np.array(a)


def main():
    GRADUAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Carregar run1 do astra_allgradual_drift
    server_files = sorted(GRADUAL_DIR.glob("server_*.h5"))
    if not server_files:
        print("Nenhum arquivo server_*.h5 encontrado em astra_allgradual_drift!")
        return
        
    target_h5 = server_files[0]
    print(f"Lendo dados de: {target_h5.name}")
    
    with h5py.File(target_h5, "r") as hf:
        acc = np.array(hf["rs_test_acc"])
        loss = np.array(hf["rs_test_loss"])
        round_time = np.array(hf["Round_time"]) if "Round_time" in hf else None
        sent_mb = np.array(hf["sended_model_Mb"]) if "sended_model_Mb" in hf else None
        
    n_rounds = len(acc)
    rounds = np.arange(1, n_rounds + 1)
    alpha = alpha_schedule(n_rounds, L=L)
    bayes_ceiling = 100.0 * np.maximum(alpha, 1.0 - alpha)
    
    # -------------------------------------------------------------
    # Gráfico 1: Acurácia Global e Loss ao longo dos 50 rounds
    # -------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color_acc = '#1f77b4'
    color_loss = '#d62728'
    
    # Linha da acurácia
    line1 = ax1.plot(rounds, acc, color=color_acc, lw=2.4, marker='o', ms=4, label='Global Test Accuracy (%)')
    ax1.set_xlabel('Communication Round (Rounds)', fontweight='bold')
    ax1.set_ylabel('Global Test Accuracy (%)', color=color_acc, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_ylim(0, 100)
    ax1.set_xlim(1, n_rounds)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax1.grid(True)
    
    # Destacar ciclos de drift (a cada L=5 rounds)
    for r in range(L, n_rounds + 1, L):
        ax1.axvline(r, color='#7f7f7f', ls=':', lw=1.2, alpha=0.7)
    
    # Eixo secundário da Loss
    ax2 = ax1.twinx()
    line2 = ax2.plot(rounds, loss, color=color_loss, lw=2.0, ls='--', marker='s', ms=3.5, label='Global Test Loss')
    ax2.set_ylabel('Global Test Loss', color=color_loss, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_loss)
    ax2.set_ylim(0, max(loss) * 1.15 if len(loss) else 10)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', framealpha=0.9)
    
    plt.title('ASTRA Benchmark — astra_allgradual_drift (Run 1, LoRA-CLIP)\n'
              'Global Test Accuracy & Cross-Entropy Loss (Dirichlet non-IID α=0.1, L=5)', fontweight='bold')
    fig.tight_layout()
    p1 = GRADUAL_DIR / "01_allgradual_run1_accuracy_loss.png"
    fig.savefig(p1, bbox_inches='tight')
    plt.close(fig)
    print(f"Salvo: {p1.name}")

    # -------------------------------------------------------------
    # Gráfico 2: Acurácia vs. Schedule α(t) e Teto Teórico de Bayes
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(rounds, acc, color='#1f77b4', lw=2.6, marker='o', ms=4, label='Global Acc (astra_allgradual_drift Run 1)')
    ax.plot(rounds, bayes_ceiling, color='#2ca02c', lw=2.0, ls='--', label=r'Bayes Optimal Ceiling $\max(\alpha, 1-\alpha)$')
    
    # Sombreamento do teto
    ax.fill_between(rounds, acc, bayes_ceiling, where=(bayes_ceiling >= acc), color='#2ca02c', alpha=0.1, label='Bayes Theoretical Margin')
    ax.set_xlabel('Communication Round', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_xlim(1, n_rounds)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.grid(True)
    
    ax2 = ax.twinx()
    ax2.step(rounds, alpha, where='mid', color='#ff7f0e', lw=1.8, alpha=0.85, label=r'$\alpha(t)$ (Phase 2 fraction)')
    ax2.set_ylabel(r'$\alpha(t)$ Schedule', color='#ff7f0e', fontweight='bold')
    ax2.set_ylim(-0.05, 1.6)
    ax2.tick_params(axis='y', colors='#ff7f0e')
    
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='lower left', ncol=2, framealpha=0.9)
    ax.set_title('ASTRA Gradual Drift — Empirical Accuracy vs. Bayes Ceiling\n'
                 'LoRA-CLIP under Continuous Label Ambiguity (CIFAR-10 Drift, L=5)', fontweight='bold')
    fig.tight_layout()
    p2 = GRADUAL_DIR / "02_allgradual_vs_bayes_ceiling.png"
    fig.savefig(p2, bbox_inches='tight')
    plt.close(fig)
    print(f"Salvo: {p2.name}")

    # -------------------------------------------------------------
    # Gráfico 3: Comparativo Dinâmico (allgradual) vs. Estático Anterior
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Carregar dados antigos de astra_periodic_gradual_drift se existirem
    old_grad_runs = []
    for f in sorted(OLD_GRADUAL_DIR.glob("server_*.h5")):
        try:
            with h5py.File(f, "r") as hf:
                old_grad_runs.append(np.array(hf["rs_test_acc"]))
        except Exception:
            pass
            
    if old_grad_runs:
        min_len = min(len(r) for r in old_grad_runs)
        old_arr = np.array([r[:min_len] for r in old_grad_runs])
        old_mean = old_arr.mean(axis=0)
        old_std = old_arr.std(axis=0)
        ax.fill_between(rounds[:min_len], old_mean - old_std, old_mean + old_std, color='#7f7f7f', alpha=0.2, label='Static Selection Prior Baseline (±1σ)')
        ax.plot(rounds[:min_len], old_mean, color='#555555', lw=2.2, ls=':', label=f'Static Sampling (Prior Mean, N={len(old_grad_runs)})')
    
    ax.plot(rounds, acc, color='#1f77b4', lw=2.6, marker='o', ms=4, label='Dynamic Sampling (astra_allgradual_drift Run 1)')
    
    for r in range(L, n_rounds + 1, L):
        ax.axvline(r, color='#d62728', ls='--', lw=1.0, alpha=0.6)
        
    ax.set_xlabel('Communication Round', fontweight='bold')
    ax.set_ylabel('Global Test Accuracy (%)', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_xlim(1, n_rounds)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.grid(True)
    ax.legend(loc='lower left', framealpha=0.9)
    ax.set_title('ASTRA Benchmark — Comparison of Client Sampling Paradigms\n'
                 'Dynamic Deterministic Selection (20 Clients, 5/Round) vs. Static Baseline', fontweight='bold')
    fig.tight_layout()
    p3 = GRADUAL_DIR / "03_allgradual_vs_static_baseline.png"
    fig.savefig(p3, bbox_inches='tight')
    plt.close(fig)
    print(f"Salvo: {p3.name}")

    # -------------------------------------------------------------
    # Gráfico 4: Tempo de Rodada e Eficiência Computacional
    # -------------------------------------------------------------
    if round_time is not None and len(round_time) == n_rounds:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.bar(rounds, round_time, color='#2ca02c', alpha=0.75, edgecolor='#1b611b', width=0.8)
        ax.axhline(round_time.mean(), color='#d62728', lw=2.0, ls='--', label=f'Mean Round Duration: {round_time.mean():.1f}s')
        ax.set_xlabel('Communication Round', fontweight='bold')
        ax.set_ylabel('Round Duration (seconds)', fontweight='bold')
        ax.set_xlim(0.5, n_rounds + 0.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.grid(True)
        ax.legend(loc='upper right', framealpha=0.9)
        ax.set_title('ASTRA Gradual Drift — Execution Time per Round\n'
                     'Federated LoRA-CLIP Training & Synchronization Overhead', fontweight='bold')
        fig.tight_layout()
        p4 = GRADUAL_DIR / "04_allgradual_round_duration.png"
        fig.savefig(p4, bbox_inches='tight')
        plt.close(fig)
        print(f"Salvo: {p4.name}")

    print("Todos os gráficos foram gerados com sucesso!")


if __name__ == '__main__':
    main()
