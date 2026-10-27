#!/usr/bin/env python3
"""
generate_allgradual_astra_plots.py
Gera os gráficos no padrão exato de astra_periodic_gradual_drift para a pasta astra_allgradual_drift:
  - 01_astra_gradual_run1.png       (Run individual)
  - 06_astra_gradual_mean_general.png (Média geral e banda ±1 std dev)
  - 07_astra_gradual_alpha_ceiling.png (Acurácia vs Schedule alpha(t) e Teto de Bayes)
  - 08_astra_abrupt_vs_gradual.png  (Comparação Abrupto vs Gradual)
  - 09_astra_gradual_acc_vs_alpha.png (Histerese de adaptação)
"""
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configuração de estilo idêntica ao plot_astra_diagnostics.py e plot_astra_benchmark.py
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'savefig.dpi': 300,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

RESULTS = Path("/home/jose/FL_ServerPuma/src/results")
TARGET_DIR = RESULTS / "astra_allgradual_drift"
ABRUPT_DIR = RESULTS / "astra_periodic_drift"
L = 5  # drift_freq


def alpha_schedule(n_rounds, L=L):
    """Mesma fórmula de client.py/server.py (drift periódico gradual)."""
    a = []
    for t in range(1, n_rounds + 1):
        p, s = (t - 1) // L, (t - 1) % L
        r = s / (L - 1) if L > 1 else 0.0
        a.append(r if p % 2 == 0 else 1.0 - r)
    return np.array(a)


def abrupt_phase(n_rounds, L=L):
    """1 = Phase 2 ativa, 0 = Phase 1 ativa."""
    ph, cur = [], 0
    for t in range(1, n_rounds + 1):
        if t % L == 0:
            cur = 1 if (t // L) % 2 == 1 else 0
        ph.append(cur)
    return np.array(ph)


def load_runs(exp_dir: Path):
    runs = []
    for sf in sorted(exp_dir.glob("server_*.h5")):
        try:
            with h5py.File(sf, "r") as hf:
                if "rs_test_acc" in hf:
                    runs.append(np.array(hf["rs_test_acc"]))
        except Exception as e:
            print(f"Erro ao ler {sf}: {e}")
    if not runs:
        return np.array([])
    min_len = min(len(r) for r in runs)
    return np.array([r[:min_len] for r in runs])


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    G = load_runs(TARGET_DIR)
    A = load_runs(ABRUPT_DIR)
    
    if len(G) == 0:
        print("[ERRO] Nenhuma run encontrada em astra_allgradual_drift!")
        return
        
    n = G.shape[1]
    rounds = np.arange(1, n + 1)
    alpha = alpha_schedule(n)
    ceiling = 100 * np.maximum(alpha, 1 - alpha)
    
    gm = G.mean(0)
    gs = G.std(0) if len(G) > 1 else np.zeros_like(gm)
    
    if len(A) > 0:
        am = A.mean(0)[:n]
        as_ = A.std(0)[:n] if len(A) > 1 else np.zeros_like(am)
    else:
        am = gm
        as_ = gs

    print(f"Processando {len(G)} run(s) de gradual e {len(A)} run(s) de abrupto...")

    # ---------- 01: Run 1 Individual ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rounds, G[0], color='#1f77b4', linewidth=2.5, label='Global Test Accuracy (Run 1)')
    for i, dr in enumerate(range(L, n + 1, L)):
        ax.axvline(x=dr, color='#d62728', linestyle='--', alpha=0.6, linewidth=1.2,
                   label='Concept Drift (Gradual)' if i == 0 else "")
    ax.set_title('ASTRA Benchmark Scenario (Gradual) - Simulation Run #1\n'
                 '(LoRA-CLIP on CIFAR-10 Drift, 20 Clients, Dynamic Sampling 5/Round)', fontweight='bold')
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Global Test Accuracy (%)')
    ax.set_xlim(1, n)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(True)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    p01 = TARGET_DIR / "01_astra_gradual_run1.png"
    fig.savefig(p01, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Gerado: {p01.name}")

    # ---------- 06: Média Geral (Mean General) ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, r in enumerate(G):
        ax.plot(rounds, r, color='#aec7e8', alpha=0.6, linewidth=1.2,
                label='Individual Simulation Run' if i == 0 else "")
    if len(G) > 1:
        ax.fill_between(rounds, gm - gs, gm + gs, color='#1f77b4', alpha=0.25, label='± 1 Std Dev')
    ax.plot(rounds, gm, color='#1f77b4', linewidth=2.8, label=f'Mean Accuracy (N={len(G)} Run{"s" if len(G)>1 else ""})')
    for i, dr in enumerate(range(L, n + 1, L)):
        ax.axvline(x=dr, color='#d62728', linestyle='--', alpha=0.6, linewidth=1.2,
                   label='Concept Drift Event (Every 5 Rounds)' if i == 0 else "")
    ax.set_title(f'Aggregated Benchmark Performance ({len(G)} Trial{"s" if len(G)>1 else ""})\n'
                 'LoRA-CLIP under Periodic Gradual Concept Drift (10 Shifts total)', fontweight='bold')
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Global Test Accuracy (%)')
    ax.set_xlim(1, n)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(True)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    p06 = TARGET_DIR / "06_astra_gradual_mean_general.png"
    fig.savefig(p06, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Gerado: {p06.name}")

    # ---------- 07: Gradual + Alpha + Teto de Bayes ----------
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for r in G:
        ax.plot(rounds, r, color='#aec7e8', lw=0.9, alpha=0.6)
    if len(G) > 1:
        ax.fill_between(rounds, gm - gs, gm + gs, color='#1f77b4', alpha=0.2, label='±1 Std Dev')
    ax.plot(rounds, gm, color='#1f77b4', lw=2.6, marker='o', ms=3.5, label=f'Mean Global Acc (N={len(G)})')
    ax.plot(rounds, ceiling, color='#2ca02c', lw=1.8, ls='--', label=r'Bayes ceiling $\max(\alpha,1-\alpha)$')
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Global Test Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.set_xlim(1, n)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.grid(True)
    ax2 = ax.twinx()
    ax2.step(rounds, alpha, where='mid', color='#d62728', lw=1.4, alpha=0.8, label=r'$\alpha(t)$ (fraction Phase 2)')
    ax2.set_ylabel(r'$\alpha(t)$', color='#d62728')
    ax2.set_ylim(-0.02, 1.6)
    ax2.tick_params(axis='y', colors='#d62728')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='lower left', ncol=2, framealpha=0.9)
    ax.set_title('LoRA-CLIP under Periodic Gradual Drift (CIFAR-10 label swap, L=5)\n'
                 'Mean accuracy vs. theoretical ceiling of the blended test set', fontweight='bold')
    fig.tight_layout()
    p07 = TARGET_DIR / "07_astra_gradual_alpha_ceiling.png"
    fig.savefig(p07, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Gerado: {p07.name}")

    # ---------- 08: Abrupto vs Gradual ----------
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ph = abrupt_phase(n)
    for t in range(n):
        if ph[t] == 1:
            ax.axvspan(t + 0.5, t + 1.5, color='#ff7f0e', alpha=0.07, lw=0)
    if len(A) > 1:
        ax.fill_between(rounds, am - as_, am + as_, color='#d62728', alpha=0.15)
    ax.plot(rounds, am, color='#d62728', lw=2.4, label=f'Abrupt (mean, N={len(A)})')
    if len(G) > 1:
        ax.fill_between(rounds, gm - gs, gm + gs, color='#1f77b4', alpha=0.15)
    ax.plot(rounds, gm, color='#1f77b4', lw=2.4, label=f'Gradual (allgradual, N={len(G)})')
    for dr in range(L, n + 1, L):
        ax.axvline(dr, color='gray', ls=':', lw=1)
    ax.plot([], [], color='#ff7f0e', alpha=0.3, lw=8, label='Phase 2 active (abrupt)')
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Global Test Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.set_xlim(1, n)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.grid(True)
    ax.legend(loc='lower left', ncol=3, framealpha=0.9)
    ax.set_title(f'LoRA-CLIP: Periodic Abrupt vs Gradual Drift (shift every {L} rounds)\n'
                 f'Overall mean: Abrupt {am.mean():.1f}%  |  Gradual {gm.mean():.1f}%', fontweight='bold')
    fig.tight_layout()
    p08 = TARGET_DIR / "08_astra_abrupt_vs_gradual.png"
    fig.savefig(p08, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Gerado: {p08.name}")

    # ---------- 09: Acc vs Alpha (Histerese) ----------
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    up = ((rounds - 1) // L) % 2 == 0
    levels = np.unique(np.round(alpha, 2))
    for mask, name, c, m in [(up, r'$\alpha$ increasing (P1→P2)', '#9467bd', 'o'),
                             (~up, r'$\alpha$ decreasing (P2→P1)', '#8c564b', 's')]:
        mu = [G[:, mask & (np.round(alpha, 2) == lv)].mean() for lv in levels]
        sd = [G[:, mask & (np.round(alpha, 2) == lv)].std() if len(G) > 1 else 0.0 for lv in levels]
        ax.errorbar(levels, mu, yerr=sd, color=c, marker=m, capsize=4, lw=2, label=name)
    xs = np.linspace(0, 1, 101)
    ax.plot(xs, 100 * np.maximum(xs, 1 - xs), 'g--', lw=1.6, label=r'Bayes ceiling $\max(\alpha,1-\alpha)$')
    ax.set_xlabel(r'$\alpha$ (fraction of Phase 2 samples)')
    ax.set_ylabel('Global Test Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.grid(True)
    ax.legend(loc='lower center', framealpha=0.9)
    ax.set_title('Gradual drift: accuracy vs. mixing ratio\n(hysteresis = model lags behind concept)', fontweight='bold')
    fig.tight_layout()
    p09 = TARGET_DIR / "09_astra_gradual_acc_vs_alpha.png"
    fig.savefig(p09, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Gerado: {p09.name}")

    print("\nTodos os 5 gráficos no padrão idêntico foram gerados em astra_allgradual_drift!")


if __name__ == '__main__':
    main()
