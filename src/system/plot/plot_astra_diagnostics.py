#!/usr/bin/env python3
"""
plot_astra_diagnostics.py — Diagnóstico científico do benchmark ASTRA (Abrupto vs Gradual) com LoRA-CLIP.

Gera em src/results/astra_periodic_gradual_drift/:
  07_astra_gradual_alpha_ceiling.png   -> Acurácia média do gradual + alpha(t) + teto de Bayes max(alpha, 1-alpha)
  08_astra_abrupt_vs_gradual.png       -> Comparação direta das médias (Abrupto vs Gradual) com ±1 std
  09_astra_gradual_acc_vs_alpha.png    -> Acurácia vs alpha (ramo ascendente vs descendente: histerese/lag)

Observação metodológica: com label_swap aplicado a TODAS as classes e teste "blended" na mesma
proporção alpha, a acurácia máxima atingível por qualquer classificador f(x) é max(alpha, 1-alpha),
pois metade (alpha=0.5) das amostras com a mesma imagem possui rótulos conflitantes.
"""
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13,
                     'legend.fontsize': 10, 'savefig.dpi': 300, 'grid.alpha': 0.3, 'grid.linestyle': '--'})

RESULTS = Path(__file__).resolve().parent.parent.parent / "results"
L = 5  # drift_freq


def load_runs(exp):
    runs = []
    for sf in sorted((RESULTS / exp).glob("server_*.h5")):
        with h5py.File(sf, "r") as hf:
            runs.append(np.array(hf["rs_test_acc"]))
    n = min(len(r) for r in runs)
    return np.array([r[:n] for r in runs])


def alpha_schedule(n_rounds, L=L):
    """Mesma fórmula de client.py/server.py (drift periódico gradual)."""
    a = []
    for t in range(1, n_rounds + 1):
        p, s = (t - 1) // L, (t - 1) % L
        r = s / (L - 1) if L > 1 else 0.0
        a.append(r if p % 2 == 0 else 1.0 - r)
    return np.array(a)


def abrupt_phase(n_rounds, L=L):
    """1 = Phase 2 ativa, 0 = Phase 1 ativa (mesma regra do cliente)."""
    ph, cur = [], 0
    for t in range(1, n_rounds + 1):
        if t % L == 0:
            cur = 1 if (t // L) % 2 == 1 else 0
        ph.append(cur)
    return np.array(ph)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ASTRA diagnostics plot generator")
    parser.add_argument("--gradual-exp", type=str, default="astra_periodic_gradual_drift", help="Gradual experiment subfolder")
    parser.add_argument("--abrupt-exp", type=str, default="astra_periodic_drift", help="Abrupt experiment subfolder")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (defaults to gradual-exp)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS / args.gradual_exp
    out_dir.mkdir(parents=True, exist_ok=True)
    G = load_runs(args.gradual_exp)
    A = load_runs(args.abrupt_exp)
    if len(G) == 0:
        print(f"[ERRO] Nenhuma run encontrada em {args.gradual_exp}")
        return
    n = G.shape[1]
    rounds = np.arange(1, n + 1)
    alpha = alpha_schedule(n)
    ceiling = 100 * np.maximum(alpha, 1 - alpha)
    gm, gs = G.mean(0), G.std(0) if len(G) > 1 else np.zeros_like(gm)
    if len(A) > 0:
        am, as_ = A.mean(0)[:n], A.std(0)[:n] if len(A) > 1 else np.zeros(n)
    else:
        am, as_ = gm, gs

    # ---------- 07: gradual + alpha + teto ----------
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for r in G:
        ax.plot(rounds, r, color='#aec7e8', lw=0.9, alpha=0.6)
    ax.fill_between(rounds, gm - gs, gm + gs, color='#1f77b4', alpha=0.2, label='±1 Std Dev')
    ax.plot(rounds, gm, color='#1f77b4', lw=2.6, marker='o', ms=3.5, label=f'Mean Global Acc (N={len(G)})')
    ax.plot(rounds, ceiling, color='#2ca02c', lw=1.8, ls='--', label=r'Bayes ceiling $\max(\alpha,1-\alpha)$')
    ax.set_xlabel('Communication Round'); ax.set_ylabel('Global Test Accuracy (%)')
    ax.set_ylim(0, 100); ax.set_xlim(1, n)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5)); ax.grid(True)
    ax2 = ax.twinx()
    ax2.step(rounds, alpha, where='mid', color='#d62728', lw=1.4, alpha=0.8, label=r'$\alpha(t)$ (fraction Phase 2)')
    ax2.set_ylabel(r'$\alpha(t)$', color='#d62728'); ax2.set_ylim(-0.02, 1.6)
    ax2.tick_params(axis='y', colors='#d62728')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='lower left', ncol=2, framealpha=0.9)
    ax.set_title('LoRA-CLIP under Periodic Gradual Drift (CIFAR-10 label swap, L=5)\n'
                 'Mean accuracy vs. theoretical ceiling of the blended test set', fontweight='bold')
    fig.tight_layout(); fig.savefig(out_dir / "07_astra_gradual_alpha_ceiling.png", bbox_inches='tight'); plt.close(fig)

    # ---------- 08: abrupto vs gradual ----------
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ph = abrupt_phase(n)
    for t in range(n):
        if ph[t] == 1:
            ax.axvspan(t + 0.5, t + 1.5, color='#ff7f0e', alpha=0.07, lw=0)
    ax.fill_between(rounds, am - as_, am + as_, color='#d62728', alpha=0.15)
    ax.plot(rounds, am, color='#d62728', lw=2.4, label=f'Abrupt (mean, N={len(A)})')
    ax.fill_between(rounds, gm - gs, gm + gs, color='#1f77b4', alpha=0.15)
    ax.plot(rounds, gm, color='#1f77b4', lw=2.4, label=f'Gradual (mean, N={len(G)})')
    for dr in range(L, n + 1, L):
        ax.axvline(dr, color='gray', ls=':', lw=1)
    ax.plot([], [], color='#ff7f0e', alpha=0.3, lw=8, label='Phase 2 active (abrupt)')
    ax.set_xlabel('Communication Round'); ax.set_ylabel('Global Test Accuracy (%)')
    ax.set_ylim(0, 100); ax.set_xlim(1, n)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5)); ax.grid(True)
    ax.legend(loc='lower left', ncol=3, framealpha=0.9)
    ax.set_title(f'LoRA-CLIP: Periodic Abrupt vs Gradual Drift (shift every {L} rounds)\n'
                 f'Overall mean: Abrupt {am.mean():.1f}%  |  Gradual {gm.mean():.1f}%', fontweight='bold')
    fig.tight_layout(); fig.savefig(out_dir / "08_astra_abrupt_vs_gradual.png", bbox_inches='tight'); plt.close(fig)

    # ---------- 09: acc vs alpha (histerese) ----------
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    # ramo: alpha subindo (ciclo par) vs descendo (ciclo ímpar)
    up = ((rounds - 1) // L) % 2 == 0
    levels = np.unique(np.round(alpha, 2))
    for mask, name, c, m in [(up, r'$\alpha$ increasing (P1→P2)', '#9467bd', 'o'),
                             (~up, r'$\alpha$ decreasing (P2→P1)', '#8c564b', 's')]:
        mu = [G[:, mask & (np.round(alpha, 2) == lv)].mean() for lv in levels]
        sd = [G[:, mask & (np.round(alpha, 2) == lv)].std() for lv in levels]
        ax.errorbar(levels, mu, yerr=sd, color=c, marker=m, capsize=4, lw=2, label=name)
    xs = np.linspace(0, 1, 101)
    ax.plot(xs, 100 * np.maximum(xs, 1 - xs), 'g--', lw=1.6, label=r'Bayes ceiling $\max(\alpha,1-\alpha)$')
    ax.set_xlabel(r'$\alpha$ (fraction of Phase 2 samples)'); ax.set_ylabel('Global Test Accuracy (%)')
    ax.set_ylim(0, 100); ax.grid(True); ax.legend(loc='lower center', framealpha=0.9)
    ax.set_title('Gradual drift: accuracy vs. mixing ratio\n(hysteresis = model lags behind concept)', fontweight='bold')
    fig.tight_layout(); fig.savefig(out_dir / "09_astra_gradual_acc_vs_alpha.png", bbox_inches='tight'); plt.close(fig)

    # ---------- resumo numérico ----------
    print("Gradual: mean %.2f | normalized (acc/ceiling) %.3f" % (gm.mean(), (gm / ceiling).mean()))
    print("Abrupt : mean %.2f" % am.mean())
    print("Abrupt  1st half %.2f | 2nd half %.2f" % (am[:n // 2].mean(), am[n // 2:].mean()))
    print("Gradual 1st half %.2f | 2nd half %.2f" % (gm[:n // 2].mean(), gm[n // 2:].mean()))
    print("Abrupt  run-final-acc:", np.round(A[:, -1], 1), " inter-run std (mean over rounds): %.2f" % as_.mean())
    print("Gradual run-final-acc:", np.round(G[:, -1], 1), " inter-run std (mean over rounds): %.2f" % gs.mean())
    dips = [am[t - 1] for t in range(L, n + 1, L)]
    pre = [am[t - 2] for t in range(L, n + 1, L)]
    print("Abrupt mean drop at drift rounds (pre -> drift):", [f"{p:.0f}->{d:.0f}" for p, d in zip(pre, dips)])
    for lv in levels:
        print(f"  alpha={lv:.2f}: up={G[:, up & (np.round(alpha,2)==lv)].mean():.1f}  down={G[:, ~up & (np.round(alpha,2)==lv)].mean():.1f}  ceiling={100*max(lv,1-lv):.0f}")


if __name__ == "__main__":
    main()
