#!/usr/bin/env python3
"""
plot_astra_diagnostics.py — Diagnóstico científico do benchmark ASTRA (Abrupto vs Gradual) com LoRA-CLIP.

Gera no diretório de saída (padrão: --gradual-exp):
  07_astra_gradual_alpha_ceiling.png          -> Acurácia no conceito vigente (G-active) + alpha(t) + teto de Bayes
  08_astra_abrupt_vs_gradual.png              -> Comparação direta (Abrupto vs Gradual) no conceito vigente
  09_astra_gradual_acc_vs_alpha.png           -> Histerese: G-active e (G-orig / G-drift) vs alpha (subindo vs descendo)
  10_astra_decomposed_abrupt_and_gradual.png  -> Decomposição completa: G-active, C-mix, G-orig, G-drift e choque pré-treino

Observação metodológica:
- Para experimentos com `--global-eval official|both`, reconstrói exatamente a acurácia global oficial
  no conceito vigente: G-active(r) = (1 - alpha(r)) * G-orig(r) + alpha(r) * G-drift(r).
- Para experimentos legados, utiliza `rs_test_acc` diretamente.
"""
import sys
import types
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
from utils.drift_schedule import phase_for_round

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'savefig.dpi': 300,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

RESULTS = _SRC_DIR / "results"
L = 5  # drift_freq padrão


def _extract_server_metrics(hf):
    """Extrai métricas do server_*.h5 reconstruindo G-active quando houver avaliação oficial."""
    has_official = (
        'rs_test_acc_orig' in hf
        and 'rs_test_acc_drift' in hf
        and bool(hf.attrs.get('drift_enabled', False))
    )
    if has_official:
        g_orig = np.array(hf['rs_test_acc_orig'], dtype=float)
        g_drift = np.array(hf['rs_test_acc_drift'], dtype=float)
        n = min(len(g_orig), len(g_drift))
        args = types.SimpleNamespace(
            drift=True,
            drift_type=str(hf.attrs.get('drift_type', 'abrupt')),
            drift_round=int(hf.attrs.get('drift_round', 10)),
            drift_length=int(hf.attrs.get('drift_length', 5)),
            drift_freq=int(hf.attrs.get('drift_freq', L)),
        )
        g_active = np.empty(n, dtype=float)
        for r in range(1, n + 1):
            st = phase_for_round(r, args, is_drifted_client=True, legacy_gradual_server_hold=False)
            if st.phase == 'phase1':
                g_active[r - 1] = g_orig[r - 1]
            elif st.phase == 'phase2':
                g_active[r - 1] = g_drift[r - 1]
            else:
                g_active[r - 1] = (1.0 - st.alpha) * g_orig[r - 1] + st.alpha * g_drift[r - 1]
        return {
            'active': g_active,
            'orig': g_orig[:n],
            'drift': g_drift[:n],
            'clients': np.array(hf['rs_test_acc_clients'], dtype=float)[:n] if 'rs_test_acc_clients' in hf else None,
            'aux': np.array(hf['rs_aux_simple_avg_acc'], dtype=float)[:n] if 'rs_aux_simple_avg_acc' in hf else None,
            'has_official': True,
        }
    acc = np.array(hf['rs_test_acc'], dtype=float)
    return {
        'active': acc,
        'orig': None,
        'drift': None,
        'clients': None,
        'aux': np.array(hf['rs_aux_simple_avg_acc'], dtype=float)[:len(acc)] if 'rs_aux_simple_avg_acc' in hf else None,
        'has_official': False,
    }


def load_runs_rich(exp):
    """Carrega todas as runs de `exp`, retornando matriz de G-active e dicionário de séries auxiliares."""
    items = []
    for sf in sorted((RESULTS / exp).glob("server_*.h5")):
        with h5py.File(sf, "r") as hf:
            items.append(_extract_server_metrics(hf))
    if not items:
        return np.empty((0, 0)), {}
    n = min(len(it['active']) for it in items)
    active = np.array([it['active'][:n] for it in items])
    meta = {'has_official': all(it['has_official'] for it in items)}
    for key in ('orig', 'drift', 'clients', 'aux'):
        if all(it[key] is not None for it in items):
            meta[key] = np.array([it[key][:n] for it in items])
        else:
            meta[key] = None
    return active, meta


def alpha_schedule(n_rounds, L=L):
    """Mesma fórmula de drift_schedule.py (drift periódico gradual)."""
    a = []
    for t in range(1, n_rounds + 1):
        p, s = (t - 1) // L, (t - 1) % L
        r = s / (L - 1) if L > 1 else 0.0
        a.append(r if p % 2 == 0 else 1.0 - r)
    return np.array(a)


def abrupt_phase(n_rounds, L=L):
    """1 = Phase 2 ativa, 0 = Phase 1 ativa (mesma regra de drift_schedule.py)."""
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
    G, G_meta = load_runs_rich(args.gradual_exp)
    A, A_meta = load_runs_rich(args.abrupt_exp)
    if len(G) == 0:
        print(f"[ERRO] Nenhuma run encontrada em {args.gradual_exp}")
        return

    n = G.shape[1]
    rounds = np.arange(1, n + 1)
    alpha = alpha_schedule(n)
    ceiling = 100.0 * np.maximum(alpha, 1.0 - alpha)

    gm = G.mean(0)
    gs = G.std(0) if len(G) > 1 else np.zeros_like(gm)
    if len(A) > 0:
        am = A.mean(0)[:n]
        as_ = A.std(0)[:n] if len(A) > 1 else np.zeros(n)
    else:
        am, as_ = gm, gs

    metric_label = "Official 10k Active Concept (G-active)" if G_meta.get('has_official') else "Global Test Accuracy"

    # ---------- 07: gradual + alpha + teto ----------
    fig, ax = plt.subplots(figsize=(11, 5.8))
    if len(G) > 1:
        for r in G:
            ax.plot(rounds, r, color='#aec7e8', lw=0.9, alpha=0.6)
        ax.fill_between(rounds, gm - gs, gm + gs, color='#1f77b4', alpha=0.2, label='±1 Std Dev')
    ax.plot(rounds, gm, color='#1f77b4', lw=2.6, marker='o', ms=4.0,
            label=f'{metric_label} (N={len(G)}, mean={gm.mean():.1f}%)')
    if G_meta.get('clients') is not None:
        cm_g = G_meta['clients'].mean(0)[:n]
        ax.plot(rounds, cm_g, color='#ff7f0e', lw=1.5, ls=':', marker='x', ms=4.0, alpha=0.9,
                label=f'Client Test Mix C-mix (mean={cm_g.mean():.1f}%)')
    ax.plot(rounds, ceiling, color='#2ca02c', lw=2.0, ls='--', label=r'Bayes ceiling $\max(\alpha, 1-\alpha)$')
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Global Test Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.set_xlim(1, n)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(True)
    ax2 = ax.twinx()
    ax2.step(rounds, alpha, where='mid', color='#d62728', lw=1.4, alpha=0.75, label=r'$\alpha(t)$ (fraction Phase 2)')
    ax2.set_ylabel(r'$\alpha(t)$', color='#d62728')
    ax2.set_ylim(-0.02, 1.55)
    ax2.tick_params(axis='y', colors='#d62728')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='lower left', ncol=2, framealpha=0.92)
    ax.set_title(
        f'LoRA-CLIP under Periodic Gradual Drift (ASTRA CIFAR-10, L={L})\n'
        f'Active-concept accuracy vs. Bayes ceiling (efficiency = {100.0 * (gm / ceiling).mean():.1f}%)',
        fontweight='bold',
    )
    fig.tight_layout()
    p07 = out_dir / "07_astra_gradual_alpha_ceiling.png"
    fig.savefig(p07, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Salvo: {p07}")

    # ---------- 08: abrupto vs gradual ----------
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ph = abrupt_phase(n)
    for t in range(n):
        if ph[t] == 1:
            ax.axvspan(t + 0.5, t + 1.5, color='#ff7f0e', alpha=0.07, lw=0)
    if len(A) > 1:
        ax.fill_between(rounds, am - as_, am + as_, color='#d62728', alpha=0.15)
    ax.plot(rounds, am, color='#d62728', lw=2.5, marker='s', ms=3.5,
            label=f'Abrupt G-active (N={len(A)}, mean={am.mean():.1f}%)')
    if len(G) > 1:
        ax.fill_between(rounds, gm - gs, gm + gs, color='#1f77b4', alpha=0.15)
    ax.plot(rounds, gm, color='#1f77b4', lw=2.5, marker='o', ms=3.5,
            label=f'Gradual G-active (N={len(G)}, mean={gm.mean():.1f}%)')
    ax.plot(rounds, ceiling, color='#2ca02c', lw=1.4, ls='--', alpha=0.8,
            label=r'Gradual Bayes ceiling $\max(\alpha, 1-\alpha)$')
    for dr in range(L, n + 1, L):
        ax.axvline(dr, color='gray', ls=':', lw=1)
    ax.plot([], [], color='#ff7f0e', alpha=0.25, lw=8, label='Phase 2 active (abrupt)')
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Global Test Accuracy — Active Concept (%)')
    ax.set_ylim(0, 105)
    ax.set_xlim(1, n)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(True)
    ax.legend(loc='lower left', ncol=2, framealpha=0.92)
    ax.set_title(
        f'LoRA-CLIP (ASTRA CIFAR-10, α=0.5, Zero Leakage): Abrupt vs. Gradual Drift (L={L})\n'
        f'Official 10k Active Concept — Abrupt mean: {am.mean():.2f}%  |  Gradual mean: {gm.mean():.2f}%',
        fontweight='bold',
    )
    fig.tight_layout()
    p08 = out_dir / "08_astra_abrupt_vs_gradual.png"
    fig.savefig(p08, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Salvo: {p08}")

    # ---------- 09: acc vs alpha (histerese: G-active + decomposição G-orig/G-drift) ----------
    up = ((rounds - 1) // L) % 2 == 0
    levels = np.unique(np.round(alpha, 2))
    has_decomp = G_meta.get('orig') is not None and G_meta.get('drift') is not None

    if has_decomp:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    else:
        fig, ax1 = plt.subplots(figsize=(7.5, 5.6))
        ax2 = None

    for mask, name, c, m in [
        (up, r'$\alpha$ increasing (P1 $\rightarrow$ P2)', '#9467bd', 'o'),
        (~up, r'$\alpha$ decreasing (P2 $\rightarrow$ P1)', '#8c564b', 's'),
    ]:
        mu = [G[:, mask & (np.round(alpha, 2) == lv)].mean() for lv in levels]
        sd = [G[:, mask & (np.round(alpha, 2) == lv)].std() for lv in levels]
        ax1.errorbar(levels, mu, yerr=sd, color=c, marker=m, capsize=4, lw=2.2, label=name)
    xs = np.linspace(0, 1, 101)
    ax1.plot(xs, 100.0 * np.maximum(xs, 1.0 - xs), 'g--', lw=1.8, label=r'Bayes ceiling $\max(\alpha, 1-\alpha)$')
    ax1.set_xlabel(r'$\alpha$ (fraction of Phase 2 samples)')
    ax1.set_ylabel('Global Test Accuracy — Active Concept (%)')
    ax1.set_ylim(0, 105)
    ax1.set_xticks(levels)
    ax1.grid(True)
    ax1.legend(loc='lower center', framealpha=0.92)
    ax1.set_title('Active-Concept Accuracy (G-active) vs. Mixing Ratio', fontweight='bold')

    if ax2 is not None:
        go = G_meta['orig']
        gd = G_meta['drift']
        for arr, concept_lbl, col in [(go, 'G-orig (Phase 1)', '#1f77b4'), (gd, 'G-drift (Phase 2)', '#d62728')]:
            mu_up = [arr[:, up & (np.round(alpha, 2) == lv)].mean() for lv in levels]
            mu_dn = [arr[:, (~up) & (np.round(alpha, 2) == lv)].mean() for lv in levels]
            ax2.plot(levels, mu_up, color=col, marker='o', lw=2.2, ls='-', label=f'{concept_lbl} — P1→P2')
            ax2.plot(levels, mu_dn, color=col, marker='s', lw=2.0, ls='--', alpha=0.8, label=f'{concept_lbl} — P2→P1')
        ax2.axvline(0.5, color='gray', ls=':', lw=1.2)
        ax2.set_xlabel(r'$\alpha$ (fraction of Phase 2 samples)')
        ax2.set_ylabel('Official 10k Test Accuracy (%)')
        ax2.set_ylim(0, 105)
        ax2.set_xticks(levels)
        ax2.grid(True)
        ax2.legend(loc='center left', bbox_to_anchor=(0.01, 0.5), fontsize=9, framealpha=0.92)
        ax2.set_title('Concept Hysteresis Loop (G-orig vs. G-drift)', fontweight='bold')

    fig.suptitle(
        'Gradual Concept Drift Hysteresis — LoRA-CLIP on ASTRA CIFAR-10 (α=0.5, Zero Leakage)',
        fontsize=14, fontweight='bold', y=1.02,
    )
    fig.tight_layout()
    p09 = out_dir / "09_astra_gradual_acc_vs_alpha.png"
    fig.savefig(p09, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Salvo: {p09}")

    # ---------- 10: decomposição completa (G-active, C-mix, G-orig, G-drift, ClientAux) ----------
    if A_meta.get('has_official') and G_meta.get('has_official'):
        fig, (ax_a, ax_g) = plt.subplots(2, 1, figsize=(12.5, 9.2), sharex=True)

        # Painel Superior: Abrupto
        for t in range(n):
            if ph[t] == 1:
                ax_a.axvspan(t + 0.5, t + 1.5, color='#ff7f0e', alpha=0.07, lw=0)
        ax_a.plot(rounds, A_meta['orig'].mean(0)[:n], color='#1f77b4', lw=1.4, ls='--', alpha=0.65,
                  label='G-orig (Official 10k — Phase 1 fixed)')
        ax_a.plot(rounds, A_meta['drift'].mean(0)[:n], color='#ff7f0e', lw=1.4, ls='--', alpha=0.65,
                  label='G-drift (Official 10k — Phase 2 fixed)')
        ax_a.plot(rounds, am, color='#2ca02c', lw=2.8, marker='o', ms=3.5,
                  label=f'G-active (Official 10k active concept, mean={am.mean():.2f}%)')
        if A_meta.get('clients') is not None:
            cm_a = A_meta['clients'].mean(0)[:n]
            ax_a.plot(rounds, cm_a, color='#111111', lw=1.3, ls=':', marker='x', ms=4.0, alpha=0.85,
                      label=f'C-mix (20-client test mix, mean={cm_a.mean():.2f}%)')
        if A_meta.get('aux') is not None:
            aux_a = A_meta['aux'].mean(0)[:n]
            ax_a.plot(rounds, aux_a, color='#d62728', lw=1.3, ls='-.', alpha=0.75,
                      label='Pre-train Client Eval (captures immediate shock before local SGD)')
        for dr in range(L, n + 1, L):
            ax_a.axvline(dr, color='gray', ls=':', lw=0.9)
        ax_a.set_ylabel('Accuracy (%)')
        ax_a.set_ylim(0, 105)
        ax_a.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax_a.grid(True)
        ax_a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), fontsize=9, ncol=3, framealpha=0.95)
        ax_a.set_title('A) Periodic Abrupt Drift — Decomposed Official & Client Server Metrics', fontweight='bold')

        # Painel Inferior: Gradual
        ax_g.plot(rounds, G_meta['orig'].mean(0)[:n], color='#1f77b4', lw=1.5, ls='--', alpha=0.7,
                  label='G-orig (Official 10k — Phase 1 fixed)')
        ax_g.plot(rounds, G_meta['drift'].mean(0)[:n], color='#ff7f0e', lw=1.5, ls='--', alpha=0.7,
                  label='G-drift (Official 10k — Phase 2 fixed)')
        ax_g.plot(rounds, gm, color='#2ca02c', lw=2.8, marker='o', ms=3.5,
                  label=f'G-active (Official 10k active concept, mean={gm.mean():.2f}%)')
        if G_meta.get('clients') is not None:
            cm_g = G_meta['clients'].mean(0)[:n]
            ax_g.plot(rounds, cm_g, color='#111111', lw=1.3, ls=':', marker='x', ms=4.0, alpha=0.85,
                      label=f'C-mix (20-client test mix, mean={cm_g.mean():.2f}%)')
        ax_g.plot(rounds, ceiling, color='#9467bd', lw=1.8, ls=':',
                  label=r'Bayes ceiling $\max(\alpha, 1-\alpha)$')
        for dr in range(L, n + 1, L):
            ax_g.axvline(dr, color='gray', ls=':', lw=0.9)
        ax_g.set_xlabel('Communication Round')
        ax_g.set_ylabel('Accuracy (%)')
        ax_g.set_ylim(0, 105)
        ax_g.set_xlim(1, n)
        ax_g.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax_g.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax_g.grid(True)
        ax_g.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), fontsize=9, ncol=3, framealpha=0.95)
        ax_g.set_title('B) Periodic Gradual Drift — Decomposed Official & Client Server Metrics', fontweight='bold')

        fig.suptitle(
            'Full Server Evaluation Decomposition — LoRA-CLIP on ASTRA CIFAR-10 (α=0.5, Zero Data Leakage)',
            fontsize=14, fontweight='bold', y=0.995,
        )
        fig.tight_layout(h_pad=3.2)
        p10 = out_dir / "10_astra_decomposed_abrupt_and_gradual.png"
        fig.savefig(p10, bbox_inches='tight')
        plt.close(fig)
        print(f"[OK] Salvo: {p10}")

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
