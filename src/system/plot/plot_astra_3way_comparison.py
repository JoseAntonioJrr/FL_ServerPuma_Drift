#!/usr/bin/env python3
"""
ASTRA 3-Way Comparative Benchmark Plot:
Stationary Baseline (No Drift) vs. Periodic Abrupt Drift vs. Periodic Gradual Drift
LoRA-CLIP on CIFAR-100 (Dirichlet alpha=0.1, 1 Local Epoch, 20 Clients, Zero Data Leakage)
"""

import sys
from pathlib import Path
import types
import h5py
import numpy as np
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
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
})

RESULTS = _SRC_DIR / "results"
L = 5


def extract_metrics(hf):
    has_official = (
        'rs_test_acc_orig' in hf
        and 'rs_test_acc_drift' in hf
        and bool(hf.attrs.get('drift_enabled', False))
    )
    loss = np.array(hf['rs_test_loss'], dtype=float) if 'rs_test_loss' in hf else None
    clients_acc = np.array(hf['rs_test_acc_clients'], dtype=float) if 'rs_test_acc_clients' in hf else None
    
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
            'active_acc': g_active,
            'orig_acc': g_orig[:n],
            'drift_acc': g_drift[:n],
            'clients_acc': clients_acc[:n] if clients_acc is not None else None,
            'loss': loss[:n] if loss is not None else None,
            'has_drift': True,
        }
    
    acc = np.array(hf['rs_test_acc'], dtype=float)
    n = len(acc)
    return {
        'active_acc': acc,
        'orig_acc': acc,
        'drift_acc': acc,
        'clients_acc': clients_acc[:n] if clients_acc is not None else None,
        'loss': loss[:n] if loss is not None else None,
        'has_drift': False,
    }


def load_exp_data(exp_name: str):
    target_dir = RESULTS / exp_name
    server_files = sorted(target_dir.glob("server_*.h5"))
    if not server_files:
        raise FileNotFoundError(f"Nenhum server_*.h5 encontrado em {target_dir}")
    
    # Load first run (or average if multiple)
    with h5py.File(server_files[0], 'r') as hf:
        data = extract_metrics(hf)
    return data


def alpha_schedule(n_rounds, L=5):
    a = []
    for t in range(1, n_rounds + 1):
        p, s = (t - 1) // L, (t - 1) % L
        r = s / (L - 1) if L > 1 else 0.0
        a.append(r if p % 2 == 0 else 1.0 - r)
    return np.array(a)


def abrupt_phase(n_rounds, L=5):
    ph, cur = [], 0
    for t in range(1, n_rounds + 1):
        if t % L == 0:
            cur = 1 if (t // L) % 2 == 1 else 0
        ph.append(cur)
    return np.array(ph)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ASTRA 3-Way Comparative Benchmark Plot")
    parser.add_argument("--nodrift-exp", type=str, default="clip_astra_cifar100_a0.1_e1_nodrift")
    parser.add_argument("--abrupt-exp", type=str, default="clip_astra_cifar100_a0.1_e1_abrupt_drift")
    parser.add_argument("--gradual-exp", type=str, default="clip_astra_cifar100_a0.1_e1_gradual_drift")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS / args.nodrift_exp
    out_dir.mkdir(parents=True, exist_ok=True)

    nodrift = load_exp_data(args.nodrift_exp)
    abrupt = load_exp_data(args.abrupt_exp)
    gradual = load_exp_data(args.gradual_exp)

    n = min(len(nodrift['active_acc']), len(abrupt['active_acc']), len(gradual['active_acc']))
    rounds = np.arange(1, n + 1)

    s_acc = nodrift['active_acc'][:n]
    a_acc = abrupt['active_acc'][:n]
    g_acc = gradual['active_acc'][:n]

    alpha = alpha_schedule(n)
    ceiling = 100.0 * np.maximum(alpha, 1.0 - alpha)
    ph = abrupt_phase(n)

    # ==========================================
    # FIGURE 1: Primary 3-Way Benchmark Trajectory
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 6.5))

    # Highlight Phase 2 regions (Abrupt)
    for t in range(n):
        if ph[t] == 1:
            ax.axvspan(t + 0.5, t + 1.5, color='#ff7f0e', alpha=0.08, lw=0)

    # Drift transition boundary lines
    for dr in range(L, n + 1, L):
        ax.axvline(dr, color='gray', linestyle=':', linewidth=1.1, alpha=0.7)

    # Plot lines
    ax.plot(rounds, s_acc, color='#2ca02c', linewidth=2.8, marker='^', markersize=4.5,
            label=f'Stationary Baseline (No Drift) — Peak: {s_acc.max():.1f}%, Final: {s_acc[-1]:.1f}%, Mean: {s_acc.mean():.1f}%')
    ax.plot(rounds, a_acc, color='#d62728', linewidth=2.4, marker='s', markersize=4.0,
            label=f'Abrupt Drift (G-active) — Peak: {a_acc.max():.1f}%, Final: {a_acc[-1]:.1f}%, Mean: {a_acc.mean():.1f}%')
    ax.plot(rounds, g_acc, color='#1f77b4', linewidth=2.4, marker='o', markersize=4.0,
            label=f'Gradual Drift (G-active) — Peak: {g_acc.max():.1f}%, Final: {g_acc[-1]:.1f}%, Mean: {g_acc.mean():.1f}%')
    
    # Bayes Ceiling for gradual drift
    ax.plot(rounds, ceiling, color='#7f7f7f', linestyle='--', linewidth=1.5, alpha=0.75,
            label=r'Gradual Bayes Ceiling $\max(\alpha, 1-\alpha)$')

    # Fake element for legend shading
    ax.plot([], [], color='#ff7f0e', alpha=0.3, linewidth=8, label='Phase 2 active window (Abrupt)')

    ax.set_title(
        'LoRA-CLIP on CIFAR-100 (ASTRA Dirichlet α=0.1, 1 Local Epoch, 20 Clients, Zero Leakage)\n'
        'Complete Benchmark: Stationary Baseline vs. Periodic Abrupt vs. Periodic Gradual Drift (L=5)',
        fontweight='bold', fontsize=13
    )
    ax.set_xlabel('Communication Round', fontweight='bold')
    ax.set_ylabel('Global Test Accuracy (%) — Active Concept', fontweight='bold')
    ax.set_xlim(1, n)
    ax.set_ylim(0, 105)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(True)
    ax.legend(loc='lower left', framealpha=0.95, edgecolor='#cccccc')

    fig.tight_layout()
    p1 = out_dir / "08_astra_3way_comparison.png"
    fig.savefig(p1, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Salvo: {p1}")

    # ==========================================
    # FIGURE 2: Multi-panel Diagnostic Comparison
    # ==========================================
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)

    # Panel A: Accuracy Curves
    ax_a = axes[0]
    for t in range(n):
        if ph[t] == 1:
            ax_a.axvspan(t + 0.5, t + 1.5, color='#ff7f0e', alpha=0.08, lw=0)
    for dr in range(L, n + 1, L):
        ax_a.axvline(dr, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)

    ax_a.plot(rounds, s_acc, color='#2ca02c', lw=2.5, marker='^', ms=3.5, label=f'Stationary Baseline (Mean: {s_acc.mean():.1f}%)')
    ax_a.plot(rounds, a_acc, color='#d62728', lw=2.2, marker='s', ms=3.5, label=f'Abrupt Drift (Mean: {a_acc.mean():.1f}%)')
    ax_a.plot(rounds, g_acc, color='#1f77b4', lw=2.2, marker='o', ms=3.5, label=f'Gradual Drift (Mean: {g_acc.mean():.1f}%)')
    ax_a.plot(rounds, ceiling, color='#7f7f7f', ls='--', lw=1.3, alpha=0.75, label=r'Bayes Ceiling $\max(\alpha, 1-\alpha)$')
    ax_a.set_ylabel('Accuracy (%)', fontweight='bold')
    ax_a.set_ylim(0, 105)
    ax_a.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax_a.grid(True)
    ax_a.legend(loc='lower left', ncol=2, framealpha=0.92, fontsize=9.5)
    ax_a.set_title('A) Global Test Accuracy under Active Concept (G-active)', fontweight='bold')

    # Panel B: Test Loss
    ax_b = axes[1]
    for dr in range(L, n + 1, L):
        ax_b.axvline(dr, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
    if nodrift['loss'] is not None:
        ax_b.plot(rounds, nodrift['loss'][:n], color='#2ca02c', lw=2.5, marker='^', ms=3.5,
                  label=f'Stationary (Loss: {nodrift["loss"][:n].min():.2f} - {nodrift["loss"][:n].max():.2f}, Final: {nodrift["loss"][:n][-1]:.2f})')
    if abrupt['loss'] is not None:
        ax_b.plot(rounds, abrupt['loss'][:n], color='#d62728', lw=2.0, marker='s', ms=3.5,
                  label=f'Abrupt (Loss: {abrupt["loss"][:n].min():.2f} - {abrupt["loss"][:n].max():.2f}, Final: {abrupt["loss"][:n][-1]:.2f})')
    if gradual['loss'] is not None:
        ax_b.plot(rounds, gradual['loss'][:n], color='#1f77b4', lw=2.0, marker='o', ms=3.5,
                  label=f'Gradual (Loss: {gradual["loss"][:n].min():.2f} - {gradual["loss"][:n].max():.2f}, Final: {gradual["loss"][:n][-1]:.2f})')
    ax_b.set_ylabel('Test Loss', fontweight='bold')
    ax_b.grid(True)
    ax_b.legend(loc='upper right', framealpha=0.92, fontsize=9.5)
    ax_b.set_title('B) Cross-Entropy Test Loss Evolution', fontweight='bold')

    # Panel C: Data Leakage Verification (|G_active - C_mix|)
    ax_c = axes[2]
    for dr in range(L, n + 1, L):
        ax_c.axvline(dr, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
    
    if nodrift['clients_acc'] is not None:
        gap_s = np.abs(s_acc - nodrift['clients_acc'][:n])
        ax_c.plot(rounds, gap_s, color='#2ca02c', lw=2.5, marker='^', ms=3.5,
                  label=f'Stationary Gap $|G_{{test}} - C_{{mix}}|$ (Mean: {gap_s.mean():.2f}%, Max: {gap_s.max():.2f}%)')
    if abrupt['clients_acc'] is not None:
        gap_a = np.abs(a_acc - abrupt['clients_acc'][:n])
        ax_c.plot(rounds, gap_a, color='#d62728', lw=2.0, marker='s', ms=3.5,
                  label=f'Abrupt Gap $|G_{{active}} - C_{{mix}}|$ (Mean: {gap_a.mean():.2f}%, Max: {gap_a.max():.2f}%)')
    if gradual['clients_acc'] is not None:
        gap_g = np.abs(g_acc - gradual['clients_acc'][:n])
        ax_c.plot(rounds, gap_g, color='#1f77b4', lw=2.0, marker='o', ms=3.5,
                  label=f'Gradual Gap $|G_{{active}} - C_{{mix}}|$ (Mean: {gap_g.mean():.2f}%, Max: {gap_g.max():.2f}%)')

    ax_c.set_xlabel('Communication Round', fontweight='bold')
    ax_c.set_ylabel('Gap (p.p.)', fontweight='bold')
    ax_c.set_xlim(1, n)
    ax_c.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax_c.grid(True)
    ax_c.legend(loc='upper left', framealpha=0.92, fontsize=9.5)
    ax_c.set_title('C) Rigorous Zero Data Leakage Verification: Gap between Server Active Test (10k) and Local Test Mix', fontweight='bold')

    fig.suptitle(
        'Comprehensive Multi-Panel Benchmark: Stationary vs. Abrupt vs. Gradual Drift\n(LoRA-CLIP on CIFAR-100, Dirichlet α=0.1, 1 Local Epoch, 20 Clients)',
        fontsize=14, fontweight='bold', y=0.995
    )
    fig.tight_layout(h_pad=2.2)
    p2 = out_dir / "09_astra_3way_multipanel_diagnostics.png"
    fig.savefig(p2, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Salvo: {p2}")


if __name__ == "__main__":
    main()
