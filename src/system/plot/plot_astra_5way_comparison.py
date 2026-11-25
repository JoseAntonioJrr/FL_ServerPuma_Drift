#!/usr/bin/env python3
"""
ASTRA 5-Way Comparative Benchmark Plot:
1. Stationary Baseline (FedAvg No Drift) [Green]
2. FedAvg Periodic Abrupt Drift [Red]
3. FedAvg Periodic Gradual Drift [Blue]
4. ASTRA-FL Periodic Abrupt Drift (Full ASTRA Method) [Yellow/Gold]
5. ASTRA-FL Periodic Gradual Drift (Full ASTRA Method) [Purple]

Evaluated on CIFAR-100 (Dirichlet alpha=0.1, 1 Local Epoch, 20 Clients, Zero Data Leakage)
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
    'legend.fontsize': 9.5,
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
    parser = argparse.ArgumentParser(description="ASTRA 5-Way Comparative Benchmark Plot")
    parser.add_argument("--nodrift-exp", type=str, default="clip_astra_cifar100_a0.1_e1_nodrift")
    parser.add_argument("--abrupt-fedavg-exp", type=str, default="clip_astra_cifar100_a0.1_e1_abrupt_drift")
    parser.add_argument("--gradual-fedavg-exp", type=str, default="clip_astra_cifar100_a0.1_e1_gradual_drift")
    parser.add_argument("--abrupt-astra-exp", type=str, default="clip_astrafl_cifar100_a0.1_e1_abrupt_drift")
    parser.add_argument("--gradual-astra-exp", type=str, default="clip_astrafl_cifar100_a0.1_e1_gradual_drift")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS / args.nodrift_exp
    out_dir.mkdir(parents=True, exist_ok=True)

    nodrift = load_exp_data(args.nodrift_exp)
    abrupt_fedavg = load_exp_data(args.abrupt_fedavg_exp)
    gradual_fedavg = load_exp_data(args.gradual_fedavg_exp)
    abrupt_astra = load_exp_data(args.abrupt_astra_exp)
    gradual_astra = load_exp_data(args.gradual_astra_exp)

    n = min(
        len(nodrift['active_acc']),
        len(abrupt_fedavg['active_acc']),
        len(gradual_fedavg['active_acc']),
        len(abrupt_astra['active_acc']),
        len(gradual_astra['active_acc']),
    )
    rounds = np.arange(1, n + 1)

    s_acc = nodrift['active_acc'][:n]
    fa_acc = abrupt_fedavg['active_acc'][:n]
    fg_acc = gradual_fedavg['active_acc'][:n]
    aa_acc = abrupt_astra['active_acc'][:n]
    ag_acc = gradual_astra['active_acc'][:n]

    alpha = alpha_schedule(n)
    ceiling = 100.0 * np.maximum(alpha, 1.0 - alpha)
    ph = abrupt_phase(n)

    # Color palette as requested:
    # 1. Stationary Baseline: Green (#2ca02c)
    # 2. FedAvg Abrupt: Red (#d62728)
    # 3. FedAvg Gradual: Blue (#1f77b4)
    # 4. ASTRA-FL Abrupt: Yellow/Gold (#d4ac0d / #df9700)
    # 5. ASTRA-FL Gradual: Purple (#7b1fa2 / #9467bd)
    color_s = '#2ca02c'
    color_fa = '#d62728'
    color_fg = '#1f77b4'
    color_aa = '#d4ac0d'   # Yellow/Gold
    color_ag = '#7b1fa2'   # Purple

    # ==========================================
    # FIGURE 1: Primary 5-Way Benchmark Trajectory
    # ==========================================
    fig, ax = plt.subplots(figsize=(13, 7))

    # Highlight Phase 2 regions (Abrupt)
    for t in range(n):
        if ph[t] == 1:
            ax.axvspan(t + 0.5, t + 1.5, color='#ff7f0e', alpha=0.07, lw=0)

    # Drift transition boundary lines
    for dr in range(L, n + 1, L):
        ax.axvline(dr, color='gray', linestyle=':', linewidth=1.1, alpha=0.65)

    # 1. Stationary Baseline (No Drift)
    ax.plot(rounds, s_acc, color=color_s, linewidth=2.8, marker='^', markersize=5.0,
            label=f'Stationary Baseline (FedAvg No Drift) — Peak: {s_acc.max():.1f}%, Mean: {s_acc.mean():.1f}%')

    # 2. FedAvg Abrupt Drift (Red)
    ax.plot(rounds, fa_acc, color=color_fa, linewidth=2.0, linestyle='--', marker='s', markersize=4.0, alpha=0.85,
            label=f'FedAvg Abrupt Drift — Peak: {fa_acc.max():.1f}%, Mean: {fa_acc.mean():.1f}%')

    # 3. ASTRA-FL Abrupt Drift (Yellow/Gold)
    ax.plot(rounds, aa_acc, color=color_aa, linewidth=2.5, marker='D', markersize=4.5,
            markeredgecolor='#8d6e10', markeredgewidth=0.8,
            label=f'ASTRA-FL Abrupt Drift (Full Method) — Peak: {aa_acc.max():.1f}%, Mean: {aa_acc.mean():.1f}%')

    # 4. FedAvg Gradual Drift (Blue)
    ax.plot(rounds, fg_acc, color=color_fg, linewidth=2.0, linestyle='--', marker='o', markersize=4.0, alpha=0.85,
            label=f'FedAvg Gradual Drift — Peak: {fg_acc.max():.1f}%, Mean: {fg_acc.mean():.1f}%')

    # 5. ASTRA-FL Gradual Drift (Purple)
    ax.plot(rounds, ag_acc, color=color_ag, linewidth=2.5, marker='P', markersize=5.0,
            label=f'ASTRA-FL Gradual Drift (Full Method) — Peak: {ag_acc.max():.1f}%, Mean: {ag_acc.mean():.1f}%')

    # Gradual Bayes Ceiling
    ax.plot(rounds, ceiling, color='#7f7f7f', linestyle=':', linewidth=1.6, alpha=0.8,
            label=r'Gradual Bayes Ceiling $\max(\alpha, 1-\alpha)$')

    # Indicator for Phase 2 active window
    ax.plot([], [], color='#ff7f0e', alpha=0.25, linewidth=8, label='Phase 2 active window (Abrupt)')

    ax.set_title(
        'LoRA-CLIP on CIFAR-100 (ASTRA Dirichlet α=0.1, 1 Local Epoch, 20 Clients, Zero Data Leakage)\n'
        'Complete Benchmark: Stationary vs. FedAvg Baselines vs. Full ASTRA-FL Method (Abrupt & Gradual Drift, L=5)',
        fontweight='bold', fontsize=13
    )
    ax.set_xlabel('Communication Round', fontweight='bold')
    ax.set_ylabel('Global Test Accuracy (%) — Active Concept', fontweight='bold')
    ax.set_xlim(1, n)
    ax.set_ylim(0, 105)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(True)
    ax.legend(loc='lower left', framealpha=0.95, edgecolor='#cccccc', ncol=2)

    fig.tight_layout()
    p1 = out_dir / "08_astra_5way_comparison.png"
    fig.savefig(p1, bbox_inches='tight')
    # Also save as 08_astra_comparison_all.png for clear referencing
    p1_alt = out_dir / "08_astra_comparison_all.png"
    fig.savefig(p1_alt, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Salvo: {p1}")

    # ==========================================
    # FIGURE 2: Multi-panel Diagnostics (Accuracy, Loss, ASTRA-FL Gain)
    # ==========================================
    fig, axes = plt.subplots(3, 1, figsize=(13, 13.5), sharex=True)

    # Panel A: Accuracy Curves
    ax_a = axes[0]
    for t in range(n):
        if ph[t] == 1:
            ax_a.axvspan(t + 0.5, t + 1.5, color='#ff7f0e', alpha=0.07, lw=0)
    for dr in range(L, n + 1, L):
        ax_a.axvline(dr, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)

    ax_a.plot(rounds, s_acc, color=color_s, lw=2.5, marker='^', ms=4.0, label=f'Stationary Baseline ({s_acc.mean():.1f}%)')
    ax_a.plot(rounds, fa_acc, color=color_fa, lw=1.8, ls='--', marker='s', ms=3.5, label=f'FedAvg Abrupt ({fa_acc.mean():.1f}%)')
    ax_a.plot(rounds, aa_acc, color=color_aa, lw=2.2, marker='D', ms=3.8, markeredgecolor='#8d6e10', label=f'ASTRA-FL Abrupt ({aa_acc.mean():.1f}%)')
    ax_a.plot(rounds, fg_acc, color=color_fg, lw=1.8, ls='--', marker='o', ms=3.5, label=f'FedAvg Gradual ({fg_acc.mean():.1f}%)')
    ax_a.plot(rounds, ag_acc, color=color_ag, lw=2.2, marker='P', ms=4.0, label=f'ASTRA-FL Gradual ({ag_acc.mean():.1f}%)')
    ax_a.plot(rounds, ceiling, color='#7f7f7f', ls=':', lw=1.3, alpha=0.75, label=r'Bayes Ceiling')
    ax_a.set_ylabel('Accuracy (%)', fontweight='bold')
    ax_a.set_ylim(0, 105)
    ax_a.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax_a.grid(True)
    ax_a.legend(loc='lower left', ncol=3, framealpha=0.92, fontsize=9.0)
    ax_a.set_title('A) Global Test Accuracy under Active Concept (G-active)', fontweight='bold')

    # Panel B: Test Loss
    ax_b = axes[1]
    for dr in range(L, n + 1, L):
        ax_b.axvline(dr, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)

    if nodrift['loss'] is not None:
        ax_b.plot(rounds, nodrift['loss'][:n], color=color_s, lw=2.2, marker='^', ms=3.5,
                  label=f'Stationary (Mean Loss: {nodrift["loss"][:n].mean():.2f})')
    if abrupt_fedavg['loss'] is not None:
        ax_b.plot(rounds, abrupt_fedavg['loss'][:n], color=color_fa, lw=1.8, ls='--', marker='s', ms=3.5,
                  label=f'FedAvg Abrupt (Mean Loss: {abrupt_fedavg["loss"][:n].mean():.2f})')
    if abrupt_astra['loss'] is not None:
        ax_b.plot(rounds, abrupt_astra['loss'][:n], color=color_aa, lw=2.2, marker='D', ms=3.5,
                  label=f'ASTRA-FL Abrupt (Mean Loss: {abrupt_astra["loss"][:n].mean():.2f})')
    if gradual_fedavg['loss'] is not None:
        ax_b.plot(rounds, gradual_fedavg['loss'][:n], color=color_fg, lw=1.8, ls='--', marker='o', ms=3.5,
                  label=f'FedAvg Gradual (Mean Loss: {gradual_fedavg["loss"][:n].mean():.2f})')
    if gradual_astra['loss'] is not None:
        ax_b.plot(rounds, gradual_astra['loss'][:n], color=color_ag, lw=2.2, marker='P', ms=3.5,
                  label=f'ASTRA-FL Gradual (Mean Loss: {gradual_astra["loss"][:n].mean():.2f})')

    ax_b.set_ylabel('Test Loss', fontweight='bold')
    ax_b.grid(True)
    ax_b.legend(loc='upper right', ncol=2, framealpha=0.92, fontsize=9.0)
    ax_b.set_title('B) Cross-Entropy Test Loss Evolution', fontweight='bold')

    # Panel C: ASTRA-FL Advantage (Delta Acc = ASTRA-FL - FedAvg)
    ax_c = axes[2]
    for dr in range(L, n + 1, L):
        ax_c.axvline(dr, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)

    delta_abrupt = aa_acc - fa_acc
    delta_gradual = ag_acc - fg_acc

    ax_c.axhline(0, color='black', linestyle='-', linewidth=1.0, alpha=0.7)
    ax_c.plot(rounds, delta_abrupt, color=color_aa, lw=2.2, marker='D', ms=4.0, markeredgecolor='#8d6e10',
              label=f'Abrupt Drift: ASTRA-FL gain over FedAvg (Mean: +{delta_abrupt.mean():.2f} p.p., Max: +{delta_abrupt.max():.2f} p.p.)')
    ax_c.plot(rounds, delta_gradual, color=color_ag, lw=2.2, marker='P', ms=4.5,
              label=f'Gradual Drift: ASTRA-FL gain over FedAvg (Mean: +{delta_gradual.mean():.2f} p.p., Max: +{delta_gradual.max():.2f} p.p.)')

    ax_c.set_xlabel('Communication Round', fontweight='bold')
    ax_c.set_ylabel('Accuracy Gain (p.p.)', fontweight='bold')
    ax_c.set_xlim(1, n)
    ax_c.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax_c.grid(True)
    ax_c.legend(loc='upper left', framealpha=0.92, fontsize=9.5)
    ax_c.set_title('C) Net Adaptation Gain of Full ASTRA-FL Method over Baseline FedAvg', fontweight='bold')

    fig.suptitle(
        '5-Way Comprehensive Diagnostics: Stationary vs. FedAvg Baselines vs. Full ASTRA-FL\n(LoRA-CLIP on CIFAR-100, Dirichlet α=0.1, 1 Local Epoch, 20 Clients)',
        fontsize=14, fontweight='bold', y=0.995
    )
    fig.tight_layout(h_pad=2.2)
    p2 = out_dir / "09_astra_5way_multipanel_diagnostics.png"
    fig.savefig(p2, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Salvo: {p2}")


if __name__ == "__main__":
    main()
