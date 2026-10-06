#!/usr/bin/env python3
"""
plot_drift_results.py — Visualizador Especializado de Federated Concept Drift

Gera gráficos de publicação (PDF e PNG) focados em análise de Concept Drift:
1. Impacto na Acurácia Global do Servidor (Resiliência do Modelo Global).
2. Divergência Local: Cliente com Drift vs. Cliente Estável.
3. Métricas Quantitativas de Drift (Immediate Drop e Tempo de Recuperação).

Uso:
    uv run python plot_drift_results.py [--exp-name <nome_ou_pattern>] [--drift-round <int>]
"""

import os
import sys
import glob
import argparse
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})


def find_drift_experiment(results_base: Path, exp_query=None):
    if exp_query:
        target = results_base / exp_query
        if target.is_dir():
            return target
        candidates = list(results_base.glob(f"*{exp_query}*"))
        if candidates:
            return sorted(candidates, key=os.path.getmtime)[-1]
        print(f"❌ Nenhum experimento encontrado com a query: {exp_query}")
        return None

    # Procura experimentos com 'drift' no nome ordenados pelos mais recentes
    candidates = list(results_base.glob("*drift*"))
    if not candidates:
        # Fallback para qualquer experimento recente
        candidates = [d for d in results_base.iterdir() if d.is_dir()]

    if not candidates:
        print(f"❌ Nenhum experimento encontrado em: {results_base}")
        return None

    selected = sorted(candidates, key=os.path.getmtime)[-1]
    return selected


def load_experiment_data(exp_dir: Path):
    server_files = sorted(exp_dir.glob("server_*.h5"))
    client_files = sorted(exp_dir.glob("client_*.h5"))

    if not server_files:
        print(f"⚠️ Nenhum arquivo server_*.h5 em: {exp_dir}")
        return None

    # Dados do servidor
    with h5py.File(server_files[0], 'r') as hf:
        global_acc = np.array(hf['rs_test_acc'])
        global_loss = np.array(hf['rs_train_loss']) if 'rs_train_loss' in hf else None

    # Dados dos clientes
    clients_data = {}
    drift_round = None
    drift_type = None
    drift_clients_str = None

    for cf in client_files:
        with h5py.File(cf, 'r') as hf:
            c_idx = None
            for part in cf.stem.split('_'):
                if part.isdigit():
                    c_idx = int(part)
                    break
            if c_idx is None:
                c_idx = len(clients_data)

            local_acc = np.array(hf['rs_local_acc']) if 'rs_local_acc' in hf else None
            ala_acc = np.array(hf['rs_ala_acc']) if 'rs_ala_acc' in hf else None
            train_acc = np.array(hf['rs_train_acc']) if 'rs_train_acc' in hf else None

            # Metadados de drift nos atributos
            if 'drift_round' in hf.attrs:
                drift_round = int(hf.attrs['drift_round'])
            if 'drift_type' in hf.attrs:
                drift_type = str(hf.attrs['drift_type'])
            if 'drift_clients' in hf.attrs:
                drift_clients_str = str(hf.attrs['drift_clients'])

            clients_data[c_idx] = {
                'local_acc': local_acc,
                'ala_acc': ala_acc,
                'train_acc': train_acc
            }

    return {
        'exp_dir': exp_dir,
        'global_acc': global_acc,
        'global_loss': global_loss,
        'clients_data': clients_data,
        'drift_round': drift_round,
        'drift_type': drift_type,
        'drift_clients': drift_clients_str
    }


def to_percentage(arr):
    """Garante que os valores estejam na escala de porcentagem (0..100%)."""
    arr = np.array(arr, dtype=float)
    if len(arr) == 0:
        return arr
    if np.nanmax(arr) <= 1.01:
        return arr * 100.0
    return arr


def plot_drift_analysis(data, manual_drift_round=None, output_dir=None):
    exp_dir = data['exp_dir']
    global_acc = to_percentage(data['global_acc'])
    clients = data['clients_data']
    drift_round = manual_drift_round or data['drift_round'] or 10
    drift_type = data['drift_type'] or "Abrupt"

    # Resolver conjunto de clientes afetados a partir dos metadados HDF5
    drift_clients_str = data.get('drift_clients', None)
    if drift_clients_str is not None:
        dc = drift_clients_str.strip().lower()
        if dc == 'all':
            drift_client_set = set(clients.keys())
        else:
            drift_client_set = set(int(x.strip()) for x in dc.split(',') if x.strip().isdigit())
    else:
        # Fallback: assume que apenas o cliente 0 é afetado (retrocompatibilidade)
        drift_client_set = {0}

    num_rounds = len(global_acc)
    rounds = np.arange(1, num_rounds + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- SUBPLOT 1: Acurácia Global do Servidor ---
    ax1.plot(rounds, global_acc, marker='o', color='#1f77b4', lw=2.5, label='Global Test Acc')
    ax1.axvline(x=drift_round, color='#d62728', linestyle='--', lw=2.2, label=f'Drift ({drift_type}) @ R{drift_round}')
    ax1.set_xlabel('Rodada de Comunicação (Round)')
    ax1.set_ylabel('Acurácia Global (%)')
    ax1.set_title('Impacto do Drift no Modelo Global (Servidor)', fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right', frameon=True)
    ax1.set_ylim(0, 105)

    # --- SUBPLOT 2: Acurácia Local por Cliente ---
    colors = ['#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4']
    markers = ['s', '^', 'D', 'v', 'o', 'P', 'X', 'h', '*', 'd']

    for cid, c_info in sorted(clients.items()):
        acc = c_info['local_acc']
        if acc is not None and len(acc) > 0:
            acc_pct = to_percentage(acc)
            c_rounds = np.arange(1, len(acc_pct) + 1)
            status = '(Afetado)' if cid in drift_client_set else '(Estável)'
            label = f"Cliente {cid} {status}"
            c_color = colors[cid % len(colors)]
            c_marker = markers[cid % len(markers)]
            ax2.plot(c_rounds, acc_pct, marker=c_marker, color=c_color, lw=2.2, label=label)

    ax2.axvline(x=drift_round, color='#d62728', linestyle='--', lw=2.2, label=f'Início do Drift')
    ax2.set_xlabel('Rodada de Comunicação (Round)')
    ax2.set_ylabel('Acurácia Local (%)')
    ax2.set_title('Comportamento Local: Drift vs. Estabilidade', fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', frameon=True)
    ax2.set_ylim(0, 105)

    plt.suptitle(f"Análise de Federated Concept Drift — {exp_dir.name}", fontsize=16, fontweight='bold')
    plt.tight_layout()

    out_folder = Path(output_dir) if output_dir else exp_dir
    out_pdf = out_folder / "concept_drift_analysis.pdf"
    out_png = out_folder / "concept_drift_analysis.png"

    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"\n📊 Gráficos salvos com sucesso:")
    print(f"  📄 PDF: {out_pdf}")
    print(f"  🖼️  PNG: {out_png}\n")


def main():
    parser = argparse.ArgumentParser(description="Plot Results for Concept Drift")
    parser.add_argument("--exp-name", type=str, default=None, help="Nome do experimento ou pasta em results/")
    parser.add_argument("--drift-round", type=int, default=None, help="Rodada do drift caso não conste nos metadados")
    parser.add_argument("--results-dir", type=str, default=None, help="Diretório base de results")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    base_results = Path(args.results_dir) if args.results_dir else script_dir.parent.parent / "results"

    exp_dir = find_drift_experiment(base_results, args.exp_name)
    if not exp_dir:
        sys.exit(1)

    print(f"📂 Analisando experimento: {exp_dir}")
    data = load_experiment_data(exp_dir)
    if not data:
        sys.exit(1)

    plot_drift_analysis(data, manual_drift_round=args.drift_round)


if __name__ == "__main__":
    main()
