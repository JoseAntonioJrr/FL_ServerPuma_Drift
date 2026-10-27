#!/usr/bin/env python3
"""
plot_astra_benchmark.py — Gerador de Visualizações do Benchmark ASTRA (SBRC 2026) vs. FL_ServerPuma

Lê os arquivos .h5 gerados pelas 5 simulações sob Concept Drift Periódico (10 drifts a cada 5 rounds)
e produz 6 gráficos em PNG de alta resolução (300 DPI):
  - 5 Gráficos Individuais: Trajetória de Acurácia Global de cada simulação (Run 1 a Run 5)
  - 1 Gráfico Agregado: Média Geral das 5 simulações com banda de Desvio Padrão (Mean ± Std Dev)
"""

import os
import sys
import glob
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configuração de estilo visual acadêmico e moderno
plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

def find_exp_h5_files(results_dir: Path, exp_name: str):
    """Localiza todos os arquivos server_*.h5 ou client_0_*.h5 do experimento."""
    target_dir = results_dir / exp_name
    if not target_dir.exists():
        print(f"[ERRO] Diretório {target_dir} não encontrado.")
        return {}

    runs_data = {}
    server_files = list(target_dir.glob("server_*.h5"))
    
    if server_files:
        for sf in server_files:
            # Extrai run_id
            name = sf.name
            try:
                run_id = int(name.split("_run")[1].split("_")[0].replace(".h5", ""))
            except Exception:
                run_id = len(runs_data) + 1
            
            with h5py.File(sf, 'r') as hf:
                if 'rs_test_acc' in hf:
                    acc = np.array(hf['rs_test_acc'])
                    runs_data[run_id] = acc
    else:
        # Fallback: tenta buscar dos clientes 0
        client_files = list(target_dir.glob("client_0_*.h5"))
        for cf in client_files:
            name = cf.name
            try:
                run_id = int(name.split("_run")[1].split("_")[0].replace(".h5", ""))
            except Exception:
                run_id = len(runs_data) + 1
            
            with h5py.File(cf, 'r') as hf:
                if 'rs_global_acc' in hf:
                    acc = np.array(hf['rs_global_acc'])
                    runs_data[run_id] = acc

    return runs_data

def plot_single_run(run_id: int, acc_series: np.ndarray, output_path: Path):
    """Gera um gráfico individual de acurácia global para um determinado Run."""
    fig, ax = plt.subplots(figsize=(10, 6))
    rounds = np.arange(1, len(acc_series) + 1)
    drift_type_str = "Gradual" if "gradual" in str(output_path).lower() else "Abrupt"

    # Plot da curva de acurácia
    ax.plot(rounds, acc_series, color='#1f77b4', linewidth=2.5, label=f'Global Test Accuracy (Run {run_id})')

    # Marcadores das 10 ocorrências de Concept Drift (a cada 5 rounds)
    drift_rounds = [r for r in range(5, len(acc_series) + 1, 5)]
    for i, dr in enumerate(drift_rounds):
        ax.axvline(x=dr, color='#d62728', linestyle='--', alpha=0.6, linewidth=1.2,
                   label=f'Concept Drift ({drift_type_str})' if i == 0 else "")

    ax.set_title(f'ASTRA Benchmark Scenario ({drift_type_str}) - Simulation Run #{run_id}\n(LoRA-CLIP on CIFAR-10 Drift, 20 Clients, 5 Local Epochs)', fontweight='bold')
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Global Test Accuracy (%)')
    ax.set_xlim(1, len(acc_series))
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(True)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Gráfico do Run #{run_id} salvo em: {output_path}")

def plot_aggregated_mean(runs_dict: dict, output_path: Path):
    """Gera o 6º gráfico mostrando a Média Geral e Desvio Padrão das simulações."""
    fig, ax = plt.subplots(figsize=(10, 6))
    drift_type_str = "Gradual" if "gradual" in str(output_path).lower() else "Abrupt"

    # Converte dicionário para matriz 2D (num_runs x num_rounds)
    max_rounds = max(len(v) for v in runs_dict.values())
    all_runs = []
    for rid in sorted(runs_dict.keys()):
        accs = runs_dict[rid]
        if len(accs) == max_rounds:
            all_runs.append(accs)

    runs_matrix = np.array(all_runs)  # (N, 50)
    mean_acc = np.mean(runs_matrix, axis=0)
    std_acc = np.std(runs_matrix, axis=0)
    rounds = np.arange(1, max_rounds + 1)

    # Curvas de cada run em cinza claro para contexto
    for i, rid in enumerate(sorted(runs_dict.keys())):
        ax.plot(rounds, runs_dict[rid], color='#aec7e8', alpha=0.5, linewidth=1.0,
                label='Individual Simulation Run' if i == 0 else "")

    # Banda de desvio padrão (Mean ± Std Dev)
    ax.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc, color='#1f77b4', alpha=0.25, label='± 1 Std Dev')

    # Curva média principal
    ax.plot(rounds, mean_acc, color='#1f77b4', linewidth=2.8, label=f'Mean Accuracy (N={len(all_runs)} Runs)')

    # Marcadores dos 10 Concept Drifts
    drift_rounds = [r for r in range(5, max_rounds + 1, 5)]
    for i, dr in enumerate(drift_rounds):
        ax.axvline(x=dr, color='#d62728', linestyle='--', alpha=0.6, linewidth=1.2,
                   label='Concept Drift Event (Every 5 Rounds)' if i == 0 else "")

    ax.set_title(f'Aggregated Benchmark Performance ({len(all_runs)} Independent Trials)\nLoRA-CLIP under Periodic {drift_type_str} Concept Drift (10 Shifts total)', fontweight='bold')
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Global Test Accuracy (%)')
    ax.set_xlim(1, max_rounds)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(True)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Gráfico Agregado (Média e Desvio Padrão) salvo em: {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot ASTRA Benchmark Results")
    parser.add_argument("--exp-name", type=str, default="astra_periodic_drift", help="Nome da pasta do experimento em src/results/")
    args = parser.parse_args()

    current_dir = Path(__file__).resolve().parent
    results_dir = current_dir.parent.parent / "results"
    output_dir = results_dir / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_data = find_exp_h5_files(results_dir, args.exp_name)
    if not runs_data:
        print(f"[AVISO] Nenhum arquivo .h5 encontrado em {results_dir / args.exp_name}. Verifique a execução das simulações.")
        sys.exit(1)

    print(f"\n--- Processando {len(runs_data)} simulações para o benchmark ASTRA ---")

    drift_tag = "gradual" if "gradual" in args.exp_name else "abrupt"

    # 1 a 5: Plot de cada run individual
    for rid in sorted(runs_data.keys()):
        out_png = output_dir / f"0{rid}_astra_{drift_tag}_run{rid}.png"
        plot_single_run(rid, runs_data[rid], out_png)

    # 6: Plot da média geral
    out_mean_png = output_dir / f"06_astra_{drift_tag}_mean_general.png"
    plot_aggregated_mean(runs_data, out_mean_png)

    print("\nTodos os 6 gráficos foram gerados com sucesso!")

if __name__ == "__main__":
    main()
