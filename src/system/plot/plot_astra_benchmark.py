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

def extract_active_global_acc(hf):
    """Extrai a acurácia global do servidor no conceito vigente de cada rodada.

    Quando o servidor gravou `rs_test_acc_orig` e `rs_test_acc_drift` sob drift
    (`--global-eval official|both`), `rs_test_acc` guarda apenas `G-orig` (Phase 1 fixa).
    Aqui reconstruímos exatamente `G-active(r) = (1 - alpha(r))*G-orig + alpha(r)*G-drift`
    usando o mesmo `phase_for_round` do servidor/cliente.
    Para arquivos legados, retorna `rs_test_acc` inalterado.
    """
    if 'rs_test_acc_orig' in hf and 'rs_test_acc_drift' in hf and bool(hf.attrs.get('drift_enabled', False)):
        g_orig = np.array(hf['rs_test_acc_orig'], dtype=float)
        g_drift = np.array(hf['rs_test_acc_drift'], dtype=float)
        n = min(len(g_orig), len(g_drift))
        args = types.SimpleNamespace(
            drift=True,
            drift_type=str(hf.attrs.get('drift_type', 'abrupt')),
            drift_round=int(hf.attrs.get('drift_round', 10)),
            drift_length=int(hf.attrs.get('drift_length', 5)),
            drift_freq=int(hf.attrs.get('drift_freq', 0)),
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
        return g_active, True
    if 'rs_test_acc' in hf:
        return np.array(hf['rs_test_acc'], dtype=float), False
    return None, False


def find_exp_h5_files(results_dir: Path, exp_name: str):
    """Localiza todos os arquivos server_*.h5 ou client_0_*.h5 do experimento."""
    target_dir = results_dir / exp_name
    if not target_dir.exists():
        print(f"[ERRO] Diretório {target_dir} não encontrado.")
        return {}

    runs_data = {}
    server_files = sorted(target_dir.glob("server_*.h5"))
    
    if server_files:
        for sf in server_files:
            # Extrai run_id
            name = sf.name
            try:
                run_id = int(name.split("_run")[1].split("_")[0].replace(".h5", ""))
            except Exception:
                run_id = len(runs_data) + 1
            
            with h5py.File(sf, 'r') as hf:
                acc, is_official_active = extract_active_global_acc(hf)
                if acc is not None:
                    runs_data[run_id] = acc
                    if is_official_active:
                        print(f"[INFO] Run #{run_id}: usando G-active (teste oficial 10k no conceito vigente)")
    else:
        # Fallback: tenta buscar dos clientes 0
        client_files = sorted(target_dir.glob("client_0_*.h5"))
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

def _parse_exp_metadata(path_or_name: str):
    name = str(path_or_name).lower()
    dataset = "CIFAR-100" if "cifar100" in name else "CIFAR-10"
    alpha = "0.1" if "a0.1" in name else ("0.5" if "a0.5" in name else "")
    alpha_str = f", α={alpha}" if alpha else ""
    epochs = "1" if ("_e1_" in name or name.endswith("_e1")) else ("5" if ("_e5_" in name or name.endswith("_e5")) else "5" if "a0.5" in name else "")
    epochs_str = f", {epochs} Local Epoch{'s' if epochs != '1' else ''}" if epochs else ""
    return dataset, alpha_str, epochs_str


def plot_single_run(run_id: int, acc_series: np.ndarray, output_path: Path):
    """Gera um gráfico individual de acurácia global para um determinado Run."""
    fig, ax = plt.subplots(figsize=(10, 6))
    rounds = np.arange(1, len(acc_series) + 1)
    drift_type_str = "Gradual" if "gradual" in str(output_path).lower() else "Abrupt"
    dataset, alpha_str, epochs_str = _parse_exp_metadata(output_path.parent.name)

    # Plot da curva de acurácia
    ax.plot(rounds, acc_series, color='#1f77b4', linewidth=2.5, label=f'Global Test Accuracy (Run {run_id})')

    # Marcadores das 10 ocorrências de Concept Drift (a cada 5 rounds)
    drift_rounds = [r for r in range(5, len(acc_series) + 1, 5)]
    for i, dr in enumerate(drift_rounds):
        ax.axvline(x=dr, color='#d62728', linestyle='--', alpha=0.6, linewidth=1.2,
                   label=f'Concept Drift ({drift_type_str})' if i == 0 else "")

    ax.set_title(f'ASTRA Benchmark Scenario ({drift_type_str}) - Simulation Run #{run_id}\n(LoRA-CLIP on {dataset} Drift{alpha_str}, 20 Clients{epochs_str})', fontweight='bold')
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
    dataset, alpha_str, epochs_str = _parse_exp_metadata(output_path.parent.name)

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

    trial_str = f"{len(all_runs)} Independent Trial" if len(all_runs) == 1 else f"{len(all_runs)} Independent Trials"
    ax.set_title(f'Aggregated Benchmark Performance ({trial_str})\nLoRA-CLIP under Periodic {drift_type_str} Concept Drift ({dataset}{alpha_str}{epochs_str}, 10 Shifts total)', fontweight='bold')
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
