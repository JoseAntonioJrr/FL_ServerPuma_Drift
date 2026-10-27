# Relatório Geral das Atividades — Dia 01: Benchmark de Concept Drift & Telemetria no FL_ServerPuma

**Data:** 01 de Outubro de 2026  
**Repositório:** `/home/jose/FL_ServerPuma`  
**Papel:** Pesquisador Engenheiro da Computação em Aprendizado Federado e Otimização de Modelos (SLMs/LLMs/CNNs)  

---

## 📌 Executive Summary (O Ponto em que Estamos)

Hoje estabelecemos e validamos a infraestrutura experimental completa para análise de **Concept Drift Gradual vs. Abrupto** no framework **FL_ServerPuma**. 

Regeneramos os datasets **CIFAR-10 Drift** e **CIFAR-100 Drift** para **10 clientes** sob distribuição Dirichlet Non-IID ($\alpha=0.1$), instrumentamos o código dos clientes para telemetria de memória RAM (`psutil`), executamos os experimentos de **50 rodadas de comunicação** induzindo o drift no **Round 20**, e construímos scripts automatizados para a geração de relatórios e gráficos em qualidade de publicação (PDF/PNG).

---

## 📑 1. Análise Contextual e Decisões de Modelagem de Dados

### 1.1 Verificação do Balanceamento Non-IID (Dirichlet $\alpha=0.1$)
* **Investigação do Código:** Analisamos a função `separate_data` em `dataset_utils.py` e constatamos que, quando a partição é do tipo Dirichlet (`partition="dir"`), a variável `balance` é intencionalmente ignorada pela literatura, mantendo o desbalanceamento realista de volume e classe entre clientes.
* **Decisão:** Mantivemos o padrão Dirichlet ($\alpha=0.1$) por ser o *gold standard* aceito em conferências internacionais (ex: PFLlib, FedDrift).
* **Regeneração de Dados:** Regeneramos os arquivos temporais (`phase1` e `phase2`) para 10 clientes em ambos os datasets.

---

## 🛠️ 2. Instrumentação e Telemetria de Memória

### 2.1 Alterações no Código do Cliente (`src/system/client.py`)
* Instrumentamos o loop de treinamento dos clientes com a biblioteca `psutil` para medir o consumo de **RAM Resident Set Size (RSS em MB)** no Host a cada rodada (`INÍCIO_ROUND`, `PÓS_DRIFT` e `FIM_ROUND_PÓS_GC`).
* Atualizamos a persistência HDF5 em `client.py` (`save_results`) para registrar o novo campo `rs_ram_rss_mb`.

### 2.2 Distinção Técnica: RAM (CPU Host) vs VRAM (GPU CUDA)
* **RAM do Host (CPU):** Medida pelo `psutil`. É onde residem os processos Python, os DataLoaders do PyTorch e as amostras dos datasets `phase1` e `phase2`. Sofre variação significativa durante o **Drift Gradual (+207 MB)** por necessitar de cache dual simultâneo para a interpolação senoidal.
* **VRAM da GPU:** Medida via PyTorch CUDA. Permaneceu constante ao longo de todas as rodadas, pois o tamanho da CNN e o minibatch ($B=32$) são fixos durante o cálculo do gradiente.

---

## 🧪 3. Experimentos Executados Hoje (50 Rodadas, 10 Clientes, CNN)

### 3.1 Experimento 1: Concept Drift Gradual (`drift_type='gradual'`)
* **Janela de Drift:** Rounds 20 a 29 (10 rodadas de mistura probabilística).
* **Comportamento da Acurácia Global:** Degradação suave da acurácia do servidor com ponto mínimo (*nadir*) no meio da transição (Round 28), seguido por recuperação gradual.
* **Comportamento da RAM:** Apresenta um **pico de uso de +207 MB por cliente** durante os Rounds 20 a 29, retornando ao baseline (~1909 MB) no Round 30 quando o cache da Fase 1 é liberado pelo `gc.collect()`.

### 3.2 Experimento 2: Concept Drift Abrupto (`drift_type='abrupt'`)
* **Ponto de Drift:** Salto direto no Round 20.
* **Comportamento da Acurácia Global:** Salto instantâneo de conceito no Round 20 com recuperação direta a partir do Round 21.
* **Comportamento da RAM:** **Sem pico de RAM** (mantém-se constante em ~1910 MB), pois o dataset antigo é destruído imediatamente no Round 20.

---

## 📊 4. Tabela de Resultados Consolidados (Acurácia Global do Servidor)

| Dataset / Cenário | Acurácia R1 | Pre-Drift (R19) | Ponto Crítico / Transição | Final (R50) | Consumo de RAM (RSS) |
|---|---|---|---|---|---|
| **CIFAR-10 Gradual** | 9.77% | 13.48% | 13.15% (Round 28) | **21.47%** | Pico de 2117 MB (+207 MB) |
| **CIFAR-10 Abrupto** | 9.77% | 13.48% | 13.52% (Round 20) | **21.47%** | Constante em ~1910 MB |
| **CIFAR-100 Gradual** | 1.70% | 4.72% | 2.10% (Round 28) | **8.50%** | Pico de 2117 MB (+176 MB) |
| **CIFAR-100 Abrupto** | 1.70% | 4.72% | 4.70% (Round 20) | **8.50%** | Constante em ~1941 MB |

*Nota: A acurácia global do servidor retrata o modelo único agregado sobre o teste diversificado com todas as classes, diferindo da acurácia local dos clientes que atinge ~80% nas suas poucas classes específicas.*

---

## 🎨 5. Entregáveis de Visualização Criados

Desenvolvemos scripts de automação em Python (`matplotlib`) localizados em `src/system/plot/`:

1. **`01_acuracia_global_drift.pdf / .png`:** Estilo padronizado do repositório, marcando o início (linha tracejada vermelha) e a janela de drift gradual (área sombreada). Legenda no canto superior direito.
2. **`02_consumo_ram_barras.pdf / .png`:** Gráfico de barras por rodada destacando o pico de RAM no drift gradual.
3. **`03_acuracia_abrupto.pdf / .png`:** Acurácia Global para a transição abrupta no Round 20.
4. **`04_consumo_ram_abrupto_barras.pdf / .png`:** Gráfico de barras de RAM para o drift abrupto.
5. **`05_comparativo_abrupto_vs_gradual.pdf / .png`:** Painel duplo comparando Gradual vs Abrupto no CIFAR-10 e CIFAR-100. **Legendas posicionadas no canto superior esquerdo (`upper left`) para total clareza visual**.

---

## 🎯 6. Ponto de Situação Atual & Próximos Passos

* **Estado Atual:** Todos os experimentos de 10 clientes e 50 rounds para Drift Gradual e Abrupto estão **concluídos, salvos e documentados**.
* **Arquivos do Servidor/Resultados:** Armazenados em `src/results/fl_puma_*`.
* **Próximas Possibilidades de Expansão:**
  1. Testar otimização de Small Language Models (SLMs) ou LoRA/SoRA com drift.
  2. Explorar técnicas de mitigação de concept drift (ex: FedDrift, adaptação dinâmica de taxa de aprendizado ou pesos de agregação).

---
*Relatório salvo em `relatorio_dia_01.md` como checkpoint oficial do projeto.*
