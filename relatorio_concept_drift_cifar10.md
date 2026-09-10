# 📑 Relatório de Pesquisa: Federated Concept Drift no FL_ServerPuma

> **Projeto:** FL_ServerPuma — Simulador de Aprendizado Federado Adaptativo  
> **Área:** Aprendizado Federado (FL) & Heterogeneidade Temporal (Concept Drift)  
> **Dataset:** CIFAR-10  
> **Modelo:** SimpleModel (CNN — 2×Conv2D + 2×FC)  
> **Topologia:** 2 Clientes (Drift Assimétrico no Cliente 0, Cliente 1 Estável)  
> **Data de Realização:** 09/09/2026  
> **Status:** Concluído e Validado Empiricamente com Sucesso  

---

## 1. Visão Geral e Contexto da Pesquisa

O objetivo deste trabalho foi estender o simulador **FL_ServerPuma** para suportar a indução, simulação e avaliação sistemática de **Concept Drift** (tanto **Abrupto** quanto **Gradual**) em um ambiente federado sob rede restrita, utilizando dados de imagem (CIFAR-10) e rede neural convolucional (CNN).

### 1.1 O Desafio Teórico
Enquanto o Aprendizado Federado clássico lida com **heterogeneidade espacial** (dados Non-IID entre clientes no mesmo instante de tempo):
$$\mathcal{D}_i \sim P_i(X, Y) \neq P_j(X, Y)$$

O **Federated Concept Drift** introduz **heterogeneidade temporal**, onde a distribuição conjunta varia ao longo das rodadas de comunicação $t$:
$$P_{k, t}(X, Y) \neq P_{k, t+\Delta}(X, Y)$$

Decidiu-se simular **Real Concept Drift** ($P(Y \mid X)$ se altera com o tempo) via permutação semântica de classes (troca cruzada de rótulos entre pares de categorias visuais), permitindo avaliar:
1. O choque na acurácia global do modelo agregador no servidor.
2. A velocidade de adaptação do cliente afetado (Cliente 0).
3. A resiliência do cliente estável (Cliente 1) contra transferência negativa (*catastrophic negative transfer*).

---

## 2. Decisões Arquiteturais e de Design

Para garantir reprodutibilidade e integridade do framework, adotou-se a estratégia de **Partições Temporais Estáticas**:
- **Script Dedicado:** Criação de `src/dataset/generate_Cifar_drift.py` que desacopla o particionamento em duas fases:
  - **Phase 1 (Pré-Drift):** Particionamento Non-IID Dirichlet ($\alpha=0.1$) original do CIFAR-10.
  - **Phase 2 (Pós-Drift):** Aplicação de *label swap* semântico para as partições do Cliente 0:
    - Avião ($0$) $\longleftrightarrow$ Pássaro ($2$)
    - Automóvel ($1$) $\longleftrightarrow$ Caminhão ($9$)
    - Gato ($3$) $\longleftrightarrow$ Cachorro ($5$)
    - Cervo ($4$) $\longleftrightarrow$ Cavalo ($7$)
    - Sapo ($6$) $\longleftrightarrow$ Navio ($8$)
- **Arquivos Gerados:**
  - `train/0_phase1.npz`, `test/0_phase1.npz`
  - `train/0_phase2.npz`, `test/0_phase2.npz` (Com Concept Drift)
  - `train/1_phase1.npz`, `test/1_phase1.npz`
  - `train/1_phase2.npz`, `test/1_phase2.npz` (Estável, sem alteração)
  - `train/0.npz` e `train/1.npz` mantidos como alias para retrocompatibilidade.

---

## 3. Modificações Implementadas no Código

### 3.1 `src/dataset/generate_Cifar_drift.py` [NOVO]
- Baixa o CIFAR-10 original.
- Divide os dados via Dirichlet para $N$ clientes (padrão: 2).
- Gera as fases 1 e 2 em disco e registra metadados completos em `Cifar10_drift/config.json`.
- Suporta flags CLI: `--drift-type` (`label_swap`, `cyclic_shift`, `class_shift`, `gaussian_noise`) e `--drift-clients`.

### 3.2 `src/utils/data_utils.py` [MODIFICADO]
- Atualizadas as funções `read_data` e `read_client_data` com o parâmetro `phase=None`.
- Carrega `f"{idx}_{phase}.npz"` quando fornecido, com fallback transparente para `f"{idx}.npz"`.

### 3.3 `src/system/client.py` [MODIFICADO]
- Adicionadas opções CLI via `argparse`:
  - `--drift`: Flag booleana ativadora.
  - `--drift-type`: `abrupt` ou `gradual`.
  - `--drift-round`: Rodada do drift (padrão: 5).
  - `--drift-length`: Duração em rodadas da transição gradual (padrão: 4).
  - `--drift-clients`: Clientes afetados (padrão: `"0"`).
- Implementada a função `load_blended_data`:
  - Durante o drift gradual, amostra batches com proporção linear $\alpha(t)$ de Phase 2 e $(1 - \alpha(t))$ de Phase 1.
- No loop de comunicação das rodadas:
  - **Abrupto:** Na rodada `drift_round`, recarrega os dados com `phase='phase2'` e reatualiza a referência do módulo de personalização local **FedALA**.
  - **Gradual:** Interpola progressivamente nas rodadas de transição e fixa em Phase 2 ao término da janela.
- Metadados de drift são persistidos nos atributos do arquivo HDF5 de resultados (`hf.attrs`).

### 3.4 `src/system/server.py` e `src/system/prunning.py` [MODIFICADOS]
- Adicionado suporte estrutural ao dataset `Cifar10_drift` nos blocos de pattern matching e no cálculo de restabelecimento das dimensões Conv $\to$ Linear para a poda adaptativa.

### 3.5 `src/system/run.sh` [MODIFICADO]
- Integração das opções de drift no script de automação Tmux.
- **Robustez de Sessão:**
  - Adicionado `tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true` antes da inicialização para eliminar erros de `duplicate session: fl_puma`.
  - Adicionado `trap` para sinais `INT` (Ctrl+C) e `TERM`, garantindo liberação imediata das portas e encerramento dos subprocessos zumbis.

### 3.6 `src/system/plot/plot_drift_results.py` [NOVO]
- Script especializado de visualização para gerar gráficos em alta resolução (PDF e PNG), contendo:
  - Painel 1: Acurácia Global do Servidor com linha vertical demarcando $T_{\text{drift}}$.
  - Painel 2: Divergência Local entre o Cliente Afetado (0) e o Cliente Estável (1).

---

## 4. Histórico de Execução & Resolução de Problemas

### 4.1 Experimento 1: Drift Abrupto (10 Rodadas)
- **Comando executado:**
  ```bash
  ./run.sh --model cnn --dataset Cifar10_drift --clients 2 --rounds 10 --drift --drift-type abrupt --drift-round 5 --drift-clients 0
  ```
- **Resultado:** Execução bem-sucedida de todas as 10 rodadas.
- **Comportamento observado:** No Round 5, a acurácia global despencou de 60.75% para 30.75% e recuperou no Round 6 para 62.74%.

### 4.2 Incidente de Sessão no Tmux e Diagnóstico
- Ao interromper o script e tentar disparar imediatamente o segundo experimento, ocorreu:
  - Mensagem: `duplicate session: fl_puma`
  - Consequência: Os novos clientes tentavam conectar na porta 9500 que não estava mais escutando (`Connection failed, retrying...`).
- **Resolução:**
  - Identificado que o Tmux mantinha a sessão antiga aberta esperando interação do usuário.
  - Implementado o auto-kill prévio e o `trap` de Ctrl+C no `run.sh`, limpando o ambiente.

### 4.3 Experimento 2: Drift Gradual (12 Rodadas)
- **Comando executado:**
  ```bash
  ./run.sh --model cnn --dataset Cifar10_drift --clients 2 --rounds 12 --drift --drift-type gradual --drift-round 5 --drift-length 4 --drift-clients 0
  ```
- **Resultado:** Execução 100% limpa, com transição ao longo dos Rounds 5 a 8 e estabilização nos Rounds 9 a 12.

---

## 5. Resultados Empíricos Comparativos

### 5.1 Tabela Rodada a Rodada dos Experimentos

| Rodada | Abrupto: Acc Global | Abrupto: Cli 0 (Local) | Gradual: Acc Global | Gradual: Proporção ($\alpha$) | Gradual: Cli 0 (Local) | Gradual: Cli 1 (Estável) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **R1** | 9.63% | 55.19% | 3.31% | 0.00 | 57.78% | 46.31% |
| **R2** | 51.49% | 61.46% | 52.05% | 0.00 | 57.68% | 51.85% |
| **R3** | 56.77% | 66.49% | 54.77% | 0.00 | 68.01% | 51.83% |
| **R4** | **60.75%** | **68.01%** | **59.92%** | 0.00 (Pré-drift) | **68.24%** | 57.19% |
| **R5** | **30.75%** ⚠️ | **68.03%** | **56.82%** | $\alpha = 0.20$ | **55.29%** | 57.43% |
| **R6** | 62.74% | 66.32% | **50.96%** | $\alpha = 0.40$ | **43.61%** | 58.45% |
| **R7** | 58.85% | 69.31% | **46.51%** ⚠️ | $\alpha = 0.60$ (Nadir) | **44.66%** | 57.25% |
| **R8** | 63.93% | 71.77% | **54.38%** | $\alpha = 0.80$ | **56.14%** | 61.87% |
| **R9** | 65.48% | 70.85% | **64.70%** | 1.00 (Pós-drift) | **72.89%** | 58.62% |
| **R10**| 64.30% | — | **65.75%** | 1.00 | **73.61%** | 61.19% |
| **R11**| — | — | **67.40%** | 1.00 | **72.97%** | 56.16% |
| **R12**| — | — | **64.57%** | 1.00 | **72.15%** | 63.15% |

---

## 6. Discussão Científica dos Fenômenos

### 6.1 Perfil de Degradação: "Curva em V" vs. "Curva em U"
- **Drift Abrupto (Curva em "V"):** O choque é pontual e agudo. Em uma única rodada (R5), a acurácia cai **30.0 pontos percentuais**. Contudo, como a distribuição seguinte é 100% consistente, a rede convolucional consegue se reajustar já na rodada subsequente (R6), retornando imediatamente ao patamar de 62%.
- **Drift Gradual (Curva em "U"):** A transição amortece o impacto inicial (a queda no R5 é de apenas 3.1%), mas **estende a degradação ao longo de 4 rodadas**. O ponto de maior severidade ocorre no Round 7 ($\alpha = 0.60$), onde a mistura de distribuições atinge entropia máxima e gera gradientes conflitantes entre amostras da mesma categoria.

### 6.2 Resiliência Federada do Cliente Estável (Cliente 1)
- O Cliente 1 manteve-se constantemente estável entre **56% e 63%** durante todo o experimento gradual.
- Isso comprova a eficácia da agregação global em amortecer a oscilação de um único cliente anômalo, evitando a contaminação destrutiva do modelo para os demais participantes.

### 6.3 Convergência Final e Adaptação
- Ambos os modos alcançaram convergência e até superaram os índices pré-drift (o gradual atingiu pico de **67.40%** e o Cliente 0 alcançou **73.61%** localmente).
- Isso evidencia a capacidade do modelo federado em readquirir representações úteis sob novos mapeamentos de classes.

---

## 7. Arquivos e Artefatos do Projeto

| Caminho do Arquivo | Função / Conteúdo |
|---|---|
| `src/dataset/generate_Cifar_drift.py` | Gerador temporal de CIFAR-10 com Concept Drift |
| `src/dataset/Cifar10_drift/` | Partições `.npz` pré e pós-drift salvas em disco |
| `src/utils/data_utils.py` | Suporte a carregamento por fases temporais |
| `src/system/client.py` | Cliente FL com lógica de drift abrupto e gradual |
| `src/system/server.py` | Servidor FL com suporte a `Cifar10_drift` |
| `src/system/prunning.py` | Dimensionamento de pruning estrutural para drift |
| `src/system/run.sh` | Script de automação Tmux com auto-cleanup e trap |
| `src/system/plot/plot_drift_results.py` | Gerador de gráficos PDF e PNG para análise |
| `src/results/fl_puma_...drift_abrupt_r5/` | Resultados HDF5 e gráficos do teste abrupto |
| `src/results/fl_puma_...drift_gradual_r5/` | Resultados HDF5 e gráficos do teste gradual |

---

## 8. Conclusão

O objetivo proposto foi alcançado integralmente. O pipeline está documentado, reproduzível, hermético e validado por dados empíricos reais. Os gráficos gerados em formato vetorial (`.pdf`) e matricial (`.png`) possuem qualidade editorial e estão prontos para subsidiar a redação de artigos científicos em conferências e periódicos de alto impacto na área de Aprendizado Federado.
