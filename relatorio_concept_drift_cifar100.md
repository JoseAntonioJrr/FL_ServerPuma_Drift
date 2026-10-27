# 📑 Relatório de Pesquisa: Federated Concept Drift no FL_ServerPuma (CIFAR-100)

> **Projeto:** FL_ServerPuma — Simulador de Aprendizado Federado Adaptativo  
> **Área:** Aprendizado Federado (FL) & Heterogeneidade Temporal (Concept Drift)  
> **Dataset:** CIFAR-100 (60.000 imagens, 100 classes)  
> **Modelo:** SimpleModel (CNN — 2×Conv2D + 2×FC, 100 classes de saída)  
> **Topologia:** Multi-Cliente (Drift Configurável por Cliente / Todos os Clientes)  
> **Data de Realização:** 24/09/2026  
> **Status:** Concluído e Integrado ao Ecossistema com Sucesso  

---

## 1. Visão Geral e Contexto da Pesquisa

O objetivo desta fase do trabalho foi expandir o arcabouço de **Federated Concept Drift** do simulador **FL_ServerPuma** do dataset CIFAR-10 (10 classes) para o dataset **CIFAR-100** (100 classes, 100 categorias visuais finas).

### 1.1 O Desafio Teórico no CIFAR-100
No Aprendizado Federado, a heterogeneidade temporal sob um espaço de rótulos estendido ($\mathcal{Y} = \{0, 1, \dots, 99\}$) intensifica drasticamente a complexidade da otimização federada:

$$P_{k, t}(X, Y) \neq P_{k, t+\Delta}(X, Y), \quad \text{onde } |\mathcal{Y}| = 100$$

Enquanto no CIFAR-10 a permutação afetava apenas 10 classes em 5 pares, no CIFAR-100 o **Real Concept Drift** causa uma reorganização semântica massiva do espaço de representação visual:
1. **Espaço de Decisão Mais Denso:** A última camada totalmente conectada da rede (FC2) possui 100 neurônios de saída (vs. 10 no CIFAR-10).
2. **Severidade do Drift:** Uma alteração nas relações $P(Y \mid X)$ sobre 100 classes desafia tanto a estabilidade dos modelos locais quanto o processo de agregação global (FedAvg / FedALA).

---

## 2. Decisões Arquiteturais e de Design

Para manter a reprodutibilidade estrita e a consistência metodológica com o CIFAR-10, foi adotada a arquitetura de **Partições Temporais Estáticas** para o CIFAR-100:

- **Gerador Especializado:** Criação do script `src/dataset/generate_Cifar100_drift.py`.
- **Particionamento Dirichlet (Non-IID):** Aplicação de amostragem probabilística Dirichlet com $\alpha = 0.1$ e `class_per_client = 10` (padrão de referência no PFLlib para CIFAR-100).
- **Estrutura Temporais em Duas Fases:**
  - **Phase 1 (Pré-Drift):** Particionamento original do CIFAR-100.
  - **Phase 2 (Pós-Drift):** Injeção de transformação controlada nos clientes selecionados.
- **Mecanismo de Permutação Semântica (`label_swap` em CIFAR-100):**
  Para abranger 100% do espaço de classes de forma determinística e simétrica, foram definidos **50 pares cruzados de troca**:
  $$i \longleftrightarrow (i + 50) \pmod{100} \quad \forall i \in [0, 49]$$
  *(Exemplo: Classe 0 $\leftrightarrow$ Classe 50, Classe 1 $\leftrightarrow$ Classe 51, ..., Classe 49 $\leftrightarrow$ Classe 99).*

---

## 3. Modificações e Adições no Código

### 3.1 `src/dataset/generate_Cifar100_drift.py` [NOVO]
- Carrega o CIFAR-100 original (50.000 treino / 10.000 teste).
- Particiona o dataset entre os $N$ clientes no formato Non-IID Dirichlet.
- Suporta 4 modalidades de Concept Drift:
  1. `label_swap`: Troca semântica simétrica de 50 pares ($i \leftrightarrow i+50$).
  2. `cyclic_shift`: Deslocamento circular de rótulos $y' = (y + \text{shift}) \pmod{100}$.
  3. `class_shift`: Re-amostragem Dirichlet independente com seed 42 (Dynamic Non-IID).
  4. `gaussian_noise`: Virtual Drift / Covariate Shift com ruído $\mathcal{N}(0, 0.25^2)$.
- Grava os arquivos temporais em disco:
  - `train/{idx}_phase1.npz` e `test/{idx}_phase1.npz`
  - `train/{idx}_phase2.npz` e `test/{idx}_phase2.npz`
  - `train/{idx}.npz` e `test/{idx}.npz` (retrocompatibilidade)
  - `config.json` contendo metadados completos de contagem e estatísticas por classe.

### 3.2 `src/system/server.py` [MODIFICADO]
- Atualizado o bloco de pattern matching na inicialização do servidor:
  ```python
  case 'Cifar100' | 'Cifar100_drift':
      self.global_model = SimpleModel(in_features=args.in_features, num_classes=args.num_classes, dim=args.dim)
      self.input_size = (1, 3, 32, 32)
  ```

### 3.3 `src/system/client.py` [MODIFICADO]
- Atualizado o pattern matching na inicialização do cliente:
  ```python
  case 'Cifar100' | 'Cifar100_drift':
      model = SimpleModel(in_features=args.in_features, num_classes=args.num_classes, dim=args.dim)
  ```

### 3.4 `src/system/prunning.py` [MODIFICADO]
- Atualizado o dimensionamento do pruning estrutural para `Cifar100_drift`:
  - `indices_not_remove_weight.repeat(25)` mantido para imagens $32 \times 32$.
  - Ajustada a dimensão da última camada totalmente conectada (`ind = 100`).

### 3.5 `src/system/run.sh` [MODIFICADO]
- Suporte nativo a `--dataset Cifar100_drift` e auto-detecção de `NUM_CLASSES=100`.
- **Nova Flag CLI `--balance`**:
  Permite alternar dinamicamente o balanceamento de amostras entre os clientes via terminal (`--balance` passa `"balance"` ao gerador; caso omisso, passa `"-"` para desbalanceado).

---

## 4. Comparativo Metodológico: CIFAR-10 Drift vs. CIFAR-100 Drift

| Propriedade | CIFAR-10 Drift (`Cifar10_drift`) | CIFAR-100 Drift (`Cifar100_drift`) |
|:---|:---:|:---:|
| **Número Total de Classes** | 10 | 100 |
| **Amostras por Classe** | 6.000 imagens | 600 imagens |
| **Classes por Cliente (Dirichlet)** | `class_per_client = 2` | `class_per_client = 10` |
| **Saída da Última Camada (FC2)** | 10 neurônios | 100 neurônios |
| **Mapeamento de Swap (`label_swap`)** | 5 pares manuais (ex: $0 \leftrightarrow 2$) | 50 pares simétricos ($i \leftrightarrow i+50$) |
| **Mapeamento Cíclico (`cyclic_shift`)** | $y' = (y + 1) \pmod{10}$ | $y' = (y + 1) \pmod{100}$ |
| **Suporte à Flag `--balance`** | ✅ Sim | ✅ Sim |

---

## 5. Como Executar Experimentos

### 5.1 Geração Manual do Dataset via Python
```bash
cd src/dataset
uv run generate_Cifar100_drift.py noniid - dir 5 --drift-type label_swap --drift-clients 0
```
*(Para gerar dados balanceados, substitua `-` por `balance`).*

### 5.2 Execução Automatizada via `run.sh`

```bash
# Experimento 1: Drift Abrupto no CIFAR-100 (5 clientes, 20 rodadas, drift no Round 10)
./run.sh --dataset Cifar100_drift --clients 5 --rounds 20 --drift-type abrupt --drift-round 10 --drift-clients 0

# Experimento 2: Drift Gradual no CIFAR-100 com Dados Balanceados
./run.sh --dataset Cifar100_drift --clients 5 --rounds 20 --drift-type gradual --drift-round 10 --drift-length 5 --drift-clients 0 --balance
```

---

## 6. Mapeamento de Arquivos do Projeto

| Arquivo / Diretório | Função / Descrição |
|---|---|
| `src/dataset/generate_Cifar100_drift.py` | Gerador temporal de CIFAR-100 com Concept Drift |
| `src/dataset/Cifar100_drift/` | Pasta contendo `config.json` e partições `.npz` de treino/teste |
| `src/system/server.py` | Servidor FL com suporte a `Cifar100_drift` |
| `src/system/client.py` | Cliente FL com suporte a `Cifar100_drift` |
| `src/system/prunning.py` | Lógica de pruning estrutural ajustada para 100 classes |
| `src/system/run.sh` | Orchestrador Tmux atualizado com suporte a `Cifar100_drift` e `--balance` |

---

## 7. Conclusão

A integração do **`Cifar100_drift`** foi finalizada com êxito. O sistema FL_ServerPuma agora suporta benchmarking empírico de Concept Drift tanto em problemas de classificação visual simples (CIFAR-10, 10 classes) quanto em cenários de alta granularidade (CIFAR-100, 100 classes), fornecendo um ambiente robusto e automatizado para pesquisas avançadas em Aprendizado Federado Temporal.
