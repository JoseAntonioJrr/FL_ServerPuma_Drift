# Registro Completo das Discussões e Decisões Técnicas — 12 de Outubro de 2026

**Projeto:** FL_ServerPuma — Federated Learning com Vision-Language Models (LoRA-CLIP)  
**Tópico Principal:** Validação do Benchmark ASTRA (SBRC 2026), Concept Drift Periódico (Abrupto vs. Gradual), Resolução do Gargalo de Amostragem de Clientes e Execução das Novas Baterias  
**Data:** 12 de Outubro de 2026  
**Documento de Saída:** `conversa_dia12.md`  

---

## 📑 Sumário Executivo da Sessão

1. [Contextualização e Leitura de Artefatos](#1-contextualização-e-leitura-de-artefatos)
2. [Análise Científica do Concept Drift Gradual (Nadir e Histerese)](#2-análise-científica-do-concept-drift-gradual)
3. [Geração dos Gráficos e Artefatos do Benchmark ASTRA](#3-geração-dos-gráficos-e-artefatos-do-benchmark-astra)
4. [Elaboração do Relatório Executivo `dia_das_criancas.md`](#4-elaboração-do-relatório-executivo-dia_das_criancasmd)
5. [Descoberta e Diagnóstico: O Gargalo da Seleção Estática de Clientes](#5-descoberta-e-diagnóstico-o-gargalo-da-seleção-estática-de-clientes)
6. [Implementação da Amostragem Dinâmica Determinística Round a Round](#6-implementação-da-amostragem-dinâmica-determinística-round-a-round)
7. [Esclarecimento Teórico: `--drift-clients` vs. `--clients`](#7-esclarecimento-teórico---drift-clients-vs---clients)
8. [Estrutura dos Dados: Particionamento Non-IID Dirichlet e Phase 1 vs. Phase 2](#8-estrutura-dos-dados-particionamento-non-iid-e-phase-1-vs-phase-2)
9. [Automação e Disparo das Baterias Experimentais Completas](#9-automação-e-disparo-das-baterias-experimentais-completas)
10. [Evidências de Execução em Tempo Real (GPU e Logs)](#10-evidências-de-execução-em-tempo-real-gpu-e-logs)

---

## 1. Contextualização e Leitura de Artefatos

O usuário solicitou uma atuação enquanto **pesquisador especialista em Engenharia da Computação, Aprendizado Federado (FL), Concept Drift e Modelos de Linguagem e Visão (LoRA-CLIP)** para realizar uma revisão minuciosa de toda a base de documentação do repositório:

* `analysis_fl_serverpuma.md`
* `atualizacoes_main_git.md`
* `implementation_plan.md`
* `oqfoifeito.md`
* `README.md`
* `relatorio_clip_drift.md`
* `relatorio_concept_drift_cifar10.md`
* `relatorio_concept_drift_cifar100.md`
* `relatorio_dia_01.md`
* `relatorio.md`
* `walkthrough.md`

### Metodologia do Estudo:
* **Backbone:** CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) congelado.
* **Mecanismo de Adaptação Eficiente:** LoRA PEFT ($r=8$, adicionando adaptadores nas projeções $q, k, v, \text{out}$ das camadas de atenção e MLP).
* **Topologia:** 20 clientes particionados via Dirichlet ($\alpha=0.1$) em `CIFAR-10 Drift`.
* **Regime de Treinamento:** 50 rounds globais, 5 épocas locais por round, batch size 32.
* **Regime de Drift:** Periódico a cada $L=5$ rodadas (10 ciclos de inversão ao longo de 50 rounds).

---

## 2. Análise Científica do Concept Drift Gradual

Foi avaliado se a curva senoidal obtida no gradual com vales próximos a $36.5\%\text{--}40.0\%$ de acurácia estava correta do ponto de vista teórico de Teoria da Decisão e Aprendizado Estatístico.

### 2.1. O Teto Irredutível de Bayes no Ponto Neutro ($\alpha=0.50$)
Quando o drift gradual atinge o meio do ciclo ($\alpha=0.50$):
* A probabilidade de uma imagem $X$ pertencer ao rótulo da Phase 1 é $50\%$ e ao da Phase 2 é $50\%$.
* A taxa ótima de acurácia de qualquer classificador perfeito (Bayes Optimal Classifier) é limitada superiormente por:
  $$\text{Bayes Ceiling} = \max(\alpha, 1 - \alpha) = \max(0.5, 0.5) = 50\%$$
* Como o modelo LoRA-CLIP atingiu entre $36.5\%$ e $40.0\%$ de acurácia, ele operou em **$73.0\%\text{--}80.0\%$ de eficiência teórica ótima**, provando que o modelo não entrou em colapso; ele apenas atingiu o limite matemático da ambiguidade dos rótulos.

### 2.2. O Fenômeno de Histerese (Inércia de Gradiente)
Foi detectado que o modelo apresenta desempenho assimétrico dependendo do sentido da transição:
* **Fase Ascendente ($\alpha: 0.0 \to 1.0$):** O modelo sofre inércia ao tentar desaprender a Phase 1 e convergir para a Phase 2.
* **Fase Descendente ($\alpha: 1.0 \to 0.0$):** O modelo recupera acurácia mais velozmente devido à facilidade de reativar pesos adaptativos prévios (*catastrophic forgetting* mitigado pelo congelamento da ViT).

---

## 3. Geração dos Gráficos e Artefatos do Benchmark ASTRA

Com a finalização das 5 simulações completas do cenário gradual em disco, foi construído o script gerador [`src/system/plot/plot_astra_diagnostics.py`](file:///home/jose/FL_ServerPuma/src/system/plot/plot_astra_diagnostics.py) para extrair os dados e plotar em alta resolução (300 DPI):

1. **Runs Individuais do Gradual (1 a 5):**
   * [`01_astra_gradual_run1.png`](file:///home/jose/FL_ServerPuma/src/results/astra_periodic_gradual_drift/01_astra_gradual_run1.png) a [`05_astra_gradual_run5.png`](file:///home/jose/FL_ServerPuma/src/results/astra_periodic_gradual_drift/05_astra_gradual_run5.png)
2. **Média Geral com Margem de Confiança ($\pm 1\sigma$):**
   * [`06_astra_gradual_mean_general.png`](file:///home/jose/FL_ServerPuma/src/results/astra_periodic_gradual_drift/06_astra_gradual_mean_general.png)
3. **Diagnóstico Científico — Acurácia vs. Teto de Bayes:**
   * [`07_astra_gradual_alpha_ceiling.png`](file:///home/jose/FL_ServerPuma/src/results/astra_periodic_gradual_drift/07_astra_gradual_alpha_ceiling.png)
4. **Comparativo Trajetória Abrupta vs. Gradual:**
   * [`08_astra_abrupt_vs_gradual.png`](file:///home/jose/FL_ServerPuma/src/results/astra_periodic_gradual_drift/08_astra_abrupt_vs_gradual.png)
5. **Curva de Dispersão e Histerese (Acurácia vs. $\alpha$):**
   * [`09_astra_gradual_acc_vs_alpha.png`](file:///home/jose/FL_ServerPuma/src/results/astra_periodic_gradual_drift/09_astra_gradual_acc_vs_alpha.png)

---

## 4. Elaboração do Relatório Executivo `dia_das_criancas.md`

Foi consolidado o relatório formal [`dia_das_criancas.md`](file:///home/jose/FL_ServerPuma/dia_das_criancas.md) contendo os 5 tópicos estruturais exigidos pelo usuário:

1. **Status de Conclusão e Visualizações Geradas:** Confirmação da integridade de todas as simulações e catálogo de figuras.
2. **Avaliação Crítica dos Cenários:** Validação teórica e metodológica do dente de serra abrupto vs. onda senoidal gradual.
3. **Aprofundamento Teórico:** Formalização matemática do limite de Bayes e histerese.
4. **Tabela Comparativa Quantitativa:**
   * Média Geral Abrupto ($66.10\% \pm 15.10\%$) vs. Gradual ($55.43\% \pm 15.44\%$).
   * Acurácia Mínima no Nadir: $51.1\%\text{--}54.2\%$ (Abrupto) vs. $36.5\%\text{--}40.0\%$ (Gradual em $\alpha=0.50$).
   * Picos de Convergência: $\approx 85\%\text{--}92\%$.
5. **Apontamentos Metodológicos para Paper:** Recomendações de escrita para submissão no SBRC 2026 / Periódico.

---

## 5. Descoberta e Diagnóstico: O Gargalo da Seleção Estática de Clientes

Durante a revisão do código-fonte em `server.py`, identificou-se um comportamento crítico na lógica de amostragem:

```python
# Código Anterior em server.py:
while len(self.client_connections) < self.args.clients_per_round:
    conn, addr = s.accept()
    ...
```

### O Problema Identificado:
* O script `run.sh` instanciava 20 processos clientes (`--max-clients 20`).
* No entanto, o `server.py` só realizava o `accept()` dos **primeiros 5 clientes que conseguissem conectar no socket TCP**.
* Assim que o 5º cliente conectava, o servidor encerrava o loop de conexões e iniciava o Round 1.
* Os outros 15 clientes ficavam tentando conectar em vão até atingir o timeout de 120 segundos.
* **Consequência:** Os mesmos 5 clientes fixos treinavam em todas as 50 rodadas!

O usuário observou e questionou:
> *"essa parte tu ta me dizendo q tipo sao 20 clientes e seleciona 5 por round, mas se sempre os clientes de 1 a 5 se conectarem primeiro vai ser sempre eles na bateria de simulacoes? quero fazer exatamente essa recomendacao"*

---

## 6. Implementação da Amostragem Dinâmica Determinística Round a Round

A recomendação foi prontamente aceita e **implementada em todos os componentes do sistema**:

### 6.1. Servidor (`src/system/server.py`)
1. **Espera de Conexão Total:**
   ```python
   total_clients_to_wait = max(self.args.clients_per_round, getattr(self.args, 'max_clients', self.args.clients_per_round))
   while len(self.client_connections) < total_clients_to_wait:
       conn, addr = s.accept()
   ```
   O servidor agora aguarda todos os 20 clientes conectarem antes de iniciar a rodada 1.
2. **Sorteio Pseudo-aleatório com Seed por Rodada:**
   ```python
   round_seed = (getattr(self.args, 'seed', 42) or 42) + round_num
   rng = random.Random(round_seed)
   selected_indices = set(rng.sample(range(total_connected), self.args.clients_per_round))
   ```
3. **Sinalização Skip/Idle para os Não Selecionados:**
   O servidor envia `"skip"` para os 15 clientes não sorteados e dispara threads de treinamento apenas para os 5 selecionados.
4. **Sincronização de Fim de Round:**
   O servidor envia o pacote `'end'` para todos os 20 sockets TCP simultaneamente, mantendo a federação sincronizada.

### 6.2. Cliente (`src/system/client.py`)
* O cliente identifica a mensagem `"skip"`.
* Registra o consumo de RAM no estado de repouso (`IDLE_ROUND`).
* Não realiza backpropagation nem inferência local, economizando GPU e CPU.
* Aguarda o pacote de encerramento do round e avança diretamente para a próxima rodada.

### 6.3. Script de Lançamento (`src/system/run.sh`)
* Adicionada a flag `--seed <n>` (default: 42).
* Cálculo de semente determinística por simulação:
  $$\text{RUN\_SEED} = \text{SEED} + (\text{RUN} - 1) \times 1000$$
  Isso garante que cada simulação de 1 a 5 tenha amostragens diferentes, mas 100% reprodutíveis caso reexecutadas.

---

## 7. Esclarecimento Teórico: `--drift-clients` vs. `--clients`

O usuário questionou o significado da flag `--drift-clients`:
> *"--drift-clients '2,8,9,16,19' \ oq significa isso aq? vai usar so esses 5 clientes?"*  
> *"certo oq vc ta me dizendo é q no meu teste todos os clientes sofrem drift, mas no astras apenas 5 sofrem"*

### A Diferenciação Conceitual:

| Argumento | Função no Sistema | Configuração no ASTRA Oficial |
| :--- | :--- | :--- |
| `--max-clients 20` | Quantidade total de clientes instanciados na federação | 20 clientes |
| `--clients 5` | Quantidade de clientes amostrados para treinar por round | 5 clientes sorteados a cada round |
| `--drift-clients` | **Quais clientes sofrem perturbação nos dados (Concept Drift)** | **`all` (todos os 20 sofrem drift simultaneamente)** |

* Se fosse configurado `--drift-clients "2,8,9,16,19"`, teríamos um cenário de **Drift Parcial**, onde apenas esses 5 clientes teriam rótulos trocados, e os outros 15 teriam dados estáveis.
* No **ASTRA Oficial**, o parâmetro correto é **`--drift-clients all`**: todos os 20 clientes sofrem a inversão cíclica de classes a cada 5 rounds, e o servidor sorteia 5 clientes aleatórios para cada iteração do FedAvg.

---

## 8. Estrutura dos Dados: Particionamento Non-IID e Phase 1 vs. Phase 2

O usuário solicitou uma explicação detalhada sobre a distribuição dos dados e o papel das fases:

### 8.1. Particionamento Dirichlet Non-IID
* Os 60.000 dados do CIFAR-10 são fatiados entre os 20 clientes com $\alpha_{dir} = 0.1$.
* Cada cliente recebe uma distribuição assimétrica de classes, reproduzindo a heterogeneidade estatística severa de redes reais de borda.

### 8.2. Phase 1 vs. Phase 2 (*Pairwise Label Swap*)
* **Phase 1 (Conceito Original):**
  * Classes originais do CIFAR-10 ($0=\text{avião}, 1=\text{carro}, 2=\text{pássaro}, 3=\text{gato}, \dots$).
* **Phase 2 (Conceito Perturbado):**
  * As imagens visuais permanecem rigorosamente as mesmas.
  * Os rótulos sofrem inversão semântica pareada:
    * $0 \leftrightarrow 2$ (Avião $\leftrightarrow$ Pássaro)
    * $1 \leftrightarrow 9$ (Carro $\leftrightarrow$ Caminhão)
    * $3 \leftrightarrow 5$ (Gato $\leftrightarrow$ Cachorro)
    * $4 \leftrightarrow 7$ (Cervo $\leftrightarrow$ Cavalo)
    * $6 \leftrightarrow 8$ (Sapo $\leftrightarrow$ Navio)
* **Objetivo Científico:** Testar a plasticidade dos adaptadores LoRA para remapear as projeções visuais congeladas do CLIP sem esquecer as representações globais.

---

## 9. Automação e Disparo das Baterias Experimentais Completas

O usuário autorizou o início das novas execuções com a separação explícita das pastas de resultados:
* Gradual: `astra_allgradual_drift`
* Abrupto: `astra_allabrupt_drift`

Para garantir execução contínua sem necessidade de intervenção manual entre os dois experimentos, foi criado o orquestrador [`src/system/run_both_astra.sh`](file:///home/jose/FL_ServerPuma/src/system/run_both_astra.sh):

```bash
#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================="
echo " [1/2] INICIANDO BATERIA GRADUAL (astra_allgradual_drift)"
echo "================================================================="
./run.sh \
    --model clip \
    --dataset Cifar10_drift \
    --clients 5 \
    --max-clients 20 \
    --rounds 50 \
    --simulations 5 \
    --seed 42 \
    --drift \
    --drift-type gradual \
    --drift-freq 5 \
    --drift-clients all \
    --exp-name astra_allgradual_drift \
    --auto-next

echo "================================================================="
echo " [2/2] INICIANDO BATERIA ABRUPTA (astra_allabrupt_drift)"
echo "================================================================="
./run.sh \
    --model clip \
    --dataset Cifar10_drift \
    --clients 5 \
    --max-clients 20 \
    --rounds 50 \
    --simulations 5 \
    --seed 42 \
    --drift \
    --drift-type abrupt \
    --drift-freq 5 \
    --drift-clients all \
    --exp-name astra_allabrupt_drift \
    --auto-next

echo "================================================================="
echo " TODAS AS SIMULAÇÕES FORAM CONCLUÍDAS COM SUCESSO!"
echo "================================================================="
```

---

## 10. Evidências de Execução em Tempo Real (GPU e Logs)

O script foi iniciado com sucesso em segundo plano e validado com os seguintes dados ao vivo:

### 10.1. Handshake do Servidor e Sorteio de Clientes no Round 1:
```text
Client 20 connected: ('127.0.0.1', 44514)
All 20 clients connected. Starting training...

--- Round 1/50 ---
Round 1: Clientes selecionados (5/20): [5, 9, 11, 18, 19]
Round 1: Handling client 5
Round 1: Handling client 19
Round 1: Handling client 18
Round 1: Handling client 11
Round 1: Handling client 9
```
* Todos os **20 clientes conectaram**.
* O servidor realizou o sorteio determinístico: clientes `5, 9, 11, 18 e 19`.
* Os outros 15 clientes entraram em repouso sem travar ou tomar timeout.

### 10.2. Telemetria de Hardware via `nvidia-smi`:
* **Placa de Vídeo:** NVIDIA GeForce RTX 4090 (24 GB VRAM)
* **Consumo de Memória de Vídeo:** 17.212 MiB alocados
* **Processos Ativos na GPU:** 21 processos Python em execução simultânea (1 servidor e 20 clientes da federação)
* **Temperatura e Carga:** 48°C, operação estável

---

## 📌 Próximos Passos Recomendados

1. **Monitoramento dos Experimentos:**
   Acompanhar o progresso dos rounds com:
   ```bash
   tail -f src/results/astra_allgradual_drift/logs/sim_1/server.log
   ```
2. **Geração Automática de Gráficos Pós-Treino:**
   Assim que ambas as baterias finalizarem (previsão de ~4 horas), executar o script `plot_astra_diagnostics.py` apontando para as novas pastas `astra_allgradual_drift` e `astra_allabrupt_drift` para gerar as curvas finais para o paper do SBRC 2026.
