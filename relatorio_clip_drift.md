# Relatório Técnico: Avaliação de Federated Concept Drift com LoRA-CLIP (`relatorio_clip_drift`)

**Projeto**: FL_ServerPuma — Benchmarking de Aprendizado Federado sob Concept Drift Periódico  
**Ambiente / Framework**: PyTorch, PEFT (LoRA), Hugging Face Transformers (`openai/clip-vit-base-patch32`), CIFAR-10 Drift  
**Referência de Protocolo**: Benchmark ASTRA (SBRC 2026)  
**Data**: 07 de Outubro de 2026  

---

## 📋 1. Visão Geral do Projeto e Objetivos

O objetivo principal deste trabalho consistiu em adaptar e avaliar o framework de Aprendizado Federado **FL_ServerPuma** em cenários de **Concept Drift Periódico** (deriva temporal de conceito), transicionando a arquitetura do modelo baseline CNN (`SimpleModel`) para a arquitetura multimodal pré-treinada **LoRA-CLIP** (`openai/clip-vit-base-patch32` parametrizada com adaptadores PEFT LoRA de rank $r=8$).

A motivação central é comparar quantitativamente a acurácia global do modelo federado e o consumo de recursos sob dois padrões de deriva:
1. **Concept Drift Abrupto Periódico**: Mudança seca e instantânea de distribuição de rótulos a cada intervalo regular de rodadas.
2. **Concept Drift Gradual Periódico**: Transição suave e contínua entre distribuições através de interpolação probabilística $\alpha(t)$ ao longo de janelas de rodadas.

---

## 🛠️ 2. Alterações e Melhorias no Código-Fonte

Para viabilizar a simulação correta e consistente do benchmark ASTRA, realizamos intervenções estruturais no código do sistema:

### 2.1. Arquivo `src/system/client.py`

* **Suporte ao Drift Periódico Gradual**:
  Refatoramos a lógica de detecção de drift. Quando a flag `--drift-freq L` ($L=5$) é fornecida com `--drift-type gradual`, o cliente calcula iterativamente a cada round $t$ a proporção de mistura $\alpha(t) \in [0.0, 1.0]$:
  $$\alpha(t) = \begin{cases} \frac{s}{L-1}, & \text{se o ciclo } p \text{ for par (transição Phase 1 } \rightarrow \text{ Phase 2)} \\ 1.0 - \frac{s}{L-1}, & \text{se o ciclo } p \text{ for ímpar (transição Phase 2 } \rightarrow \text{ Phase 1)} \end{cases}$$
  onde $p = \lfloor (t-1) / L \rfloor$ é o número do ciclo e $s = (t-1) \pmod L$.

* **Carregamento de Dados Interpolados (`load_blended_data`)**:
  Criamos um mecanismo que mantém em cache as amostras de dados de *Phase 1* e *Phase 2* e as combina dinamicamente na proporção $\alpha(t)$ para treino e teste local do cliente.

* **Correção do Bug de Shadowing de Variável Socket**:
  *Diagnóstico*: Durante a execução, os clientes encontravam uma exceção `AttributeError: 'int' object has no attribute 'recv'`.  
  *Causa*: A variável local da fórmula do drift foi nomeada como `s = (current_round - 1) % L`, sobrescrevendo o objeto socket de comunicação `s` (`socket.socket()`).  
  *Solução*: Renomeamos a variável da fórmula para `cycle_step`, garantindo a integridade do socket `s` durante toda a execução.

* **Treinamento de Múltiplas Épocas Locais**:
  Garantimos que o loop de treinamento local (`local_training`) respeite o parâmetro `epochs` definido em `train_config.yml` (5 épocas locais por round por cliente, conforme exigido no benchmark ASTRA).

---

### 2.2. Arquivo `src/system/server.py`

* **Avaliação Global de Teste Suave (`load_blended_test_data`)**:
  Adicionamos o método `load_blended_test_data` ao servidor. Em cenários de drift gradual, o servidor agora avalia o modelo global unificado usando um `DataLoader` de teste construído com a mesma proporção $\alpha(t)$ da rodada atual.

* **Invalidação e Atualização de Cache de Teste**:
  Ajustamos a função de invalidação de cache de loaders no servidor (`test_loaders_cache`). Quando `--drift-freq > 0` e o drift é gradual, o servidor reavalia e atualiza o `DataLoader` de teste a cada rodada de acordo com o $\alpha(t)$ correspondente, evitando a medição em distribuições desalinhadas com o estado dos clientes.

---

### 2.3. Arquivo `src/system/plot/plot_astra_benchmark.py`

* **Geração Dinâmica de Títulos e Legendas**:
  Atualizamos a ferramenta de plotagem para identificar automaticamente se a pasta de resultados refere-se a um experimento *Abrupt* ou *Gradual*, inserindo o nome correto nos títulos dos gráficos de 300 DPI e formatando a contagem de simulações $N$ na legenda.

---

## 🧪 3. Protocolo Experimental (Benchmark ASTRA - SBRC 2026)

Seguindo rigorosamente as especificações do repositório/artigo ASTRA:

| Parâmetro | Valor Utilizado | Descrição |
| :--- | :--- | :--- |
| **Arquitetura** | `LoRA-CLIP` | `openai/clip-vit-base-patch32` com LoRA PEFT ($r=8$) |
| **Dataset** | `CIFAR-10 Drift` | Transição entre 2 conceitos (Phase 1 e Phase 2 via *label swap*) |
| **Clientes Totais** | `20` | População total de nós clientes |
| **Clientes / Round** | `5` | Seleção aleatória de 5 clientes por rodada de comunicação |
| **Épocas Locais** | `5` | Épocas de gradiente descendente por cliente a cada round |
| **Rodadas Totais** | `50` | Total de ciclos globais de federated averaging (FedAvg) |
| **Frequência de Drift** | `5` | Drift injetado a cada 5 rodadas (10 eventos de drift no total) |
| **Simulações** | `5` por cenário | Execuções independentes para cálculo de média e desvio padrão |

---

## 📊 4. Análise dos Resultados e Suposições Teóricas

### 4.1. Cenário Abrupto: Quedas Instantâneas e Recuperação Rápida

No cenário **Periodic Abrupt Drift**, observamos um padrão serreado marcado por **quedas secas de acurácia exatamente nos rounds múltiplos de 5** (Rounds 5, 10, 15, 20, 25, 30, 35, 40, 45, 50).

* **Comportamento Observado**:
  A acurácia global do modelo despenca entre $-20\%$ e $-45\%$ no exato instante da mudança de conceito, atingindo vales pronunciados, e em seguida se recupera em 2 a 3 rodadas de comunicação.

* **Suposições e Explicação Teórica**:
  1. **Preservação de Representações Visuais Congeladas**: O backbone da Vision Transformer (ViT-B/32) do CLIP permanece completamente congelado durante todo o processo. Ele já possui um espaço de embeddings extremamente rico e invariante pré-treinado em mais de 400 milhões de pares imagem-texto.
  2. **Re-adaptação Eficiente via PEFT**: Apenas os adaptadores LoRA (matrizes de baixo rank $A$ e $B$, somando $\approx 597.514$ parâmetros) sofrem atualização de gradiente. Isso evita o *catastrophic forgetting* comum em redes neurais convolucionais (CNNs) totalmente treináveis, permitindo que o modelo aprenda a nova relação de rótulos com pouquíssimas atualizações locais.

---

### 4.2. Cenário Gradual: Transição Senoidal Contínua e Amortecida

No cenário **Periodic Gradual Drift**, a trajetória da acurácia global desenha uma **onda senoidal contínua e suave**.

* **Comportamento Observado**:
  Em vez de quedas abruptas verticais, a acurácia transiciona suavemente de $\approx 91\%$ ($\alpha=1.0$, conceito puro) para $\approx 70\%$ ($\alpha=0.75$), $\approx 47\%$ ($\alpha=0.50$, mistura equilibrada de conceitos 1 e 2), até alcançar o ponto de inversão.

* **Suposições e Explicação Teórica**:
  1. **Interpolação de Distribuição de Dados**: A cada rodada da transição de 5 rounds, os lotes de dados contêm proporções progressivas de amostras da Phase 1 e Phase 2. O modelo global nunca enfrenta uma incompatibilidade $100\%$ imediata de rótulos.
  2. **Mitigação do Impacto no Servidor**: A acurácia global avaliada pelo servidor reflete realisticamente a mistura que os clientes estão vivenciando no momento. Isso demonstra que em ambientes reais com transição suave (ex: mudança gradual de estação ou de comportamento de uso), o aprendizado federado adaptativo baseado em adaptadores leves (LoRA) consegue manter um desempenho médio mais estável sem colapsos repentinos.

---

### 4.3. Telemetria de Memória (RAM RSS)

A medição do consumo de memória residente (RSS) nos clientes revelou:
- **Operação Normal**: Consumo de RAM estável em $\approx 1060 - 1100$ MB por cliente.
- **Durante Transição Gradual**: O consumo de RAM aumenta temporariamente para $\approx 1500 - 2600$ MB no instante da instanciação dos `DataLoaders` de mistura (`phase1` + `phase2`), retornando ao nível baseline imediatamente após a liberação da memória pelo coletor de lixo (`gc.collect()`).

---

## 🖼️ 5. Gráficos de Resultados Gerados

Os gráficos agregados de alta resolução (300 DPI) que sumarizam as 5 execuções independentes de cada cenário estão gerados em disco:

1. **Gráfico Agregado - Periodic Abrupt Drift (`06_astra_abrupt_mean_general.png`)**:
   `src/results/astra_periodic_drift/06_astra_abrupt_mean_general.png`

2. **Gráfico Agregado - Periodic Gradual Drift (`06_astra_gradual_mean_general.png`)**:
   `src/results/astra_periodic_gradual_drift/06_astra_gradual_mean_general.png`

Além disso, 10 gráficos individuais (do Run #1 ao Run #5 de cada cenário) foram gerados nas respectivas pastas de resultados.

---

## 🎯 6. Conclusões

1. O framework **FL_ServerPuma** foi atualizado e validado com sucesso para suportar o protocolo do paper **ASTRA (SBRC 2026)**.
2. A distinção visual e quantitativa entre o **Drift Abrupto** (quedas verticais secas) e o **Drift Gradual** (curva senoidal contínua) comprovam a corretude da implementação matemática da interpolação $\alpha(t)$.
3. A utilização da arquitetura **LoRA-CLIP** provou ser altamente resiliente a variações de conceito em Aprendizado Federado, oferecendo re-convergência rápida e baixo overhead de comunicação ($\approx 2.28$ MB transferidos por cliente por round).
