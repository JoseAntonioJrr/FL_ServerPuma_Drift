# Walkthrough — Treinamento Federado de SLM com PEFT/LoRA e OPALA

Implementamos com sucesso a extensão do framework **PUMA** para suportar o treinamento federado e fine-tuning de **Small Language Models (SLMs)** com **PEFT/LoRA** e a técnica de poda adaptativa baseada na filosofia **OPALA**.

---

## 🛠️ Mudanças Realizadas

### 1. Dependências do Ecossistema PEFT/SLM
- **Arquivo**: [pyproject.toml](file:///home/jose/veiga/FL_ServerPuma/pyproject.toml)
- **Modificação**: Adicionadas as dependências `peft>=0.7.0`, `datasets`, `sentencepiece` e `protobuf`. Sincronizadas com o ambiente local.

### 2. Módulo de Modelo SLM + LoRA
- **Arquivo**: [slm_model.py](file:///home/jose/veiga/FL_ServerPuma/src/system/slm_model.py)
- **Funções**:
  - `load_slm_with_lora`: Carrega a SLM base congelada (`TinyLlama-1.1B`) e adiciona adaptadores LoRA treináveis.
  - `get_lora_state_dict` / `set_lora_state_dict`: Permite extrair/injetar apenas as matrizes LoRA treináveis (economizando banda de comunicação).
  - `evaluate_perplexity`: Avalia o modelo final calculando a perplexidade no dataset de validação.

### 3. Poda Adaptativa de LoRA (OPALA-style)
- **Arquivo**: [prunning_lora.py](file:///home/jose/veiga/FL_ServerPuma/src/system/prunning_lora.py)
- **Estratégia**:
  - A importância de cada dimensão de rank $i$ é calculada por: $\text{importancia}[i] = ||B_{[:, i]}||_2 \times ||A_{[i, :]}||_2$.
  - Remove dinamicamente as dimensões de menor rank (OPALA) para clientes com limitações de recursos.
  - No servidor, os pesos de tamanhos heterogêneos são alinhados à estrutura global com máscaras e integrados via média ponderada.

### 4. Geração de Dataset
- **Arquivo**: [generate_Alpaca.py](file:///home/jose/veiga/FL_ServerPuma/src/dataset/generate_Alpaca.py)
- **Modificação**: Baixa o dataset `yahma/alpaca-cleaned`, tokeniza com o tokenizer do TinyLlama e divide de forma Non-IID baseada no tamanho da instrução.
- **Status**: Já foi executado com sucesso e os arquivos `0.npz`, `1.npz` e `2.npz` foram gerados para os clientes em [Alpaca/](file:///home/jose/veiga/FL_ServerPuma/src/dataset/Alpaca/).

### 5. Servidor e Cliente FL para SLM
- **Arquivos**: [server_slm.py](file:///home/jose/veiga/FL_ServerPuma/src/system/server_slm.py) e [client_slm.py](file:///home/jose/veiga/FL_ServerPuma/src/system/client_slm.py)
- **Fluxo**:
  - O servidor gerencia o LoRA global, calcula os thresholds de tempo e aplica a poda LoRA antes de enviar aos clientes.
  - Os clientes recebem o LoRA, treinam apenas os parâmetros adaptadores no dataset Alpaca e enviam o LoRA atualizado de volta.
  - Resultados de perplexidade e perda são salvos em formato `.h5` na pasta `results/`.

### 6. Orquestração e Tmux
- **Arquivos**: [run_slm.sh](file:///home/jose/veiga/FL_ServerPuma/src/system/run_slm.sh) e [generate_compose.py](file:///home/jose/veiga/FL_ServerPuma/generate_compose.py)
- **Modificação**: Criado script automatizado via Tmux para simulação local e adicionada a flag `--mode slm` na geração do Docker Compose.

---

## 🚀 Como Executar na Prática

Você pode rodar a simulação local dividida em painéis no Tmux executando o script `run_slm.sh` de dentro de `src/system`:

```bash
cd src/system
./run_slm.sh --clients 3 --rounds 3 --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --device cuda
```

### Principais Opções de Execução:
- `--clients`: Número de clientes (geramos dados para 3 clientes).
- `--rounds`: Número de rodadas de comunicação federada.
- `--prune`: `0` para habilitar a poda adaptativa OPALA (padrão) ou `1` para desabilitar (LoRA completo).
- `--device`: `cuda` para rodar na GPU ou `cpu` para testes rápidos de CPU.
