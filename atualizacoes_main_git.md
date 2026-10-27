# Plano de Integração: Mesclando Atualizações da `main` no Projeto Local

## Visão Geral
A branch remota `origin/main` (e `origin/Dev-main`) possui 17 novos commits contendo melhorias importantes (suporte a SLM global evaluation, medição de bytes reais em sockets, refinamento de PaCA, novos orquestradores de experimentos, padronização de datasets, etc.).

No momento, o repositório local possui alterações não commitadas e arquivos novos correspondentes ao módulo de **Concept Drift no CIFAR-10** (desenvolvido e reportado no `relatorio_concept_drift_cifar10.md`).

Para trazer as novidades da `main` sem perder nada do trabalho de Concept Drift, realizaremos o salvamento seguro das alterações locais seguido do merge e resolução cuidadosa dos conflitos.

---

## User Review Required

> [!IMPORTANT]
> **Alterações Locais Preservadas**: As alterações não commitadas no seu ambiente (lógica de Concept Drift no `client.py`, `server.py`, `data_utils.py`, `run.sh`, etc.) serão primeiramente salvas em um commit local para evitar qualquer perda.
>
> > [!WARNING]
> **Resolução de Conflitos**: Arquivos como `src/system/client.py` e `src/system/server.py` passaram por reformulações significativas na `main`. Os conflitos serão resolvidos manualmente combinando o código atualizado da `main` com as funcionalidades de Concept Drift.

---

## Mudanças Propostas

### 1. Preservação do Estado Atual (Concept Drift)
- Criar um commit na branch atual (`Dev-main`) registrando todas as modificações e arquivos não monitorados relacionados ao Concept Drift (`generate_Cifar_drift.py`, `plot_drift_results.py`, `relatorio_concept_drift_cifar10.md`, etc.).

### 2. Integração com `origin/main`
- Executar `git merge origin/main`.
- Resolver conflitos nos seguintes arquivos:

#### [MODIFY] [client.py](file:///home/jose/FL_ServerPuma/src/system/client.py)
- Reconciliar os novos argumentos de CLI (`--drift`, `--drift-type`, `--drift-round`, etc.), a função `load_blended_data` e o recarregamento de dados no loop de treinamento com os novos recursos de socket/bytes reais e adaptação de PaCA da `main`.

#### [MODIFY] [server.py](file:///home/jose/FL_ServerPuma/src/system/server.py)
- Manter o suporte ao dataset `Cifar10_drift` no pattern matching e no dimensionamento do modelo, preservando as novas correções de quantização int8 e avaliação global de SLM da `main`.

#### [MODIFY] [run.sh](file:///home/jose/FL_ServerPuma/src/system/run.sh)
- Manter a lógica de `trap` Ctrl+C e `tmux kill-session` prévio junto com os novos parâmetros/configurações trazidos pela `main`.

#### [MODIFY] [data_utils.py](file:///home/jose/FL_ServerPuma/src/utils/data_utils.py)
- Garantir a presença do argumento `phase=None` em `read_data` e `read_client_data` para suporte a `_phase1.npz` e `_phase2.npz`.

#### [MODIFY] [prunning.py](file:///home/jose/FL_ServerPuma/src/system/prunning.py)
- Manter `Cifar10_drift` no mapeamento de dimensões para poda adaptativa.

---

## Plano de Verificação

### Testes Automatizados e Sintaxe
- Verificar a compilação/sintaxe Python de todos os arquivos modificados:
  ```bash
  python3 -m py_compile src/system/client.py src/system/server.py src/utils/data_utils.py src/dataset/generate_Cifar_drift.py
  ```

### Verificação Manual
- Validar `git status` e `git log -n 5` para confirmar o merge bem-sucedido.
- Testar a execução do menu de ajuda do cliente para garantir que os parâmetros de drift continuam disponíveis:
  ```bash
  uv run python src/system/client.py --help
  ```
