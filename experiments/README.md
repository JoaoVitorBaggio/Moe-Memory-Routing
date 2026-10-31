# experiments/

Entradas finas do protocolo. O desenho está em [`docs/protocolo-experimental.md`](../docs/protocolo-experimental.md).

O passo sintético e os dois loaders leem [`configs/default.yaml`](../configs/default.yaml):

```bash
python3 experiments/smoke_train.py
python3 experiments/smoke_eval.py
python3 experiments/smoke_train.py --lambda 0
python3 experiments/train_tinystories.py --steps 1
python3 experiments/smoke_wikitext.py
python3 experiments/wall_latency.py
```

`smoke_train.py` ainda é um batch aleatório. `train_tinystories.py` faz um passo curto no corpus de desenvolvimento (streaming, tamanho do `smoke`). `smoke_wikitext.py` só confere train/valid/test e um batch; não treina. `wall_latency.py` sem CUDA explica a medição e não imprime ms. Nada disso é número de relatório e nada vai para `results/`.

## Corrida overnight (WikiText-2, um λ por vez)

O desenho não muda: `λ` sai de `loss.lambda_grid` (`0`, `0.5`, `1`), `β` fica `0.01`, `B` é 25% de `E`, arquitetura e tabela `c` ficam no YAML. fp32, sem `torch.compile`. Cada λ é um treino separado, reusando `prepare_train` / `train_one_batch`.

Da raiz do repositório, na GTX 1060:

```bash
python3 experiments/run_lambda_grid.py --device cuda
```

Padrão: uma época do split oficial `train`, `seq_len = model.max_seq_len` (256), batch operacional 4, semente 0, checkpoint a cada 25 passos. O batch **não** é eixo do protocolo; se o processo morrer por OOM, loss não finita ou crash, a grade retoma o checkpoint e desce o batch (4 → 2 → 1) sem pedir intervenção. SIGTERM/SIGINT espera o passo em curso, grava o checkpoint e a grade repete o mesmo batch. Erro de uso (λ fora da grade, corpus ilegível) não reduz batch. Um λ que esgota o lote 1 entra no resumo como falha e o próximo segue. O arquivo do checkpoint é o peso mais o Adam (centenas de MB em `outputs/`); o plano de hardware já reserva ~2 GB de disco quando há checkpoint.

Um λ só, ou uma fumaça de CPU (não é a corrida):

```bash
python3 experiments/train_wikitext.py --lambda 0.5 --device cuda
python3 experiments/train_wikitext.py --lambda 0 --steps 2 --device cpu --seq-len 32 --batch-size 2
```

`--epochs` (padrão 1) e `--steps` são exclusivos. `--seed` mexe no peso e na ordem dos blocos; o loader em si não embaralha. Retomar é repetir o mesmo comando: o cursor de sequências está em `outputs/wikitext/lambda-<λ>/checkpoint.pt` (gitignored), junto com `train.jsonl`. A grade escreve `outputs/wikitext/summary.json` e, por λ, `train.log`. Trocar `--seq-len` ou `--seed` no meio pede outro `--work-dir`.

O jsonl é traço de treino (`L`, `L_task`, `C_mem`, `|grad|`, trocas). Não é perplexidade de valid/test nem dose–resposta. O LM denso tem corrida própria, abaixo. LRU como adversário separado e o hold-out ainda não têm script de relatório. O emulador já aceita trocar o teto (`set_hot_budget`, 0,5× / 1× / 2×) para quando essa avaliação entrar.

## Baseline denso (WikiText-2)

Modelo 3 do protocolo: LM causal **sem** roteador, **sem** experts, **sem** emulador e **sem** `C_mem` / `L_balanço`. A loss é só `L_tarefa`.

Isso não é o ponto `λ=0` da grade. No YAML, `λ=0` é o baseline 1: o mesmo MoE (`E=32`, residência, roteador). `C_mem` é calculado e entra com peso zero, e a loss continua `L_tarefa + β L_balanço` (`β=0.01`). O denso não tem esses termos. `total` e `L_task` coincidem; no jsonl, `c_mem` e `balance` são `null`.

A largura do FFN é `k * expert_hidden` (1024 com o YAML atual). Parâmetros ativos ≈ tronco + `k`·expert, não os 47.939.328 do MoE inteiro. Com esse YAML o denso tem 16.090.880 parâmetros; o MoE ativo (tronco com roteador + `k` experts) fica em 16.389.888. A folga de 299.008 é o roteador, que o denso não tem, mais o viés `fc2` dos `k` experts. Atenção, embedding amarrada, camadas, cabeças e contexto são os do YAML. fp32, sem `torch.compile`.

O corpus padrão, e o de relatório, é o WikiText-2. Semente 0, uma época, `seq_len = model.max_seq_len` (256), batch operacional 4. OOM, loss não finita ou crash descem o batch (4 → 2 → 1) e retomam o checkpoint, com a mesma regra da grade. O Adam gravado é cópia: o estado vivo não muda de device.

Na GTX 1060:

```bash
python3 experiments/run_baseline.py --device cuda
```

Um corte curto em CPU. Este script não tem `--lambda`:

```bash
python3 experiments/train_baseline.py --steps 2 --device cpu --seq-len 32 --batch-size 2
```

TinyStories fica como caminho opcional de desenvolvimento. Não é o padrão. O contexto desse caminho, se `--seq-len` não vier, é o `smoke.seq_len` (32), como `train_tinystories.py`:

```bash
python3 experiments/train_baseline.py --corpus TinyStories --steps 2 --device cpu
```

A entrega cai em `outputs/baseline/wikitext/` (gitignored): `summary.json` do runner, mais `done.json`, `train.jsonl`, `train.log` e `checkpoint.pt`. O resumo usa os mesmos campos da grade para comparar: `status`, `steps`, `tokens`, `last_total`, `last_task`. `lambda` fica `null`. TinyStories, se rodar, vai para `outputs/baseline/tinystories/` e não mistura com esse resumo. Hold-out e dose–resposta continuam fora.

Não colocar saída grande nesta pasta. Métrica pequena vai para `results/`; checkpoint e dataset ficam de fora do Git. O home frio do MoE cai em `outputs/` (gitignored). O baseline denso não grava memmap de expert.
