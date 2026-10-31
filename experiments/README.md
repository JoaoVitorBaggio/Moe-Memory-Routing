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

Padrão: uma época do split oficial `train`, `seq_len = model.max_seq_len` (256), batch operacional 4, sementes `0 .. eval.seeds-1` (`seeds: 3` no YAML, ou seja 0, 1 e 2), checkpoint a cada 25 passos. `--seed N` treina só essa semente. O batch **não** é eixo do protocolo; se o processo morrer por OOM, loss não finita ou crash, a grade retoma o checkpoint e desce o batch (4 → 2 → 1) sem pedir intervenção. SIGTERM/SIGINT espera o passo em curso, grava o checkpoint e a grade repete o mesmo batch. Erro de uso (λ fora da grade, corpus ilegível) não reduz batch. Um λ que esgota o lote 1 entra no resumo como falha e o próximo segue. O arquivo do checkpoint é o peso mais o Adam (centenas de MB em `outputs/`); o plano de hardware já reserva ~2 GB de disco quando há checkpoint.

Um λ só, ou uma fumaça de CPU (não é a corrida):

```bash
python3 experiments/train_wikitext.py --lambda 0.5 --device cuda
python3 experiments/train_wikitext.py --lambda 0 --steps 2 --device cpu --seq-len 32 --batch-size 2
```

`--epochs` (padrão 1) e `--steps` são exclusivos. A semente mexe no peso e na ordem dos blocos; o loader em si não embaralha. Cada semente repete a grade inteira antes da próxima. Retomar é repetir o mesmo comando: o cursor de sequências está no `checkpoint.pt` daquela semente e daquele λ. A grade escreve `outputs/wikitext/summary.json` (status e caminho de cada semente) e, por job, `train.log` mais `train.jsonl`. Trocar `--seq-len` no meio pede outro `--work-dir`.

```text
outputs/wikitext/summary.json
outputs/wikitext/seed-<n>/lambda-<λ>/checkpoint.pt
outputs/wikitext/seed-<n>/lambda-<λ>/train.jsonl
outputs/wikitext/seed-<n>/lambda-<λ>/train.log
```

Semente 0 já gravada no layout antigo continua lá, se `seed-0/lambda-<λ>/` ainda não tiver `done.json`, `checkpoint.pt`, `train.jsonl` nem `train.log`:

```text
outputs/wikitext/lambda-<λ>/
```

Se os dois tiverem artefato, vale `seed-0`. Para migrar à mão, mover `lambda-<λ>/` para `seed-0/lambda-<λ>/` antes de relançar. `train_wikitext.py` sozinho continua sendo um job (`--seed`, padrão 0); o diretório padrão segue a mesma regra.

O jsonl é traço de treino (`L`, `L_task`, `C_mem`, `|grad|`, trocas). Não é perplexidade de valid/test nem dose–resposta. O LM denso tem corrida própria, abaixo. O hold-out, que também varre `eval.B_mult` sem retreinar, está na seção seguinte. O modelo 2 não tem treino próprio: é o mesmo `λ=0` e o mesmo LRU, nomeado em `run_model_comparison.py`. O teto da avaliação muda por `set_hot_budget` (0,5× / 1× / 2×).

## Baseline denso (WikiText-2)

Modelo 3 do protocolo: LM causal **sem** roteador, **sem** experts, **sem** emulador e **sem** `C_mem` / `L_balanço`. A loss é só `L_tarefa`.

Isso não é o ponto `λ=0` da grade. No YAML, `λ=0` é o baseline 1: o mesmo MoE (`E=32`, residência, roteador). `C_mem` é calculado e entra com peso zero, e a loss continua `L_tarefa + β L_balanço` (`β=0.01`). O denso não tem esses termos. `total` e `L_task` coincidem; no jsonl, `c_mem` e `balance` são `null`.

A largura do FFN é `k * expert_hidden` (1024 com o YAML atual). Parâmetros ativos ≈ tronco + `k`·expert, não os 47.939.328 do MoE inteiro. Com esse YAML o denso tem 16.090.880 parâmetros; o MoE ativo (tronco com roteador + `k` experts) fica em 16.389.888. A folga de 299.008 é o roteador, que o denso não tem, mais o viés `fc2` dos `k` experts. Atenção, embedding amarrada, camadas, cabeças e contexto são os do YAML. fp32, sem `torch.compile`.

O corpus padrão, e o de relatório, é o WikiText-2. Sementes `0 .. eval.seeds-1` (3 no YAML), uma época, `seq_len = model.max_seq_len` (256), batch operacional 4. `--seed N` limita a uma semente. OOM, loss não finita ou crash descem o batch (4 → 2 → 1) e retomam o checkpoint, com a mesma regra da grade. Uma semente que falha não cancela a próxima. O Adam gravado é cópia: o estado vivo não muda de device.

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

A entrega cai em `outputs/baseline/wikitext/` (gitignored). O `summary.json` fica na raiz e lista cada semente (`per_seed`: status, `work_dirs`, `logs`). O job em si fica em `seed-<n>/` (`done.json`, `train.jsonl`, `train.log`, `checkpoint.pt`). O resumo usa os mesmos campos da grade para comparar: `status`, `steps`, `tokens`, `last_total`, `last_task`. `lambda` fica `null`. TinyStories, se rodar, vai para `outputs/baseline/tinystories/seed-<n>/` e não mistura com o resumo do WikiText-2. O hold-out lê o checkpoint da semente 0. Na curva de `B` esse checkpoint entra como `n/a`: o denso não tem teto quente.

Semente 0 já gravada solta na raiz (`checkpoint.pt` ou `done.json` ao lado do `summary.json`, sem `seed-0/`) continua nessa raiz. Se `seed-0/` também tiver artefato, vale `seed-0`. Para migrar à mão, mover o checkpoint, o `done.json` e os logs para `seed-0/` antes de relançar. `train_baseline.py` sozinho é um job; o `--work-dir` padrão já é `seed-<n>/`, com a mesma exceção da raiz antiga.

## Hold-out (perplexidade no WikiText-2)

O treino não calcula este número. `experiments/run_holdout_eval.py` lê os checkpoints já gravados e mede a perplexidade de token nos splits oficiais `valid` (seleção) e `test` (relatório). Não treina. Se `corpus.paths.WikiText-2` for um diretório local, não baixa o Hub. Em seguida, sem retreinar, cada MoE repete valid e test em `eval.B_mult`.

O empacotamento é o do treino (BPE do GPT-2, `<|endoftext|>` entre artigos, blocos não sobrepostos via `iter_documents` / `iter_token_ids` / `iter_batches`, sem o embaralhamento da época). O tensor de treino bate com `load_train_sequences`. `seq_len` sai do checkpoint — 256 na corrida da GTX 1060, que é `model.max_seq_len`. A CE é `task_cross_entropy`: a posição `t` prediz `t+1`, e o resto que não fecha `seq_len` cai fora, como no treino. Perplexidade = `exp` dessa média por token previsto. Bits por byte = soma da NLL em nats / `ln(2)` / bytes UTF-8 do decode dos tokens que entraram na conta, sem o EOT. Não é a perplexidade por palavra dos papers de WikiText: o denominador aqui é token, para os quatro checkpoints serem comparáveis entre si e com a `L_task`.

O denso e o MoE não compartilham classe. `λ=0` continua sendo MoE (roteador, emulador, `β`). O roteador vê `ℓ(e)`. Cada split recomeça o LRU no `B` do YAML — não reaproveita o cache do fim do treino, nem o do outro split — e dá um passo de emulador a cada batch gravado no checkpoint. `--batch-size` só parte o forward; o grão do LRU não muda, então a PPL do MoE não depende desse microbatch. Os pesos ficam no device do forward, em fp32. O script não chama `sync_physical` no memmap do treino.

Cada split grava `swaps`, `bytes_moved` e a parede do forward (`wall_p50_ms`, `wall_p95_ms`, `wall_clock`, `wall_report_grade`). No CUDA o cronômetro é o de [`docs/plano-de-hardware.md`](../docs/plano-de-hardware.md): `materialize()` fora, `synchronize` + forward + `synchronize`, `emulator.step` fora. **Número de relatório da parede só na GTX 1060**, com o batch do checkpoint e o split inteiro. Em CPU `wall_clock` é `unavailable` e os ms ficam `null`. `wall_latency.py` continua sendo o smoke sintético.

`smoke_eval.py` segue sendo um forward de init aleatória. Não é este número.

Na máquina do treino (Fedora, GTX 1060), a partir da raiz, com os quatro checkpoints no lugar:

```bash
python3 experiments/run_holdout_eval.py --device cuda
```

Layout esperado, semente 0 (gitignored). A descoberta prefere `seed-0` e, se o `checkpoint.pt` novo não existir, aceita o caminho antigo:

```text
outputs/baseline/wikitext/seed-0/checkpoint.pt
outputs/wikitext/seed-0/lambda-0/checkpoint.pt
outputs/wikitext/seed-0/lambda-0.5/checkpoint.pt
outputs/wikitext/seed-0/lambda-1/checkpoint.pt
```

Antigo, ainda lido quando é o único checkpoint da semente 0:

```text
outputs/baseline/wikitext/checkpoint.pt
outputs/wikitext/lambda-<λ>/checkpoint.pt
```

As sementes 1 e 2 ficam no disco do treino. Este script não as agrega. `--work-dir` aponta para a pasta que contém o `checkpoint.pt` (`seed-0` ou a raiz antiga).

Um checkpoint só:

```bash
python3 experiments/run_holdout_eval.py --checkpoint outputs/wikitext/seed-0/lambda-0.5/checkpoint.pt --device cuda
python3 experiments/run_holdout_eval.py --work-dir outputs/baseline/wikitext/seed-0 --device cuda
```

Um corte de CPU que não é o relatório (código de saída 7). Não mexe no `seq_len` do checkpoint — isso trocaria o contexto do treino:

```bash
python3 experiments/run_holdout_eval.py --checkpoint outputs/wikitext/seed-0/lambda-0/checkpoint.pt --max-batches 1 --device cpu
```

Se o forward estourar a VRAM, repetir com `--batch-size 1` (ou 2). No MoE a PPL continua a do batch gravado no checkpoint. fp32, batch modesto: o treino operacional é 4, e o eval usa esse valor salvo a grade ter descido para 2 ou 1.

A saída é `holdout.json` ao lado de cada `checkpoint.pt` e, na descoberta dos quatro, `outputs/holdout/summary.json` (os dois gitignored, como o resto de `outputs/`). O JSON traz, por split, perplexidade, `nll_nats`, bits por byte, tokens previstos, sequências, o `seq_len`, o batch do checkpoint, os passos/cursor do treino, `swaps`, `bytes_moved` e o bloco de parede. `wall_note` diz quando o ms é coluna de relatório. `protocol_complete` fica verdadeiro quando valid e test inteiros rodaram no `seq_len` do checkpoint — isso não exige GPU. Checkpoint ausente na descoberta, `--max-batches` ou `--seq-len` diferente do treino: código 7, e o JSON não é número de relatório. `wall_report_grade` falso também tira a parede da tabela.

A dose–resposta usa o mesmo `score_split`, nos mesmos checkpoints (semente 0 na descoberta; o layout antigo entra só quando é o único `checkpoint.pt`). O `B` do YAML continua sendo a medição de referência (`splits` no `holdout.json`). Para cada multiplicador, `prepare_split_budget` recomeça o LRU nesse `B` e `set_hot_budget` aplica o teto sem apagar o carimbo — não há outro `reset` antes do forward, não há `--b-mult` e o script não atribui `hot_slots` à parte. A parede e os bytes desse passe são os do mesmo forward. O denso não entra na curva: `status` `n/a`. A saída extra é `dose_response.json` e `dose_response.md` (a tabela markdown é PPL por multiplicador; o JSON do split também traz a parede), ao lado do `summary.json` na descoberta, ou ao lado do `holdout.json` quando se pede um checkpoint só sem `--summary`. `dose_response: true` no resumo significa que a curva foi medida.

Agregar as sementes além da 0 e a correlação proxy×emulador continuam fora deste script. O treino das sementes é a grade e o baseline (`eval.seeds`). O rótulo do modelo 2 não é uma passagem a mais: é o mesmo `λ=0` da semente 0.

## Comparação dos quatro braços (modelo 2)

`experiments/run_model_comparison.py` não treina e não mede. Ele nomeia os braços do protocolo e escreve o schema da tabela que o relatório preenche. O checkpoint é o da semente 0, o mesmo que `run_holdout_eval.py` descobre (`prefer_checkpoint`: `seed-0` se o arquivo existir, senão o layout antigo, senão o caminho `seed-0`).

| id | Rótulo | Código hoje |
| --- | --- | --- |
| model1 | MoE top-k λ=0 | `MemoryMoEModel` + `ResidencyEmulator` (LRU), `seed-0/lambda-0` |
| model2 | MoE + LRU adversary | o mesmo código, o mesmo checkpoint e o mesmo λ=0 do model1 |
| model3 | Denso | `DenseLanguageModel`; não é o MoE com λ=0; `seed-0` do baseline |
| model4 | MoE λ>0 | o mesmo emulador LRU; o treino é que usa λ>0, também na semente 0 |

Não há segunda política de eviction. O modelo 2 é a fatia de relatório: hold-out no `B` de referência e a curva de dose, contra o denso e contra λ>0. Os números do modelo 2 são cópia do modelo 1 (`metrics_shared_with`). Se o JSON do hold-out trouxer a parede (`wall_p50_ms`, `wall_p95_ms`, `wall_clock` e o restante do bloco), a tabela copia esses campos; em CPU eles ficam null e a perplexidade continua válida. Célula `pending` não é número.

O hold-out entra se `outputs/holdout/summary.json` existir (ou `--holdout`). `dose_response: true` nesse resumo significa que cada corrida já traz o bloco `dose_response` (o denso é `n/a`). O arquivo irmão é `outputs/holdout/dose_response.json`, protocolo `wikitext-2-dose-response`, residência `lru_reset_per_split_then_set_hot_budget`, método `reset_lru_at_yaml_b_then_set_hot_budget`. O teto de cada `multiplier` tem de bater com `MemoryConfig.scaled_hot_slots`. Este script não chama `set_hot_budget` e não retreina.

```bash
python3 experiments/run_model_comparison.py --manifest-only
python3 experiments/run_model_comparison.py --holdout outputs/holdout/summary.json
```

A saída padrão é `outputs/comparison/comparison.json` (gitignored). `metrics_complete` só fica verdadeiro quando o hold-out de referência e a curva de dose preencheram a tabela. Enquanto isso, o JSON não é número de relatório.

Não colocar saída grande nesta pasta. Métrica pequena vai para `results/`; checkpoint e dataset ficam de fora do Git. O home frio do MoE cai em `outputs/` (gitignored). O baseline denso não grava memmap de expert.
