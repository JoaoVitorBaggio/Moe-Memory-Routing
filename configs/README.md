# configs/

Hiperparâmetros versionados do experimento mínimo.

| Arquivo | Papel |
| --- | --- |
| [`default.yaml`](default.yaml) | Partida fechada na call 30/10/2026 (e blueprint) |

## Como ler

- `model.*` — arquitetura fina: `d_model`, `n_layers` (todo FFN é MoE), `n_heads` (`d_head = 64`), `expert_hidden` (`d_ff`; o comentário no YAML aponta 768 se faltar capacidade), `max_seq_len` (256, knob até 512), tokenizer GPT-2, fp32. `tie_embeddings` amarra o `lm_head` para o total ficar na faixa 10–50 M. O LM denso não tem chave própria: a largura do FFN é `k * expert_hidden` (ativos ≈ tronco + `k`·expert). A corrida é `experiments/run_baseline.py` e não usa `loss.lambda_grid`.
- `memory.warm_slots` — vagas do nível morno para os três níveis existirem. Não é eixo do protocolo; o teto que se varre é `B` no quente.
- `c.*` — custo de **acesso** por nível. `w(a,b)` **não** aparece no YAML: é `|c(b)−c(a)|` e muda com `c`.
- `loss.beta` — fixo pequeno (ordem Switch); valor `0.01` é partida, não busca.
- `loss.lambda_grid` — treinos separados; **não** é Optuna/HPO no mínimo. A corrida overnight que percorre essa lista está em [`experiments/README.md`](../experiments/README.md). Passo, batch e época não entram neste YAML: são knobs operacionais do script.
- `loss.train_topk_agg` / `infer_topk_agg` — média ponderada no treino, soma na inferência. É o proxy de custo, não a mistura do hidden.
- `eval.B_mult` — multiplicadores do teto quente na avaliação. `experiments/run_holdout_eval.py` mede a perplexidade no `B` de referência (valid e test, semente 0) e, sem retreinar, a curva `B_mult` nos checkpoints MoE. O denso fica `n/a`. `experiments/run_model_comparison.py` preenche a célula quando o resumo traz `dose_response: true` ou existe `outputs/holdout/dose_response.json`.
- `eval.seeds` — quantas sementes a grade de λ e o baseline denso lançam (`0 .. seeds-1`). Partida: 3. O hold-out automático e a comparação do modelo 2 leem a semente 0 e não agregam as outras.
- `nvme: 20` — escolhido entre 20 e 30; recalibrar na 1060 se a dose–resposta sair flat/invertida.
- Granularidade do emulador e `router_sees_level` estão aqui para o código não divergir do doc.
- `corpus.active` — `dev` ou `report`. O nome em si (`TinyStories`, `WikiText-2`) também vale na API.
- `corpus.cache_dir` — cache do Hugging Face. `corpus.paths` vazio baixa TinyStories (`roneneldan/TinyStories`) e o WikiText-2 cru (`Salesforce/wikitext`, config `wikitext-2-raw-v1`). Um diretório local tem `train.txt` / `valid.txt` / `test.txt`.
- `smoke` — tamanho do batch do passo curto e um `λ` da grade. Não é a corrida de entrega. O passo de TinyStories reusa esse tamanho.

Fórmulas e contexto: [docs/funcao-de-custo.md](../docs/funcao-de-custo.md).
