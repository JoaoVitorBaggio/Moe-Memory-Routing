# Protocolo experimental

Os números medidos no WikiText-2 estão em [`results/`](../results/README.md). O índice dessa pasta lista o que já tem número e o que ainda falta. Onde faltar número, o texto diz **a definir**.

Discussão de 2026-11-04: há dois regimes. Um modelo que cabe em `B` não se compara, em troca, com um modelo que não cabe. O denso já medido não tem teto quente; fica como teto de qualidade. O par justo de quem cabe continua sem número.

## Dois regimes

`B` de referência continua **25% de `E`** (**8/32**), salvo nota explícita. A métrica principal muda com o regime.

### Cabe

A pegada residente cabe em `B`: os experts residentes do MoE, ou o denso inteiro, ≤ `B`.

Esperado: `swaps` = 0. No denso, que não tem emulador, `swaps` e `bytes_moved` ficam null. As colunas de I/O ficam em zero ou null. Se o MoE reportar `swaps` > 0, a corrida saiu deste regime.

Métrica principal: PPL de hold-out, mais parede p50/p95.

### Não cabe

A capacidade passa de `B`, então o LRU do emulador troca. É o MoE com `E = 32` sob o `B` de referência (8 vagas) e sob os multiplicadores em que a curva ainda troca.

Métrica principal: PPL contra `swaps`, `bytes_moved` e parede. Aqui `λ > 0` / `C_mem` tem o que cortar em relação a `λ = 0`, no mesmo `B`.

O hold-out e a dose de MoE já publicados são este regime. O denso dessas notas fica de fora da comparação de troca.

## Adversário justo em cabe

Primário: denso com parâmetros *ativos* ≈ tronco + `k`·tamanho_do_expert (`k = 2`), não o total dos `E` experts. O treino e a avaliação usam o mesmo orçamento quente do MoE que cabe. `swaps` ausentes (null).

MoE que cabe: a mesma família (`E`, `k`, camadas). `B`, ou a residência, grandes o bastante para nenhum expert ficar frio na avaliação. Um destes caminhos basta:

- `B = E` (32/32);
- `B` ≥ experts distintos usados no passo, por camada;
- residência toda quente.

A corrida só conta se `swaps` e `bytes_moved` forem reportados e forem 0. Com `swaps` > 0, o regime é **não cabe**.

O `ResidencyEmulator` de hoje exige `warm_slots ≥ 1` e quente + morno ≤ `E`. Com o YAML (`warm_slots: 8`, `E: 32`) o maior `hot_slots` que `set_hot_budget` aceita é 24. `B = E` e a residência toda quente não passam sem outro arranjo do morno. Nenhum dos três caminhos está ligado como corrida. Como chegar no zero reportado fica **a definir**. Este texto não abre config.

O `DenseLanguageModel` já em [`results/`](../results/README.md) (swaps e bytes null, dose `n/a`) permanece **teto de qualidade**, em tabela à parte. O par justo — denso desse tamanho contra o MoE que cabe, os dois no mesmo orçamento residente, swaps 0 no MoE — está **a definir**. Não há número desse par.

## Adversário justo em não cabe

Ficam os modelos 1 e 2 (`λ = 0` + LRU) e o modelo 4 (`λ > 0`) nos mesmos multiplicadores **0,5× / 1× / 2×** de `B`.

O denso sem teto quente não entra como adversário de troca. A comparação de roteamento e de `C_mem` sob `B` é 1/2 contra 4.

Esse eixo no WikiText-2 está **já medido** (hold-out e dose em `results/`). Essa tabela não substitui o braço **cabe**.

## Experimento mínimo

1. Emulador único VRAM/RAM/SSD com política de colocação dinâmica.
2. Uma tabela de custo (perfil consumer ~6 GB / 5,5 GB úteis; HBM ≫ DDR ≫ NVMe).
3. Modelo MoE na faixa ~10–50 M; **`E = 32`**; **`k = 2`**; se faltar capacidade, aumentar o expert.
4. Orçamento de referência **`B = 25%` de `E`** (**8/32**). Treino **especializado** por `λ ∈ {0, 0,5, 1}`; `β` fixo pequeno (ordem Switch). Emulador **por batch**; roteador com **`ℓ(e)`**.
5. Avaliação em `0,5×B`, `B` e `2×B` no hold-out. Esse eixo é o regime **não cabe**.
6. Baselines no mesmo corpus. Em **não cabe**, o mesmo `B`: top-k (`λ=0`) e **LRU de experts (único adversário de offload — não mudar no meio)** contra o modelo com penalidade. Em **cabe**, o denso com ativos ≈ tronco + `k`·expert contra o MoE que cabe. O denso já medido, sem teto quente, é o teto de qualidade.
7. Relatar correlação proxy↔emulador e entropia do roteador.

## O que fica igual entre modelos

- Corpus: TinyStories no desenvolvimento; **WikiText-2** nas corridas de entrega (PC à noite). Mesmo pipeline, config diferente.
- Família e tamanho do LM; número de camadas MoE. O denso que cabe usa a mesma família, com FFN de largura `k`·expert.
- Implementação PyTorch compartilhada o máximo possível.
- Tabela de custos da corrida mínima (salvo ablação explícita).

O denso iguala parâmetros *ativos* ≈ tronco + `k`·tamanho_do_expert (`k=2`), não o total de experts. A conta dimensiona o adversário de **cabe**. O denso continua sem emulador: a coluna de troca dele é null, e ela não se compara com o `swaps` > 0 do MoE que não cabe.

## Os quatro modelos

| # | Nome | Treino | Regime | Papel | Estado |
| --- | --- | --- | --- | --- | --- |
| 1 | MoE top-k | `λ = 0` + balanço | não cabe | residência na avaliação: **mesmo LRU sob `B`** (não treina com `C_mem`); o `λ=0` da grade **é** este modelo | **já medido** (WikiText-2) |
| 2 | MoE + **LRU de experts** | igual ao 1 | não cabe | único adversário de offload, mesmo `B` e mesmos multiplicadores | **já medido**: mesma linha que o 1 |
| 3 | Denso que cabe | sem roteador | cabe | ativos ≈ tronco + `k`·expert; mesmo orçamento residente que o MoE que cabe; swaps ausentes | fórmula fechada; par **a definir** |
| 4 | Este projeto | `λ > 0` + balanço | não cabe | emulador + colocação dinâmica, mesmo `B` que 1 e 2 | **já medido** (`λ ∈ {0,5, 1}`) |

O modelo 2 existe para mostrar que o ganho **não** é “qualquer MoE” nem “qualquer offload”.

O MoE que cabe não ganha um quinto número na tabela. É a mesma família, com residência que deixa `swaps` = 0. Estado: **a definir**.

No código de hoje o modelo 1 e o modelo 2 são o mesmo caminho. `ResidencyEmulator` já é LRU; não há segunda classe de eviction, e o denso não é o MoE com `λ=0`. O modelo 2 é a fatia de relatório desse adversário: o hold-out no `B` de referência e a dose (`0,5×` / `1×` / `2×`, `set_hot_budget` depois do LRU fresco) quando o JSON da curva existir. O script ainda grava a linha do denso ao lado; na leitura deste protocolo essa linha é o teto, e o confronto de **não cabe** é o modelo 4. O empacotamento está em `experiments/run_model_comparison.py`. Ele não treina de novo e não mede sozinho: copia o checkpoint da semente 0 (`seed-0/lambda-0`, ou o layout antigo se for o único arquivo) para as duas linhas. Célula `pending` não é número. A corrida de 2026-11-02 preencheu essa tabela, denso inclusive. O preenchimento vale como teto mais MoE que não cabe. Não fecha o regime **cabe**.

## Dose–resposta (ponto 6 do feedback)

Instrumento do regime **não cabe**.

- Eixo de dose na **avaliação:** orçamento `B` ∈ {0,5×, 1×, 2×}.
- `λ` ∈ {0, 0,5, 1} são **curvas de treino** separadas (não confundir 0,5 de `λ` com 0,5×`B`).
- Entrega: **uma figura** dose–resposta (qualidade vs latência ou vs I/O/custo medido) e **uma tabela** dos modelos 1/2 contra o 4 no orçamento de referência (e, se couber, nos extremos).
- Interpretação: resposta monótona e útil ⇒ o sinal de memória está vivo; flat ou invertido ⇒ revisar proxy/política.
- Um ponto da curva com `swaps` = 0 marca a fronteira em que o MoE passou a caber. O braço **cabe** pede, além disso, o denso no mesmo orçamento. O denso sem teto segue `n/a` na dose.

## Métricas (hold-out, escolha discreta)

| Métrica | Cabe | Não cabe |
| --- | --- | --- |
| Perplexidade | Principal | Principal, ao lado do I/O |
| Bytes / trocas | 0, ou null no denso; `swaps` > 0 tira a corrida do regime | Principal |
| Latência p50/p95 | Principal, com a PPL | Principal, com a PPL e o I/O |
| Entropia do roteador | Diagnóstico de colapso | Diagnóstico de colapso |
| Correlação proxy × emulador | I/O parado; não decide o braço | Diagnóstico de alinhamento |

Seeds: **3** no mínimo. Hold-out: no WikiText-2, **splits oficiais** valid (seleção) e test (relatório); no TinyStories, o valid do dataset. A ordem de grandeza é ~10% do treino.

## O que o texto pode afirmar

**Cabe.** Com o mesmo orçamento residente e `swaps` = 0, o MoE roteado ganha ou empata o denso que cabe, em PPL? A parede p50/p95 vai ao lado.

**Não cabe.** Com `B` fixo e `swaps` > 0, `λ > 0` corta swaps, bytes e parede em relação a `λ = 0`, sem derrubar a PPL?

O WikiText-2 já em `results/` é o teto (denso sem teto quente, swaps e bytes null, dose `n/a`) mais o MoE que **não cabe** (modelos 1/2/4). O braço justo de **cabe** continua **a definir**. Não há número desse braço.

## Script de perplexidade

`experiments/run_holdout_eval.py` calcula a perplexidade de token do WikiText-2 em `valid` e `test` a partir dos checkpoints já treinados. Os resumos publicados estão em `results/`. O JSON bruto fica em `outputs/` (gitignored). A corrida de relatório é na GTX 1060. A loss de treino da semente 0 não entra nesta tabela.

O que o script fecha: a mesma tokenização e os mesmos blocos não sobrepostos do treino (`seq_len` do checkpoint; 256 na entrega), a CE `task_cross_entropy`, a classe densa no baseline e a classe MoE nos três λ (inclusive 0). O LRU do MoE recomeça em cada split, no `B` de referência, um passo por batch do checkpoint. `--batch-size` só corta o forward. Esse caminho mede o regime **não cabe** no MoE e o teto no denso. O MoE que cabe não tem corrida aqui.

O treino da grade e do baseline denso percorre `eval.seeds` (sementes `0 .. N-1`, partida 3). O hold-out automático lê a semente 0.

A dose–resposta entra no mesmo script, sem retreino: depois da perplexidade no `B` do YAML, cada checkpoint MoE dessa descoberta (ou o arquivo pedido na linha de comando) repete `valid` e `test` em `eval.B_mult` (`0,5×`, `1×`, `2×`). O teto é só `prepare_split_budget`: cada split recomeça o LRU nesse `B` e `set_hot_budget` aplica o multiplicador sem apagar o carimbo dessa reatribuição. O denso não tem teto quente: a curva dele é `n/a`. A tabela é `dose_response.md`, ao lado de `dose_response.json`. O resumo marca `dose_response: true` quando a curva de algum MoE foi medida.

Cada split, no `B` de referência e em cada ponto de `eval.B_mult`, grava `swaps`, `bytes_moved` e a parede do forward (`wall_p50_ms`, `wall_p95_ms`). A parede de relatório exige a **GTX 1060 com CUDA**, o batch do checkpoint e o split inteiro (`wall_report_grade`). Em CPU `wall_clock` fica `unavailable` e os milissegundos ficam null.

O modelo 2 não é uma corrida a mais: é o rótulo do mesmo `λ=0` da semente 0, empacotado por `experiments/run_model_comparison.py`. Agregar as outras sementes continua fora. A correlação proxy×emulador e a curva de entropia não treinam: `experiments/plot_proxy_emulator.py` lê o `train.jsonl` em `outputs/wikitext/seed-*/lambda-*/` e, no layout antigo da semente 0, em `outputs/wikitext/lambda-*/`. O resumo numérico está em `results/`. `smoke_eval.py` não é este número.

## Latência de parede

Decidido: reportar latência de parede na GTX 1060 no protocolo mínimo, porque `B` é fixo (não randomizado no treino). Em **não cabe** o emulador segue com bytes e trocas; a parede é coluna adicional. Em **cabe** a parede entra com a PPL, e o I/O fica em zero ou null.

O caminho físico (quente na VRAM, morno em CPU pinned, frio em memmap) está em `src/moe_memory/residency.py`. O cronômetro está em [plano-de-hardware.md](plano-de-hardware.md). O hold-out grava p50/p95, `swaps` e `bytes_moved` no JSON, também em cada ponto da curva de `B`. Os números publicados estão em `results/`. CPU deixa os ms null.

## Ablações úteis (depois do mínimo)

- `λ = 0` vs `λ > 0` no modelo 4, no regime **não cabe**.
- O par **cabe** (denso que cabe e MoE que cabe), quando a corrida existir.
- Trocar a tabela de custo (apêndice mercado).
- Orçamento amostrado / variável no treino: **adiado** (fora da entrega mínima; extensão).
