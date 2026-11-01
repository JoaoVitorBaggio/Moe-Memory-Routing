# Protocolo experimental

A loss de treino da semente 0 no WikiText-2 está em [`results/`](../results/README.md). O resto deste protocolo (hold-out, sementes 1 e 2, dose–resposta, parede, modelo 2 como linha de relatório, proxy×emulador) não tem número medido neste repositório. Onde faltar número, o texto diz **a definir**.

## Experimento mínimo

1. Emulador único VRAM/RAM/SSD com política de colocação dinâmica.
2. Uma tabela de custo (perfil consumer ~6 GB / 5,5 GB úteis; HBM ≫ DDR ≫ NVMe).
3. Modelo MoE na faixa ~10–50 M; **`E = 32`**; **`k = 2`**; se faltar capacidade, aumentar o expert.
4. Orçamento de referência **`B = 25%` de `E`** (**8/32**). Treino **especializado** por `λ ∈ {0, 0,5, 1}`; `β` fixo pequeno (ordem Switch). Emulador **por batch**; roteador com **`ℓ(e)`**.
5. Avaliação em `0,5×B`, `B` e `2×B` no hold-out.
6. Baselines no mesmo corpus e mesmo `B` de referência: top-k (`λ=0`), **LRU de experts (único adversário de offload — não mudar no meio)**, denso com ativos ≈ tronco + `k`·expert, e o modelo com penalidade.
7. Relatar correlação proxy↔emulador e entropia do roteador.

## O que fica igual entre modelos

- Corpus: TinyStories no desenvolvimento; **WikiText-2** nas corridas de entrega (PC à noite). Mesmo pipeline, config diferente.
- Família e tamanho do LM; número de camadas MoE.
- Implementação PyTorch compartilhada o máximo possível.
- Tabela de custos da corrida mínima (salvo ablação explícita).

O denso iguala parâmetros *ativos* ≈ tronco + `k`·tamanho_do_expert (`k=2`), não o total de experts.

## Os quatro modelos

| # | Nome | Treino | Residência |
| --- | --- | --- | --- |
| 1 | MoE top-k | `λ = 0` + balanço | residência na avaliação: **mesmo LRU sob `B`** (só não treina com `C_mem`); o `λ=0` da grade **é** este modelo |
| 2 | MoE + **LRU de experts** | igual ao 1 | único adversário de offload |
| 3 | Denso | sem roteador | ativos ≈ tronco + `k`·expert; cabe no orçamento quente |
| 4 | Este projeto | `λ > 0` + balanço | emulador + colocação dinâmica |

O modelo 2 existe para mostrar que o ganho **não** é “qualquer MoE” nem “qualquer offload”.

No código de hoje o modelo 1 e o modelo 2 são o mesmo caminho. `ResidencyEmulator` já é LRU; não há segunda classe de eviction, e o denso não é o MoE com `λ=0`. O modelo 2 é a fatia de relatório desse adversário: o hold-out no `B` de referência e a dose (`0,5×` / `1×` / `2×`, `set_hot_budget` depois do LRU fresco) quando o JSON da curva existir, confrontados com o modelo 3 e com o modelo 4. O empacotamento está em `experiments/run_model_comparison.py`. Ele não treina de novo e não mede sozinho: copia o checkpoint da semente 0 (`seed-0/lambda-0`, ou o layout antigo se for o único arquivo) para as duas linhas. Célula `pending` não é número.

## Dose–resposta (ponto 6 do feedback)

- Eixo de dose na **avaliação:** orçamento `B` ∈ {0,5×, 1×, 2×}.
- `λ` ∈ {0, 0,5, 1} são **curvas de treino** separadas (não confundir 0,5 de `λ` com 0,5×`B`).
- Entrega: **uma figura** dose–resposta (qualidade vs latência ou vs I/O/custo medido) e **uma tabela** contra os baselines no orçamento de referência (e, se couber, nos extremos).
- Interpretação: resposta monótona e útil ⇒ o sinal de memória está vivo; flat ou invertido ⇒ revisar proxy/política.

## Métricas (hold-out, escolha discreta)

| Métrica | Nota |
| --- | --- |
| Perplexidade | Principal de qualidade |
| Bytes / trocas do emulador | Principal de recurso no mínimo |
| Latência p50/p95 | **Sim — medir parede na 1060** (orçamento `B` fixo, não sorteado no treino mínimo) |
| Entropia do roteador | Diagnóstico de colapso |
| Correlação proxy × emulador | Diagnóstico de alinhamento |

Seeds: **3** no mínimo. Hold-out: no WikiText-2, **splits oficiais** valid (seleção) e test (relatório); no TinyStories, o valid do dataset. A ordem de grandeza é ~10% do treino.

## Script de perplexidade

`experiments/run_holdout_eval.py` calcula a perplexidade de token do WikiText-2 em `valid` e `test` a partir dos checkpoints já treinados. Não há perplexidade de hold-out medida neste repositório: a corrida é na GTX 1060, fora daqui. A loss de treino da semente 0 não entra nesta tabela.

O que o script fecha: a mesma tokenização e os mesmos blocos não sobrepostos do treino (`seq_len` do checkpoint; 256 na entrega), a CE `task_cross_entropy`, a classe densa no baseline e a classe MoE nos três λ (inclusive 0). O LRU do MoE recomeça em cada split, no `B` de referência, um passo por batch do checkpoint. `--batch-size` só corta o forward.

O treino da grade e do baseline denso percorre `eval.seeds` (sementes `0 .. N-1`, partida 3). O hold-out automático lê a semente 0.

A dose–resposta entra no mesmo script, sem retreino: depois da perplexidade no `B` do YAML, cada checkpoint MoE dessa descoberta (ou o arquivo pedido na linha de comando) repete `valid` e `test` em `eval.B_mult` (`0,5×`, `1×`, `2×`). O teto é só `prepare_split_budget`: cada split recomeça o LRU nesse `B` e `set_hot_budget` aplica o multiplicador sem apagar o carimbo dessa reatribuição. O denso não tem teto quente: a curva dele é `n/a`. A tabela é `dose_response.md`, ao lado de `dose_response.json`. O resumo marca `dose_response: true` quando a curva de algum MoE foi medida.

Cada split, no `B` de referência e em cada ponto de `eval.B_mult`, grava `swaps`, `bytes_moved` e a parede do forward (`wall_p50_ms`, `wall_p95_ms`). A parede de relatório exige a **GTX 1060 com CUDA**, o batch do checkpoint e o split inteiro (`wall_report_grade`). Em CPU `wall_clock` fica `unavailable` e os milissegundos ficam null. Não há bytes, trocas nem parede medidos neste repositório.

O modelo 2 não é uma corrida a mais: é o rótulo do mesmo `λ=0` da semente 0, empacotado por `experiments/run_model_comparison.py`. Agregar as outras sementes continua fora. A correlação proxy×emulador e a curva de entropia não treinam: `experiments/plot_proxy_emulator.py` lê o `train.jsonl` em `outputs/wikitext/seed-*/lambda-*/` e, no layout antigo da semente 0, em `outputs/wikitext/lambda-*/`. Não há correlação proxy×emulador nem curva de entropia medida neste repositório. `smoke_eval.py` não é este número.

## Latência de parede

Decidido: reportar latência de parede na GTX 1060 no protocolo mínimo, porque `B` é fixo (não randomizado no treino). O emulador continua com bytes/trocas; a parede é coluna adicional.

O caminho físico (quente na VRAM, morno em CPU pinned, frio em memmap) está em `src/moe_memory/residency.py`. O cronômetro está em [plano-de-hardware.md](plano-de-hardware.md). O hold-out grava p50/p95, `swaps` e `bytes_moved` no JSON, também em cada ponto da curva de `B`. Não há latência de parede medida neste repositório: a coluna de relatório sai na GTX 1060. CPU deixa os ms null.

## Ablações úteis (depois do mínimo)

- `λ = 0` vs `λ > 0` no modelo 4.
- Trocar a tabela de custo (apêndice mercado).
- Orçamento amostrado / variável no treino: **adiado** (fora da entrega mínima; extensão).
