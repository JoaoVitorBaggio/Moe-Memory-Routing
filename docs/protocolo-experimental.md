# Protocolo experimental

Nada disto foi executado. Onde faltar número, o texto diz **a definir**.

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

## Latência de parede

Decidido: reportar latência de parede na GTX 1060 no protocolo mínimo, porque `B` é fixo (não randomizado no treino). O emulador continua com bytes/trocas; a parede é coluna adicional.

O caminho físico (quente na VRAM, morno em CPU pinned, frio em memmap) está em `src/moe_memory/residency.py`. O cronômetro está em [plano-de-hardware.md](plano-de-hardware.md). Não há número medido neste repositório.

## Ablações úteis (depois do mínimo)

- `λ = 0` vs `λ > 0` no modelo 4.
- Trocar a tabela de custo (apêndice mercado).
- Orçamento amostrado / variável no treino: **adiado** (fora da entrega mínima; extensão).
