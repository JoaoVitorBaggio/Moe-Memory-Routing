# Referências

Obras-âncora do blueprint. Esta página não traz métrica. Os números medidos estão em [`results/`](../results/README.md).

## Shazeer et al., 2017

Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR 2017.  
https://arxiv.org/abs/1701.06538

Base do MoE com roteamento esparso (modelo 1 e arquitetura esparsa).

## Fedus et al., 2022

Fedus et al. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR, 2022.  
https://jmlr.org/papers/v23/21-0998.html

Loss de balanceamento adaptada em [funcao-de-custo.md](funcao-de-custo.md).

## Alizadeh et al.

Alizadeh et al. *LLM in a flash: Efficient Large Language Model Inference with Limited Memory.* arXiv:2312.11514.  
https://arxiv.org/abs/2312.11514

Contexto de offload para nível frio; não é o sistema reproduzido. Diferença: aqui o custo entra no *treino* do roteador.

## Song et al.

Song et al. *PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU.* arXiv:2312.12456.  
https://arxiv.org/abs/2312.12456

Motivação de serving em GPU de consumidor; baselines deste trabalho são os quatro modelos do protocolo, em escala de estudante.

## O que estas obras não fecham

Fechados: TinyStories no dev / WikiText-2 na entrega; `k=2`; `B=25%` de `E`; `λ ∈ {0, 0,5, 1}` especializado; `β` Switch-like; adversário LRU; denso = tronco+`k`·expert; VRAM útil 5,5 GB; RAM trabalho 10 GB; SSD frio 1–2 GB. Também fechados: 3 seeds; hold-out ~10%; latência de parede na 1060; orçamento variável no treino adiado. Fechados na call 30/10/2026: `E=32`; `c`/`w` de partida (HBM=1, DDR=10, NVMe=20|30; `w=Δc` amarrado a `c`); `β` ordem Switch; proxy `k=2` (média ponderada no treino, soma na inferência); emulador **por batch**; roteador recebe `ℓ(e)`. Detalhe em [funcao-de-custo.md](funcao-de-custo.md). Partida numérica e arquitetura fina (`d_model`, camadas, cabeças, `d_ff`, contexto, `nvme: 20`, `β: 0.01`) em [`configs/default.yaml`](../configs/default.yaml); recalibrar `c` na 1060 se precisar. Emulação + dose–resposta + adversário LRU são decisões deste blueprint (out/2026), não dos papers acima.
