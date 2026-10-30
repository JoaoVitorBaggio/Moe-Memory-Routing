# Referências

Obras-âncora do blueprint. Não há número de desempenho deste projeto.

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

Fechados: TinyStories no dev / WikiText-2 na entrega; `k=2`; `B=25%` de `E`; `λ ∈ {0, 0,5, 1}` especializado; `β` Switch-like; adversário LRU; denso = tronco+`k`·expert; VRAM útil 5,5 GB; RAM trabalho 10 GB; SSD frio 1–2 GB. Também fechados: 3 seeds; hold-out ~10%; latência de parede na 1060; orçamento variável no treino adiado. Proposta de partida de `c`/`w` registrada em [funcao-de-custo.md](funcao-de-custo.md) (HBM=1, DDR=10, NVMe=20|30; `w` = Δ`c`); ainda abertos (código): fechar NVMe 20 vs 30 e `β` no `configs/`; arquitetura fina (camadas, `E`∈{16,32}, tamanho do expert); granularidade do emulador (token/batch/camada); forma do proxy com `k=2` (média vs soma); se o roteador recebe `ℓ(e)` como entrada. Emulação + dose–resposta + adversário LRU são decisões deste blueprint (out/2026), não dos papers acima.
