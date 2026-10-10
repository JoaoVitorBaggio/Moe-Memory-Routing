# src/

Código do modelo. **Em construção:** ainda não há módulo.

Quando existir, esta pasta guarda a rede de linguagem, a camada MoE, o roteador, a residência nos três níveis (VRAM, RAM pinned, arquivo mmap) e a loss `L_tarefa + λ · C_mem + β · L_balanço`. A definição da loss está em [`docs/funcao-de-custo.md`](../docs/funcao-de-custo.md).

Tudo em PyTorch. Sem TensorFlow.
