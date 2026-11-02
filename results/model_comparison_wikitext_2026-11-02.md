# Comparação dos quatro braços — WikiText-2, semente 0 (2026-11-02)

**Status:** `comparison.json` preenchido em 2026-11-02 a partir do hold-out e da dose–resposta. `metrics_complete` verdadeiro. Nenhuma linha com `status` `pending`. Host: Fedora, GTX 1060, America/Sao_Paulo.

Fonte local (gitignored, fora deste commit): `outputs/comparison/comparison.json`. `run_model_comparison.py` não treina e não mede (`produces_measurements` falso, `evaluated_by_this_module` falso). Este commit não reexecuta o script.

## Braços

O JSON marca `model1_equals_model2` verdadeiro. Os dois usam `MemoryMoEModel`, `ResidencyEmulator` (LRU) e λ = 0.0, no checkpoint `lambda-0` da semente 0. Não há segunda política de eviction. O modelo 2 é a fatia de relatório (`metrics_shared_with` = model1): os campos de métrica repetem o modelo 1 em cada célula (perplexidade, `nll_nats`, `swaps`, `bytes_moved`, parede, `status`). O denso é `DenseLanguageModel`, não o MoE com λ = 0. O modelo 4 reusa o mesmo emulador; o que muda é o λ do treino (0.5 e 1.0).

Os caminhos gravados são o layout antigo: `outputs/wikitext/lambda-<λ>/checkpoint.pt` e `outputs/baseline/wikitext/checkpoint.pt`. `checkpoint_seed` 0.

| id | rótulo | família | λ | de onde a célula sai |
| --- | --- | --- | --- | --- |
| model1 | MoE top-k λ=0 | lambda-0 | 0.0 | hold-out no 1×B; dose em 0,5× e 2× |
| model2 | MoE + LRU | lambda-0 | 0.0 | cópia do model1 |
| model3 | Denso | baseline | — | hold-out no 1×B; 0,5× e 2× são `not_applicable` |
| model4 | MoE λ>0 | lambda-0.5 e lambda-1 | 0.5 e 1.0 | hold-out no 1×B; dose em 0,5× e 2× |

30 linhas: 2 splits × 3 multiplicadores × (model1, model2, model3, λ=0.5, λ=1.0). 26 `filled`. 4 `not_applicable`: o denso em `valid` e `test`, nos multiplicadores 0.5 e 2.0 (`perplexity` null, `source` none). A nota do schema ainda diz que célula `pending` não é número; nesta corrida não sobrou nenhuma.

No 1×B a perplexidade copiada (2 casas) é a do hold-out:

| braço | família | valid | test |
| --- | --- | --- | --- |
| model1 e model2 | lambda-0 | 349.22 | 351.88 |
| model3 | baseline | 333.48 | 339.05 |
| model4 | lambda-0.5 | 333.99 | 337.37 |
| model4 | lambda-1 | 326.59 | 330.18 |

Nas linhas MoE de 1×B, `filled_from` é `holdout+dose_response`. No denso de 1×B, `filled_from` é `holdout`. Nas pontas 0,5× e 2× do MoE, `filled_from` é `dose_response`. A PPL dessas pontas está em [dose_response_wikitext_2026-11-02.md](dose_response_wikitext_2026-11-02.md). Parede, `swaps` e bytes do `B` de referência estão em [holdout_wikitext_2026-11-02.md](holdout_wikitext_2026-11-02.md).

Nas células `filled`, `wall_report_grade` é verdadeiro, `wall_clock` é cuda, `wall_gpu` é NVIDIA GeForce GTX 1060 6GB e `wall_forward_batch` é 4.
