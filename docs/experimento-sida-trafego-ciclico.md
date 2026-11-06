# Plano: SiDA em tráfego cíclico

Nota de plano. A corrida não rodou. Não há acurácia, latência, vazão nem overhead medidos aqui.

Isto não muda a pergunta do experimento atual: custo de memória no roteador, λ, WikiText-2 e os quatro braços. O que está em [`definicao.md`](definicao.md) e [`protocolo-experimental.md`](protocolo-experimental.md) continua valendo.

A pergunta desta nota é outra: o ganho do SiDA some quando o tráfego é previsível, por exemplo acessos que se repetem em ciclo? Se o preditor acerta quase tudo, o que sobra é só o overhead do mecanismo.

## Carga

Dois traços, o mesmo saldo de ativação (o mesmo número de experts ativos por token):

- **Cíclico.** Uma sequência de experts que se repete. O mesmo conjunto volta a cada período. É o tráfego previsível.
- **Contraste.** O tráfego do MoE atual, pouco enviesado: o roteamento top-k do experimento de WikiText-2, sem ciclo imposto. Serve para ver se o efeito aparece só quando a repetição some.

O traço cíclico é de experts, não um corpus novo. O WikiText-2 do experimento atual não é reescrito.

## O que medir

Contra um Switch sem o preditor, no mesmo traço:

- acurácia do preditor (acerto do expert previsto);
- latência ou vazão de ponta a ponta;
- overhead do prefetch (o custo do mecanismo, separado do ganho de ponta a ponta).

Nada disso foi medido.

## O que conta

**Útil.** No traço cíclico, a latência ou a vazão de ponta a ponta ainda melhora em relação ao Switch sem o preditor, e essa diferença passa do overhead do prefetch.

**Não útil.** O ganho some. O preditor pode acertar quase tudo e, mesmo assim, a ponta a ponta não fica melhor que o Switch sem ele: o que resta é o overhead do mecanismo.
