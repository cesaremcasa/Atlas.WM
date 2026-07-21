# Atlas.WM - roteiro canônico da versão final

Este roteiro define a versão final do artigo vivo do Atlas.WM. Ele não é um resumo de
resultados nem um paper escrito depois do produto: organiza, desde a primeira decisão de
arquitetura até a release, o corpus de pesquisa, o produto executável, as falhas, as
retratações e as próximas validações.

## Posição editorial

O Atlas.WM não deve ser apresentado como “um modelo que resolveu identificação de
física”. É um programa de pesquisa em modelos de mundo estruturados que responde a uma
pergunta mais exigente:

> **Que condições tornam uma conclusão sobre identificabilidade física auditável - e
> quais partes pertencem à arquitetura, à evidência, à coleta de dados e ao protocolo de
> avaliação?**

A contribuição do laboratório é o método de construção e correção de um artefato de
pesquisa: arquitetura explícita, contratos físicos, dados reproduzíveis, testes, ledger
de claims e retratações públicas quando a evidência não sobrevive à revisão.

## Regra de verdade

| Camada | Fonte primária | Uso no artigo |
|---|---|---|
| Origem da arquitetura | `docs/Atlas-WM-v3-Architecture-Plan.md` | decisões, promessas e fronteiras iniciais |
| Primeira falsificação | `docs/v2.0-TECHNICAL-POSTMORTEM.md` e `docs/v2.0-COMPLETION-REPORT.md` | por que desempenho aparente não era validação |
| Reconstrução | `docs/v4.0-ROADMAP.md`, `CHANGELOG.md` | red-team, correções e blocos v4 |
| Evidência detalhada | `docs/RESULTS.md`, `docs/MODEL_CARD.md`, scripts, configs, dados e testes | claims, métricas, limites e proveniência |
| Produto público | release `v4.0.0`, README, pacote e checkpoints | o que um leitor pode executar e inspecionar |
| Evidência posterior | `AI Search/Research/2026-07-21-atlas-reproduction-report.md` | resultados atuais, divergências e status de validação |

Quando duas fontes divergem, o artigo mostra a divergência e a data; não escolhe a mais
favorável silenciosamente.

## Arquitetura narrativa da versão final

### 0. Frente do artigo - Atlas como artefato de laboratório

**Título de trabalho:** *Atlas.WM: Building an Auditable Structured World Model Through
Architecture, Retraction, and Reproduction.*

Abrir com uma frase curta sobre o objeto: um sistema de pesquisa pequeno, executável e
inspecionável para estudar quando parâmetros físicos podem ser inferidos de observações.
Incluir versão do artefato, release pública, licença, commit/release, autores e estado
da reprodução mais recente.

### 1. A pergunta antes do modelo

Apresentar o problema de modelos de mundo e identificação física sem prometer robótica
real. Formular a hipótese de trabalho: observabilidade, desenho de features, coleta e
protocolo de avaliação podem determinar o resultado mais do que escala de modelo.

**Não fazer:** abrir com números, benchmarks ou linguagem de “SOTA”.

### 2. A primeira arquitetura - o que foi construído e por quê

Partir do plano v3 e mostrar a decomposição latente: imutável, lenta, dinâmica e
controlável. Explicar decisões de engenharia como hipóteses testáveis: action routing,
passthrough imutável, checkpoints seguros, lockfile, contratos físicos, canário de
determinismo e exportação.

**Pergunta do capítulo:** que propriedades podem ser garantidas pela arquitetura e quais
apenas precisam ser medidas?

### 3. O primeiro fracasso - quando o treino “perfeito” não significava ciência

Usar o postmortem v2 como ponto de virada: validação suspeitamente perfeita,
memorização e ambiente insuficiente. Este capítulo estabelece a ética epistemológica do
artigo: arquitetura pode sobreviver a um experimento falho, mas o claim não.

### 4. A red-team review e as retratações

Reconstruir a transição v3→v4. Explicar, com precisão, os três defeitos da conclusão
antiga sobre friction: oracle não versionado, fragilidade estatística a colisões e bug de
simulador que removia sinal físico. Mostrar a retratação como resultado do laboratório,
não como nota de rodapé.

**Figura recomendada:** linha de evidência “claim inicial → defeito encontrado → correção
do ambiente/protocolo → claim substituído”.

### 5. O protocolo de evidência

Definir o sistema experimental final: ambientes, observações, randomização por episódio,
splits agrupados por episódio, seeds, métricas, checkpoints e critérios de reprodução.
Separar:

- garantias arquiteturais;
- estimadores/oracles;
- modelos aprendidos;
- políticas de coleta;
- resultados históricos versus resultados reexecutados.

Este capítulo deve conter uma tabela de claims com estado `VERIFICADO`, `PARCIAL`,
`EM_VALIDAÇÃO` ou `REFUTADO` e links para os artefatos.

### 6. Resultados I - o que a arquitetura garante

Apresentar passthrough imutável, determinismo, contratos físicos, segurança de
checkpoint, frame stacking, objetivo VICReg/prediction grounding e rollout training.
Não vender garantias de representação como prova de entendimento de física. Mostrar o
trade-off de horizonte e o limite linear quando aplicável.

**Fontes:** Architecture Plan, Blocks B6–B9, Results Ledger, Model Card e testes.

### 7. Resultados II - informação, features e belief

Contar a sequência completa: o oracle mostra informação disponível; raw-GRU falha;
o pacote B10 (features físicas + head distribucional + contraste) produz resultados
positivos. Distinguir rigorosamente “o pacote foi reproduzido” de “features isoladas
causaram o ganho”.

Usar os números reproduzidos do relatório de 2026-07-21 e não substituir o estado de
um claim por narrativa conveniente.

### 8. Resultados III - coleta como parte do modelo

Apresentar active versus random como experimento de desenho de evidência. Reportar as
duas seeds de treino, a faixa 38,42%–45,26%, a média 41,80%, o não-ganho médio de
`friction_box` e a falha de obstáculos não observáveis. A figura de anti-fragilidade
deve mostrar que uma política informativa também pode produzir viés sistemático.

### 9. Escala e fronteira - MuJoCo como experimento em validação

Descrever a motivação, a modelagem de contato, COAST/PUSH e a questão estrutural μ·g.
Não tratar os valores históricos de friction/mass como claim final até que MuJoCo esteja
pinado e o protocolo seja reproduzido. A versão final pode incluir este capítulo como
**linha de pesquisa em validação**, com uma caixa visual separada de resultados
confirmados.

### 10. O produto final

Explicar o que o leitor recebe hoje: release pública v4.0.0, pacote Python, scripts,
configs, ambientes, testes, checkpoints safetensors, licença MIT, documentação, ONNX e
ledger. Separar explicitamente:

- **release v4.0.0:** artefato público e citável;
- **linha posterior no `main`:** trabalho pós-release que ainda não é release;
- **artefatos de auditoria 2026-07-21:** reprodução local, não release.

### 11. Limites, implicações e agenda

Fechar com as fronteiras reais: dois ambientes simulados, sem visão, sem robótica real,
sem controle safety-critical, dependência de protocolo e ausência atual de reprodução
MuJoCo. A agenda deve ser concreta: ablação fatorial de features, avaliação final em
test split, mais coletas independentes, dependência MuJoCo pinada e release posterior.

## Apêndices obrigatórios

1. Ledger de claims e estado atual.
2. Protocolo de dados, splits, seeds e seleção de checkpoint.
3. Tabela de retratações e correções, derivada de
   [`../docs/ERRORS_AND_CORRECTIONS.md`](../docs/ERRORS_AND_CORRECTIONS.md).
4. Tabela de ambientes e limites de generalização.
5. Referências bibliográficas completas e citações de software/datasets.
6. Declaração de assistência por IA, se houver, e revisão humana dos claims.

## Figuras existentes e função narrativa

| Figura | Função na versão final | Condição |
|---|---|---|
| `fig1_thesis.pdf` | tese e cadeia de evidência | abertura/cap. 1 |
| `fig3_horizon.pdf` | trade-off de rollout e horizonte | cap. 6 |
| `fig4_antifragility.pdf` | custo da política informativa | cap. 8 |
| `fig2_mujoco.pdf` | fronteira/experimento em validação | cap. 9; legenda não conclusiva |

## Gates de escrita

| Gate | Condição para avançar |
|---|---|
| G1 - corpus | fontes da tabela de verdade cruzadas; nenhuma métrica sem origem |
| G2 - claims | cada claim classificado e toda divergência exibida |
| G3 - narrativa | capítulos 1–8 escritos a partir de evidência, não de retrospectiva publicitária |
| G4 - produto | release, acesso, instalação e limites descritos corretamente |
| G5 - revisão | César aprova claims visíveis; só então gerar versão Lab OS/Frontend/paper |
| G6 - publicação | decisão humana separada para cada superfície pública |

## Tom do Mycellium Lab

Escrever com precisão, ambição e responsabilidade: explicar o que foi aprendido, onde o
sistema falhou e como a evidência mudou a arquitetura. Evitar “breakthrough”, “SOTA” e
promessas de autonomia ou robótica que os dados não sustentam. O valor editorial do
Atlas é a capacidade de tornar seu próprio processo científico inspecionável.
