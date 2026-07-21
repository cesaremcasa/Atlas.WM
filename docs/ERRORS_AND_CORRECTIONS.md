# Atlas.WM - ledger público de erros, correções e limites

Este documento é a fonte canônica para erros registrados no Atlas.WM. Ele acompanha o
artigo e o produto: cada linha preserva o problema, como foi detectado, o efeito sobre os
claims, a correção e o estado atual. Um erro corrigido não é apagado do histórico.

Fontes: postmortem v2, plano de arquitetura v3, red-team v4, CHANGELOG, Model Card,
Results Ledger e auditoria de reprodução de 2026-07-21.

## Como ler

- `CORRIGIDO`: o defeito possui alteração e teste/artefato de prevenção.
- `CORRIGIDO, CLAIM REBASED`: o defeito foi corrigido, mas os números antigos não valem.
- `LIMITADO`: o sistema funciona no escopo declarado, mas a fronteira permanece.
- `ABERTO`: não deve ser apresentado como resultado final.

## Linha de correções

| ID | Erro ou limite | Detecção | Como lidamos | Efeito público atual | Estado |
|---|---|---|---|---|---|
| V2-01 | Ambiente simples permitia memorização e validation loss 0,000. | Postmortem v2. | Aumentamos complexidade, parcial observabilidade e física variável; passamos a desconfiar de métricas perfeitas. | O sucesso arquitetural v2 não é tratado como generalização. | `CORRIGIDO, CLAIM REBASED` |
| C1 | Oracle de `friction_agent` não era versionado; conclusão “não identificável” era falsa. | Red-team e reimplementação independente. | Oracle robusto median-of-ratios foi versionado; claim anterior foi retratado. | Identificabilidade qualitativa é confirmada; métrica exata histórica segue em reconciliação. | `CORRIGIDO, CLAIM REBASED` |
| C2 | Boxes ignoravam paredes/obstáculos, saíam do grid e removiam sinal físico. | Red-team e inspeção de trajetória. | Colisões/containment adicionados para todos os corpos; teste de contenção em múltiplas seeds. | Ceilings v3 de gravity/friction_box são inválidos. | `CORRIGIDO, CLAIM REBASED` |
| C3 | Garantia de latente imutável era tautológica: encoder podia colapsar para constante. | Revisão da arquitetura versus implementação. | Âncora intra/interepisódio e teste de piso de variância; permanece desligada no ambiente sem identidade observável. | “Drift zero” é garantia de dinâmica, não prova de conteúdo semântico. | `LIMITADO` |
| C4 | Critic adversarial de invariância de ação era no-op sob política aleatória. | Análise de independência ação/observação. | Critic foi aposentado; action routing passou a ser arquitetural. | Não há claim de adversarial invariance. | `CORRIGIDO` |
| C5 | Normalização in-place dependia da ordem de scripts e podia criar escala 20× diferente. | Red-team do pipeline. | Normalização passou para memória; split guarda hash de dados brutos; diretórios legados são rejeitados. | Resultados anteriores ao rebaseline não são comparáveis. | `CORRIGIDO, CLAIM REBASED` |
| H1 | Loss auto-preditiva era degenerada, com L2 como remendo e sem rollout multi-step. | Red-team e ablações. | VICReg/EMA, prediction grounding e rollout K-step; seleção por MSE observável. | Ganhos e trade-offs são reportados por horizonte. | `CORRIGIDO` |
| H2 | Treino e DataLoader não tinham seeds. | Inspeção de código. | Seeds Python/NumPy/PyTorch/DataLoader e canário de treino bit-identical. | Resultados reproduzidos devem declarar seed. | `CORRIGIDO` |
| H3 | CI, chaos test e security gate não executavam o que declaravam. | Red-team de workflows. | Workflows foram reconectados a scripts/testes existentes. | Gates anteriores não são evidência; gates atuais são. | `CORRIGIDO, CLAIM REBASED` |
| H4 | Verificação de assinatura de checkpoint era fail-open. | Revisão de segurança. | Modo fail-closed e testes para manifest/key/arquivo inválidos. | Checkpoint assinado é verificável; testes locais podem usar unsigned explicitamente. | `CORRIGIDO` |
| H5 | Janela por episódio tinha off-by-one; testes aceitavam o bug. | Revisão do dataset. | Índice corrigido e contratos revisados. | Contagens de janela anteriores não são comparáveis. | `CORRIGIDO, CLAIM REBASED` |
| H6 | Dimensão imutável configurável não chegava ao modelo/exportação. | Red-team de plumbing/ONNX. | Config e export inferem os splits reais; canário cobre valores não padrão. | Garantia vale apenas para checkpoint/config correspondente. | `CORRIGIDO` |
| M1 | Belief encoder não era integrado ao world model. | Revisão do caminho de inferência. | Belief causal é pré-computado e pode condicionar `z_static_slow`. | Ganho é modesto e dependente de ruído/ambiente. | `LIMITADO` |
| M2 | Uma posição única não torna velocidade/obstáculo observável. | Análise de observabilidade. | Frame stacking de duas observações. | O projeto não reivindica observabilidade total a partir de um frame. | `CORRIGIDO` |
| M3 | Split sequencial de janelas sobrepostas vazava labels por episódio. | Red-team estatístico. | Split agrupado por episódio. | R² v3 de probes é inválido. | `CORRIGIDO, CLAIM REBASED` |
| M4 | Geração sobrescrevia RAW enquanto split antigo podia persistir. | Revisão de sentinela. | Hash de RAW força re-split. | Nenhum treino deve usar split sem fingerprint. | `CORRIGIDO` |
| M5 | Avaliador não calculava métrica. | Inspeção de `evaluate.py`. | MSE open-loop por horizonte e drift de passthrough foram implementados. | Avaliação pré-v4 não é prova de rollout. | `CORRIGIDO, CLAIM REBASED` |
| M6 | Física toy usa Euler explícito e descontinuidades; não equivale a física real. | Revisão de modelo. | Limite documentado; MuJoCo introduz tier de contato separado. | Não há claim de fidelidade física real. | `LIMITADO` |
| M7 | Dependências divergiam entre manifests/locks. | Auditoria de tooling. | `uv.lock` virou lockfile canônico. | Reprodução deve usar `uv sync --locked` e runtime declarado. | `CORRIGIDO` |
| M8 | Modos/checkpoints e hashes de ambiente tinham falhas de restauração. | Red-team de checkpoints. | Plumbing e metadata revisados; comportamento legado documentado. | Checkpoints migrados têm restrições de ambiente. | `CORRIGIDO` |
| L1 | Tags, versões e layout documentado estavam desatualizados. | Auditoria de release. | Release v4, README, changelog e model card foram rebaselined. | Apenas release/tag existente é citável como produto público. | `CORRIGIDO` |
| L2 | EntityEncoder era código morto; testes decorativos e config v1 eram enganosos. | Red-team de cobertura. | Código foi arquivado, testes reforçados e configuração legada marcada. | Não há claim de generalização multiobjeto/equivariance. | `CORRIGIDO` |
| A21-01 | Oracle atual não repete exatamente R² 0,865 / MAE 0,006 do ledger. | Reprodução 2026-07-21. | Resultado atual e população de episódios foram preservados; comparação foi exposta. | Publicar 0,835426 / 0,006753 como execução datada; valor histórico é pendente de reconciliação. | `ABERTO` |
| A21-02 | Comparação raw versus engineered troca mais que features. | Reprodução B10. | Controle raw/window 40 foi adicionado à auditoria; ablação fatorial foi definida. | Não afirmar causalidade de features isoladas. | `ABERTO` |
| A21-03 | Ganho active foi medido em duas seeds de treino sobre uma coleta-base. | Reprodução B11. | Resultados por seed e por parâmetro foram registrados. | Usar faixa 38%–45%, não ganho universal; expandir coletas independentes. | `LIMITADO` |
| A21-04 | MuJoCo não está declarado/pinado no ambiente Python atual. | Reprodução 2026-07-21. | Nenhuma dependência arbitrária foi instalada. | Métricas MuJoCo são históricas, não resultado atual reproduzido. | `ABERTO` |
| A21-05 | Documentação citava 195/196 testes, mas a execução atual aprovou 186 e pulou MuJoCo. | Reprodução 2026-07-21. | Contagem atual foi registrada; documentação precisa de reconciliação. | Não usar contagem antiga como claim. | `ABERTO` |

## Regra para artigo e site

O artigo principal deve conter uma tabela resumida de V2-01, C1–C5, H1–H4, M3, M5 e
A21-01–A21-05. Este ledger completo deve ser linkado como apêndice público. Nenhum erro
deve ser recontado como “melhoria” sem declarar o claim que ele invalidou ou limitou.

## Próxima atualização

Uma linha só muda de estado quando houver comando reproduzível, artefato preservado e
revisão humana da mudança de claim. Fontes de reprodução: `AI Search/Research/` e
`artifacts/article-audit-2026-07-21/`.
