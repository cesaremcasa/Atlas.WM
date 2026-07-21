# Atlas.WM - auditoria de evidência para Article Map

Origem: `GENERATED_BY_AGENT` - README, WIKI, Results Ledger, Model Card, código,
testes e fontes externas listadas abaixo - verificado em 2026-07-21.
Estado: `UNVERIFIED` até decisão de César sobre a promoção dos claims.

## Fotografia observada

- Repositório público: `cesaremcasa/Atlas.WM`; `main` local em `db1016664fad`.
- Última release pública: `v4.0.0`, 2026-07-04.
- A execução local com Python 3.11.15 concluiu: **186 testes passaram**.
- Python 3.13/3.14 não instalou PyTorch neste ambiente; Python 3.11 é o runtime
  local validado nesta auditoria.

## Claims e classificação proposta

| Claim | Estado proposto | Evidência | Limite / ação necessária |
|---|---|---|---|
| O `z_static_immutable` não deriva em rollout open-loop. | `VERIFICADO` | suíte local passou, incluindo contrato de rollout; arquitetura usa passthrough. | O número “exatamente 0,0” é garantia desta arquitetura e ambiente, não claim sobre sistemas físicos reais. |
| `friction_agent` é identificável em dados de posição sob política aleatória. | `VERIFICADO` | oracle atual, 400 episódios/50 passos/seed 42/ruído 0,05: R² **0,835**, MAE **0,0068**; `tests/test_oracle_friction.py` passou. | O número histórico R² 0,865/MAE 0,006 não foi reproduzido exatamente; publicar a faixa/condição ou reconciliar artefatos antes de afirmar o número histórico. |
| Features de dinâmica tornam o belief encoder capaz de recuperar parâmetros. | `PARCIAL` | tabela e scripts em `docs/RESULTS.md` e `scripts/train_physics_belief.py`; testes de plumbing passam. | Reexecutar o experimento completo e guardar artefato de métricas antes de publicar os R² específicos. |
| Coleta ativa melhora em 39% a qualidade do belief. | `PARCIAL` | ledger e `tests/test_active_exploration.py` preservam a vantagem qualitativa. | Reexecutar a comparação de treinamento, fixar dataset/config/checkpoint e registrar intervalo/seed; não publicar “39%” antes disso. |
| A tese transfere para MuJoCo com friction +0,28 e mass +0,11. | `PARCIAL` | código, ambiente e ledger existem; claim está no `main` posterior à release v4.0.0. | Resultado não está em release pública e requer artefato reproduzido; manter como experimento interno. |
| O projeto não é robótica real, visão nem controle safety-critical. | `VERIFICADO` | escopo e limites explícitos no Model Card. | Deve aparecer no artigo e no Frontend. |

## Contexto externo - não prova os números do Atlas

- Ha & Schmidhuber, *World Models*, arXiv:1803.10122 - contexto de representações
  espaciais e temporais compactas.
- Zintgraf et al., *VariBAD*, arXiv:1910.08348 - contexto de inferência de ambiente e
  adaptação em ambientes desconhecidos.
- Kumar et al., *RMA*, arXiv:2107.04034 - contexto de módulo de adaptação para dinâmica
  não observada.
- Bardes, Ponce & LeCun, *VICReg*, arXiv:2105.04906 - contexto da regularização usada
  para evitar colapso de representação.

Os PDFs foram preservados em `AI Search/Papers/` com SHA-256 registrado no terminal da
auditoria. Eles devem ser citados como trabalho relacionado, nunca como evidência dos
resultados internos.

| Arquivo | SHA-256 |
|---|---|
| `2018-ha-schmidhuber-world-models-arxiv-1803.10122.pdf` | `b0c1e30aab53efd28ddf61d661f680150918d4d03b77bae62bc52d62dbd76cce` |
| `2019-zintgraf-variBAD-arxiv-1910.08348.pdf` | `e8b471be1a5f35dc49c9f49a9c46521d1248a811ac5fb9c30935363a13f4f1e9` |
| `2021-kumar-rma-arxiv-2107.04034.pdf` | `d8660da881851756287efab24d85f5c629ba97878f70297eb429a8e8a9a57846` |
| `2021-bardes-vicreg-arxiv-2105.04906.pdf` | `c227c290d5eb2ba459fd502de3b4e1da36a0ea4c88989b7d8153e211751e9e77` |

## Tese editorial proposta

**Atlas.WM mostra que conclusões negativas sobre identificabilidade precisam sobreviver a
oracles reproduzíveis, splits sem vazamento e auditoria de simulador; quando a evidência
é suficiente, estrutura de features e coleta de dados podem ser mais decisivas que a
capacidade do modelo.**

## Rascunho seguro para Lab OS e Frontend

### Atlas.WM - auditoria antes da ambição

Atlas.WM é um projeto de pesquisa sobre modelos de mundo estruturados. Em vez de tratar
uma representação latente como uma caixa-preta, ele separa componentes imutáveis,
dinâmicos e controláveis e testa explicitamente o que cada parte consegue preservar.

O resultado mais sólido não é uma promessa de robótica real: é uma prática de auditoria.
Na revalidação atual, um oracle reproduzível recuperou o atrito do agente a partir de
observações de posição sob uma política aleatória (R² 0,835; MAE 0,0068, neste ambiente).
Isso substitui uma conclusão anterior que foi retratada quando o oracle e o simulador
foram revisados.

O projeto é um artefato de pesquisa em dois ambientes sintéticos, não um sistema de
controle em produção, visão ou robótica física. Claims sobre features de dinâmica,
coleta ativa e transferência para MuJoCo continuam internos até uma nova execução
produzir artefatos de métricas atuais.

## Próximo gate

Antes de publicação externa: decidir se o artigo apresenta apenas os claims `VERIFICADO`
ou se reexecuta os experimentos `PARCIAL` para publicar métricas numéricas atuais.
