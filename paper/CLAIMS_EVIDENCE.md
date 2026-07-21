# Atlas.WM - claims, evidências e gates editoriais

Esta é a tabela canônica de claims do artigo vivo. Ela não substitui os artefatos de
execução: aponta para eles e informa o que pode entrar em cada versão editorial. A
classificação final é humana; os estados abaixo são a leitura técnica da auditoria de
2026-07-21.

## Regra de uso

- Um capítulo só pode apresentar um claim como resultado quando sua linha estiver
  `VERIFICADO` e a formulação reproduzir exatamente o escopo indicado.
- `PARCIAL` pode entrar apenas como evidência limitada, com a limitação no mesmo bloco.
- `EM VALIDAÇÃO` e `HISTÓRICO` não entram como conclusão; entram, quando úteis, como
  agenda de pesquisa ou proveniência.
- Toda alteração cria uma nova linha ou atualiza `última revisão`, preservando a fonte
  anterior no `WIKI.md` e no Lab OS.

| ID | Claim editorial permitido | Tipo | Estado técnico | Evidência primária | Limite ou divergência | Próximo gate |
|---|---|---|---|---|---|---|
| AT-01 | Atlas.WM é um artefato executável para estudar identificabilidade em ambientes simulados parcialmente observáveis. | escopo/produto | `VERIFICADO` | `README.md`, `docs/MODEL_CARD.md`, release v4.0.0 | Não é robótica real, visão ou controle safety-critical. | Revisão humana de escopo público. |
| AT-02 | `z_static_immutable` não deriva em rollout open-loop no contrato avaliado. | garantia arquitetural | `VERIFICADO` | `docs/RESULTS.md`, `scripts/evaluate.py`, testes de rollout | Zero drift não prova conteúdo semântico do latente. | Manter teste de regressão na próxima release. |
| AT-03 | Há sinal suficiente para estimar qualitativamente `friction_agent` sob a política aleatória e o protocolo registrado. | resultado | `VERIFICADO` | `AI Search/Research/2026-07-21-atlas-reproduction-report.md`, oracle comprometido | Escopo limitado ao ambiente toy e à execução registrada. | Aprovação humana da formulação pública. |
| AT-04 | O oracle atinge exatamente R² = 0.865 e MAE = 0.006. | métrica histórica | `PARCIAL` | `docs/RESULTS.md` | Auditoria atual: R² = 0.835426, MAE = 0.006753; população/protocolo ainda não reconciliados. | Reconciliar dataset, episódios e comando histórico. |
| AT-05 | O pacote B10 reproduz scores positivos de gravity, `friction_agent` e `friction_box` no toy environment. | resultado | `VERIFICADO` | relatório de reprodução, checkpoints e artefatos de auditoria | Vale para o pacote completo e seed/protocolo descritos. | Aprovação humana do texto e inclusão da tabela de protocolo. |
| AT-06 | Features físicas isoladas causam o ganho do B10. | causalidade | `PARCIAL` | controle raw/window 40, relatório de reprodução | Head, loss, InfoNCE e janela também mudam. | Executar ablação fatorial. |
| AT-07 | A coleta ativa supera a aleatória no protocolo auditado. | resultado | `VERIFICADO` | relatório de reprodução: +38.42% e +45.26%, duas seeds | Uma única base de coleta; `friction_box` não ganha em média. | Adicionar coletas independentes e revisar claim público. |
| AT-08 | A política informativa precisa ser robusta a variáveis não observadas. | finding/metodologia | `PARCIAL` | `docs/RESULTS.md`, documentação B11 | A evidência atual vem do ambiente toy. | Reproduzir em coletas independentes. |
| AT-09 | COAST/PUSH transfere a identificação para MuJoCo. | resultado de transferência | `EM VALIDAÇÃO` | scripts e valores históricos em `docs/RESULTS.md` | MuJoCo não está declarado/pinado; não houve reexecução atual. | Escolher versão, piná-la, reproduzir e preservar artefatos. |

## Leituras para cada superfície

| Superfície | Pode usar agora | Não pode usar agora |
|---|---|---|
| Artigo vivo | AT-01 a AT-05 e AT-07, com limites visíveis; AT-06, AT-08 e AT-09 como agenda. | Métrica histórica AT-04 como número atual; MuJoCo como conclusão. |
| Lab OS | Toda a tabela, estados, fontes, mudanças e gates. | Simplificar estados ou ocultar retratações. |
| Frontend público | AT-01, AT-02, AT-03 e a formulação limitada de AT-05/AT-07, após aprovação humana. | Claims causais e transferência MuJoCo. |

## Próxima revisão

Criar uma atualização quando houver ablação fatorial, reconciliação do oracle ou execução
MuJoCo pinada. O artigo em [`DRAFT.md`](DRAFT.md) deve refletir esta tabela, nunca
antecipar um estado futuro.
