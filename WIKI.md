# WIKI - Atlas.WM

## Agora

Estado: `COMPLETED / RELEASED`.

- GitHub canônico: <https://github.com/cesaremcasa/Atlas.WM>.
- Repositório público, branch padrão `main`.
- Checkout local e GitHub alinhados em `db1016664fad`.
- Release atual: `v4.0.0`, publicada em 2026-07-04.
- Nenhuma PR ou issue aberta na verificação de 2026-07-20.
- Worktree local limpo.

A revisão confirmou o estado do repositório e da release; a suíte de 195 testes citada
no README não foi reexecutada nesta revisão.

## Evolução

### 2026-07-04

- Publicada a v4.0.0 com ledger de resultados, limites e retratações.

### 2026-07-06

- `uv.lock` regenerado depois do bump de versão.
- Este é o commit atual de `main` e do checkout local.

### 2026-07-20

- Estado local confrontado com o GitHub.
- Projeto permanece concluído e preservado; nenhuma promoção pendente foi encontrada.

### 2026-07-21 - piloto do roadmap de artigo

- O roadmap canônico foi aplicado ao Atlas: inventário, registro de claims, busca de
  contexto externo e reexecução local.
- A suíte local passou com **186 testes** em Python 3.11.15. Python 3.13/3.14 não foi
  validado neste Mac porque PyTorch não foi resolvido nesse ambiente.
- O oracle confirmou a direção do claim de identificabilidade sob política aleatória
  (R² 0,835; MAE 0,0068), mas não repetiu exatamente o número histórico do ledger
  (R² 0,865; MAE 0,006). O número histórico não deve ser promovido ao artigo sem
  reconciliação de artefatos.
- Claims de features, ganho de coleta ativa e transferência MuJoCo permanecem candidatos
  internos até reexecução completa com artefatos de métricas. O dossiê está em
  `AI Search/Research/2026-07-21-atlas-article-evidence-audit.md`.
