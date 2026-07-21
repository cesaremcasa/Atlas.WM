# Atlas.WM - relatório de reprodução dos claims parciais

Origem: `GENERATED_BY_AGENT` a partir do checkout local, documentação, código,
artefatos históricos e execuções descritas neste relatório.

Data da execução: 2026-07-21. Estado: relatório de auditoria; **não autoriza
publicação nem alteração de claims canônicos**.

## Resumo executivo

- A coleta, os splits e os três checkpoints seed 42 de gridworld foram reproduzidos
  de forma isolada. Os arrays novos são byte a byte idênticos aos históricos, e os
  tensores dos checkpoints raw, engineered-random e engineered-active também são
  bit a bit idênticos aos checkpoints históricos correspondentes.
- O belief v2 com features de dinâmica repetiu, na seed 42, gravity **0,4286**,
  friction_agent **0,2172** e friction_box **0,0928** (mean **0,2462**). O baseline
  raw-GRU/window 20 repetiu no ridge probe **−0,0506/−0,4901/−0,0952**.
- O ganho active versus random repetiu o ledger na seed 42: **0,2462 → 0,3408**,
  diferença absoluta **0,0946**, ou **+38,42%**. Na seed de treino 43, o ganho foi
  **0,2397 → 0,3481**, diferença **0,1085**, ou **+45,26%**. A média de duas seeds
  foi **+41,80%**; logo “+39%” é reproduzível na seed histórica, não uma constante.
- A atribuição causal “as features, sozinhas, causam o ganho” continua parcial: o
  pacote B10 também troca head, loss, InfoNCE e janela; além disso, a tabela histórica
  compara o ridge probe do raw-GRU com o head supervisionado do belief v2. Um controle
  raw/window 40 no mesmo head supervisionado ficou negativo (mean **−0,0978**), o que
  reforça a direção, mas não isola todos os componentes do pacote B10.
- MuJoCo não pôde ser reexecutado: o módulo Python `mujoco` não está instalado e não
  é declarado nem pinado em `pyproject.toml`/`uv.lock`. O teste do ambiente é pulado
  por essa ausência. Nenhuma versão arbitrária foi instalada e nenhum número histórico
  foi reapresentado como resultado novo.
- O oracle solicitado produziu **R² 0,835426 / MAE 0,006753**, 363 episódios usados e
  37 descartados. Isso coincide com a auditoria anterior após arredondamento
  (0,835/0,0068), mas não com o ledger histórico (0,865/0,006).

## Status sugerido - sujeito a decisão humana

| Claim auditado | Resultado desta execução | Status sugerido |
|---|---|---|
| A. O pacote belief v2 com features de dinâmica recupera gravity, friction_agent e friction_box melhor que o raw-GRU. | Números seed 42 e tensores históricos reproduzidos; seed 43 mantém resultados positivos; raw/window 40 permanece negativo. A comparação documental mistura métricas e múltiplas mudanças do pacote B10. | `PARCIAL` para a atribuição causal às **features isoladas**; o resultado do **pacote B10** está reproduzido. |
| B. Coleta ativa melhora em aproximadamente 39% a qualidade média do belief. | Seed 42: +38,42% (reprodução do +39% arredondado); seed 43: +45,26%; média de duas seeds: +41,80%. | `VERIFICADO` para “active supera random neste protocolo”; publicar a faixa/duas seeds, não “39%” universal. |
| C. A identificação transfere para MuJoCo via COAST/PUSH (friction +0,28; mass +0,11). | Sem resultado novo: import de `mujoco` falha e a dependência não está pinada. Artefatos históricos foram apenas inventariados. | `EM_VALIDAÇÃO`. |
| D1. friction_agent é identificável sob política aleatória. | R² 0,835426 e MAE 0,006753 no controle pedido; direção positiva forte. | `VERIFICADO` para o claim qualitativo. |
| D2. O número exato do oracle é R² 0,865 / MAE 0,006 no protocolo atual. | Não reproduzido pelo comando atual; há diferença de protocolo/amostra em relação ao ledger on-disk histórico. | `PARCIAL`, não `REFUTADO`, até reconciliar dataset e população de episódios. |

Esses estados são propostas de auditoria. Conforme `Projects/ARTICLE_ROADMAP.md`, a
classificação canônica depende de decisão humana.

## Fotografia Git e ambiente

Estado no início da auditoria:

```text
branch: main
commit: db1016664fada226bdcfbe4167b60fdbc63f6ccd
git status --short:
 M README.md
?? "AI Search/"
?? WIKI.md
```

As mudanças pré-existentes em `README.md`, `WIKI.md` e `AI Search/` foram preservadas.
Nenhum commit, push, merge, deploy ou publicação foi executado.

| Item | Valor observado |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.12.1 |
| Sistema | macOS 26.5.1, build 25F80, arm64 |
| Kernel | Darwin 25.5.0, RELEASE_ARM64_T8132 |
| Preparação | `uv sync --python 3.11 --extra dev` - sucesso, 85 pacotes verificados |
| Testes | 186 aprovados com `uv run --python 3.11 pytest -q` |
| Observação dos testes | warnings de checkpoints unsigned em testes; nenhum failure |
| Teste MuJoCo | módulo pulado na coleta: `No module named 'mujoco'`; execução direcionada terminou com pytest exit 5 por ausência de testes coletados |

O número 195 ainda aparece em `docs/USAGE.md`; a coleta atual aprovou 186 testes e o
módulo MuJoCo foi pulado. Isso deve ser reconciliado antes de usar uma contagem de testes
como evidência editorial.

## Comandos exatos

### Preparação e fotografia

```bash
git status --short
git branch --show-current
git rev-parse HEAD
uv sync --python 3.11 --extra dev
uv run --python 3.11 pytest -q
uv run --python 3.11 python -c "import platform, sys, torch; print(sys.version); print(torch.__version__); print(platform.platform())"
sw_vers
uname -a
```

### Geração e split isolados - gridworld

```bash
uv run --python 3.11 python scripts/generate_data.py \
  --randomize-physics --process-noise-std 0.05 \
  --episode-reset-prob 0.02 --seed 42 --num-samples 50000 \
  --policy random \
  --out-dir artifacts/article-audit-2026-07-21/gridworld/random/raw

uv run --python 3.11 python scripts/generate_data.py \
  --randomize-physics --process-noise-std 0.05 \
  --episode-reset-prob 0.02 --seed 42 --num-samples 50000 \
  --policy active \
  --out-dir artifacts/article-audit-2026-07-21/gridworld/active/raw

uv run --python 3.11 python scripts/split_data.py \
  --raw-dir artifacts/article-audit-2026-07-21/gridworld/random/raw \
  --processed-dir artifacts/article-audit-2026-07-21/gridworld/random/processed

uv run --python 3.11 python scripts/split_data.py \
  --raw-dir artifacts/article-audit-2026-07-21/gridworld/active/raw \
  --processed-dir artifacts/article-audit-2026-07-21/gridworld/active/processed
```

### Treino raw-GRU histórico e belief v2

O raw-GRU foi executado diretamente do blob preservado no commit B5, sem checkout:

```bash
git show 91513051594b5af66e1f54d54e6c39be5ffecbea:scripts/train_physics_belief.py | \
  uv run --python 3.11 python - \
  --config artifacts/article-audit-2026-07-21/configs/gridworld-random.yaml \
  --window-k 20 --epochs 100 --seed 42 \
  --output artifacts/article-audit-2026-07-21/gridworld/checkpoints/random-raw-w20-seed42.safetensors

git show 91513051594b5af66e1f54d54e6c39be5ffecbea:scripts/train_physics_belief.py | \
  uv run --python 3.11 python - \
  --config artifacts/article-audit-2026-07-21/configs/gridworld-random.yaml \
  --window-k 40 --epochs 100 --seed 42 \
  --output artifacts/article-audit-2026-07-21/gridworld/checkpoints/random-raw-w40-seed42.safetensors
```

Belief v2, random e active, seeds de treino 42 e 43:

```bash
uv run --python 3.11 python scripts/train_physics_belief.py \
  --config artifacts/article-audit-2026-07-21/configs/gridworld-random.yaml \
  --window-k 40 --epochs 100 --seed 42 \
  --output artifacts/article-audit-2026-07-21/gridworld/checkpoints/random-engineered-w40-seed42.safetensors

uv run --python 3.11 python scripts/train_physics_belief.py \
  --config artifacts/article-audit-2026-07-21/configs/gridworld-active.yaml \
  --window-k 40 --epochs 100 --seed 42 \
  --output artifacts/article-audit-2026-07-21/gridworld/checkpoints/active-engineered-w40-seed42.safetensors

uv run --python 3.11 python scripts/train_physics_belief.py \
  --config artifacts/article-audit-2026-07-21/configs/gridworld-random.yaml \
  --window-k 40 --epochs 100 --seed 43 \
  --output artifacts/article-audit-2026-07-21/gridworld/checkpoints/random-engineered-w40-seed43.safetensors

uv run --python 3.11 python scripts/train_physics_belief.py \
  --config artifacts/article-audit-2026-07-21/configs/gridworld-active.yaml \
  --window-k 40 --epochs 100 --seed 43 \
  --output artifacts/article-audit-2026-07-21/gridworld/checkpoints/active-engineered-w40-seed43.safetensors
```

Probe agrupado por episódio, repetido trocando apenas o belief checkpoint:

```bash
uv run --python 3.11 python scripts/probe_physics.py \
  --checkpoint checkpoints/best_model.safetensors \
  --belief-checkpoint artifacts/article-audit-2026-07-21/gridworld/checkpoints/random-raw-w20-seed42.safetensors \
  --data-dir artifacts/article-audit-2026-07-21/gridworld/random/processed \
  --split val --ridge-alpha 1.0 --train-frac 0.8

uv run --python 3.11 python scripts/probe_physics.py \
  --checkpoint checkpoints/best_model.safetensors \
  --belief-checkpoint artifacts/article-audit-2026-07-21/gridworld/checkpoints/random-engineered-w40-seed42.safetensors \
  --data-dir artifacts/article-audit-2026-07-21/gridworld/random/processed \
  --split val --ridge-alpha 1.0 --train-frac 0.8
```

### MuJoCo - verificação do bloqueio

```bash
uv run --python 3.11 python -c "import mujoco"
# ModuleNotFoundError: No module named 'mujoco'

uv run --python 3.11 pytest -q tests/test_mujoco_pointmass.py -ra
# SKIPPED [1] ... could not import 'mujoco'
# exit 5: nenhum teste coletado
```

O comando de coleta que ficaria pendente após uma versão ser escolhida, declarada e
pinada por autorização humana é:

```bash
uv run --python 3.11 python scripts/generate_data.py \
  --env mujoco --policy active --randomize-physics \
  --seed 42 --num-samples 100000 --episode-reset-prob 0.01 \
  --out-dir artifacts/article-audit-2026-07-21/mujoco/active/raw
```

Dependência faltante: pacote Python `mujoco` compatível com Python 3.11/arm64, com
versão ainda não declarada. Um futuro `uv add 'mujoco==<VERSÃO_APROVADA>'` alteraria
`pyproject.toml` e `uv.lock`; por isso não foi executado nesta auditoria.

### Oracle

```bash
uv run --python 3.11 python scripts/oracle_friction_agent.py \
  --episodes 400 --steps 50 --seed 42 --process-noise-std 0.05
```

## A. Features de dinâmica / belief encoder

### Protocolo e dados

- Dataset random: 50.000 transições; 959 episódios; média 52,1 passos.
- Split por episódio, shuffle seed 42: train 39.422, val 4.563, test 6.015
  transições.
- Janelas válidas: raw/window 20 = 27.280 train e 3.061 val; window 40 =
  18.571 train e 2.001 val.
- Features engineered: `dynamics_v1`, 27 dimensões; janela 40; head
  heteroscedástico; Gaussian NLL; half-window InfoNCE 0,1.
- Treino: batch 256, Adam 3e-4, 100 epochs, melhor checkpoint por mean val R².

Os cinco arrays raw e os 16 arrays/sentinel comuns dos splits novos correspondem byte a
byte aos diretórios históricos `data/` e `data_active/`. O arquivo extra
`obs_scale.json`, hoje gerado pela CLI, também foi preservado no diretório isolado.

### Resultado do head supervisionado

| Variante | Seed | gravity | friction_agent | friction_box | mean R² |
|---|---:|---:|---:|---:|---:|
| raw, window 20 | 42 | −0,1165 | +0,0002 | −0,0186 | −0,0450 |
| raw, window 40 (controle de janela) | 42 | −0,2290 | −0,0009 | −0,0636 | −0,0978 |
| engineered, window 40 | 42 | **+0,4286** | **+0,2172** | **+0,0928** | **+0,2462** |
| engineered, window 40 | 43 | **+0,4168** | **+0,2173** | **+0,0849** | **+0,2397** |

O checkpoint engineered seed 42 foi salvo no epoch 10; o raw/window 20 no epoch 2.
Treinar até 100 epochs degrada o valor corrente, mas o protocolo escolhe o melhor
checkpoint, exatamente como o script histórico.

### Ridge probe sobre o embedding

| Variante | gravity | friction_agent | friction_box | mean R² |
|---|---:|---:|---:|---:|
| raw-GRU/window 20, 63 episódios | **−0,0506** | **−0,4901** | **−0,0952** | −0,2120 |
| engineered/window 40, 42 episódios | −0,5887 | −0,8253 | +0,3272 | −0,3623 |

O primeiro row reproduz o baseline B5 documentado. O segundo confirma o aviso do
MODEL_CARD: o ridge probe em apenas 42 episódios é frágil e não equivale ao head
supervisionado que gerou +0,43/+0,22/+0,09. A tabela B10 apresenta o baseline raw do
ridge B5 ao lado do head do belief v2. Esses números são reproduzíveis, mas a comparação
não usa uma métrica idêntica.

### Equivalência com artefatos históricos

Comparação por chave/tensor, com `torch.equal`:

| Novo checkpoint | Histórico | Resultado |
|---|---|---|
| random raw/window 20/seed 42 | `checkpoints/physics_belief.safetensors` | mesmas chaves; **todos os tensores idênticos** |
| random engineered/window 40/seed 42 | `checkpoints/belief_v2.safetensors` | mesmas chaves; **todos os tensores idênticos** |
| active engineered/window 40/seed 42 | `checkpoints/belief_active.safetensors` | mesmas chaves; **todos os tensores idênticos** |

Os arquivos `.safetensors` têm SHA-256 diferentes porque a metadata inclui
`trained_at_utc`; a igualdade relevante dos parâmetros foi verificada tensor a tensor.

### Conclusão A

Os números históricos são reprodutíveis na seed 42 e mudam pouco na seed de treino 43.
O pacote B10 é reprodutível. A redação “features de dinâmica causaram sozinhas o ganho”
deve continuar parcial até uma ablação fatorial usar a mesma janela, o mesmo head, a
mesma loss e a mesma métrica, trocando apenas o vetor de entrada.

## B. Coleta ativa versus aleatória

### Protocolo

- Mesmo código/config do belief v2, janela 40, 100 epochs, batch 256, LR 3e-4.
- Mesmo orçamento: 50.000 transições por política; geração seed 42; ruído 0,05;
  reset 0,02.
- Seeds de treino pareadas: 42 e 43.
- Random: 959 episódios; active: 998 episódios. Essa diferença decorre do consumo de
  RNG e da trajetória/política, mas o orçamento de transições permanece igual.
- Métrica: melhor mean val R² do head supervisionado; splits agrupados por episódio.

### Resultado por seed

| Seed de treino | random mean R² | active mean R² | diferença absoluta | diferença percentual |
|---:|---:|---:|---:|---:|
| 42 | 0,246221 | 0,340829 | +0,094609 | **+38,42%** |
| 43 | 0,239659 | 0,348134 | +0,108475 | **+45,26%** |
| média | 0,242940 | 0,344482 | +0,101542 | **+41,80%** |

Desvio-padrão amostral em duas seeds: random 0,00464; active 0,00517; ganho absoluto
0,00980. Duas seeds não sustentam intervalo de confiança confiável.

### Média por parâmetro nas duas seeds

| Parâmetro | random | active | diferença absoluta | diferença percentual |
|---|---:|---:|---:|---:|
| gravity | 0,422691 | 0,653842 | +0,231151 | +54,69% |
| friction_agent | 0,217259 | 0,291091 | +0,073833 | +33,98% |
| friction_box | 0,088870 | 0,088511 | −0,000358 | −0,40% |

O ganho médio é dirigido por gravity e friction_agent; friction_box não melhora em
média nessas duas seeds. A redação editorial deve dizer isso explicitamente.

### Falha conhecida da política ativa

A primeira política determinística usava uma linha fixa. Quando essa linha atravessava
um obstáculo não observável, o agente colidia repetidamente no mesmo ponto e contaminava
a mediana do episódio: R² do estimador caiu para cerca de 0,23 com MAE quase inalterado,
assinatura de cauda pesada. A implementação atual gira o eixo pela razão do ângulo áureo
(rosette), convertendo colisões persistentes em outliers esparsos. Isso mitiga, mas não
elimina, o limite geral: uma política info-seeking determinística pode repetir erros
quando obstáculos relevantes não aparecem na observação.

### Conclusão B

O +39% histórico é reproduzido exatamente por arredondamento na seed 42 e a vantagem
permanece na seed 43. A formulação segura é: “neste protocolo, active elevou mean belief
R² em 38%–45% em duas seeds de treino (média 41,8%)”; não promover uma seed a constante
geral nem afirmar melhora de friction_box.

## C. Transferência para MuJoCo

### Verificação de disponibilidade

- `import mujoco`: falhou com `ModuleNotFoundError`.
- `pyproject.toml`: não declara `mujoco` em dependências ou extras.
- `uv.lock`: não contém pacote MuJoCo.
- `tests/test_mujoco_pointmass.py`: usa `pytest.importorskip("mujoco")`; o módulo foi
  pulado e nenhum contrato de ambiente MuJoCo rodou nesta máquina.
- Nenhum software de sistema, pacote global ou wheel Python não pinado foi instalado.

### Artefatos históricos observados, não reproduzidos

| Item | Evidência histórica observada |
|---|---|
| Dataset `data_mjactive/raw` | 100.000 transições, 1048 episódios, actions com 9 dimensões, physics `[gravity, friction, mass]`; tree SHA-256 em `hashes.json` |
| Features | checkpoint declara `features=mujoco_v1`, `gru_input_dim=16`, window 60 |
| Targets | checkpoint declara `[friction, mass]` |
| Checkpoint | `checkpoints/belief_mujoco.safetensors`, SHA-256 `21afe19e…` |
| Resultado histórico | RESULTS/commit: friction +0,28; mass +0,11 - **não medido novamente** |
| Seed | não consta na metadata do belief checkpoint; não foi inferida como fato novo |

A metadata histórica ainda registra `n_features=27`, embora `gru_input_dim=16` e o código
MuJoCo definam 16 features; registra também `action_dim=8`, enquanto o dataset COAST/PUSH
tem 9 ações. Esses campos não impedem necessariamente o treino, mas reduzem a força da
proveniência e devem ser corrigidos/explicados numa futura reexecução autorizada.

### Bloqueio preciso e próximo gate

Para reproduzir C é necessário primeiro escolher e pinar uma versão do pacote Python
`mujoco` compatível com Python 3.11/arm64 em dependência de projeto, regenerar o lock,
sincronizar a `.venv` e rodar os contratos MuJoCo. Essas ações alteram arquivos canônicos
e exigem autorização humana. Depois disso: gerar 100k transições COAST/PUSH em saída
isolada, split por episódio, treinar window 60 com targets friction/mass e registrar a
seed dentro da metadata do checkpoint.

## D. Oracle de friction_agent

| Fonte/protocolo | R² | MAE | Episódios |
|---|---:|---:|---:|
| Ledger histórico, dataset on-disk | 0,865 | 0,006 | 557 usados, segundo MODEL_CARD |
| Auditoria anterior de 2026-07-21 | 0,835 | 0,0068 | 400 solicitados, não detalhado no relatório anterior |
| Execução atual: 400 × 50, seed 42, ruído 0,05 | **0,835426** | **0,006753** | 363 usados; 37 descartados |

Versus o ledger: ΔR² = −0,029574; ΔMAE = +0,000753. A execução atual repete a auditoria
anterior após arredondamento e não esconde a divergência. O ledger mede o dataset on-disk
e outra população de episódios; por isso a diferença não deve ser chamada de refutação
sem executar o estimator exatamente sobre o dataset/557 episódios históricos ou
reconstruir o artefato original.

## Artefatos, hashes e proveniência

Raiz isolada: `artifacts/article-audit-2026-07-21/`.

| Artefato | Local |
|---|---|
| Fotografia de ambiente | `artifacts/article-audit-2026-07-21/environment.json` |
| Configs wrappers | `artifacts/article-audit-2026-07-21/configs/` |
| Datasets random/active raw + processed | `artifacts/article-audit-2026-07-21/gridworld/{random,active}/` |
| Seis checkpoints novos | `artifacts/article-audit-2026-07-21/gridworld/checkpoints/` |
| Métricas de head, probes e active/random | `artifacts/article-audit-2026-07-21/gridworld/metrics/` |
| Oracle | `artifacts/article-audit-2026-07-21/oracle/oracle-seed42.json` |
| Bloqueio MuJoCo | `artifacts/article-audit-2026-07-21/mujoco/metrics/blocker.json` |
| Hashes resumidos | `artifacts/article-audit-2026-07-21/hashes.json` |
| Manifest de arquivos | `artifacts/article-audit-2026-07-21/manifest.sha256` |

Hashes principais:

| Item | SHA-256 |
|---|---|
| Config canônico `v3_variable_physics.yaml` | `d0a49ae6069a763b22bb1be03336908a41a49440f3f46cc2a66ad73147afcaad` |
| Dataset random/raw (tree) | `71c3dc44876c40cf4c5ebcef0ef5ff1ec320dd77407764a24e7ccfe8a1ae3e44` |
| Dataset random/processed (tree) | `e240db4cfda8549815679595bb12721045103221f47b662db366b2e55061a3db` |
| Dataset active/raw (tree) | `71ee57b0906a31a61a0beb2dd1758aaac06c2e47865d0572622f280ecff9fd49` |
| Dataset active/processed (tree) | `30fb75077d70d3cefc6c0a45181075ad4bd7b9377d5845ceffbc699f1ad7b152` |
| Raw trainer B5 preservado (SHA-256 do blob) | `6e02a9c5995ec6e2f5bcc58a36b27868816631c2132351936c6cf93deaa66a94` |
| `uv.lock` | `a4ee0b84798e36269c0082761db137dc5b3b482ca7998daac2490433c2a43b25` |

O arquivo `hashes.json` registra todos os checkpoints, scripts e tree hashes relevantes;
`manifest.sha256` registra cada arquivo novo individualmente.

## Divergências versus `docs/RESULTS.md`

1. **A, seed 42:** nenhuma divergência numérica no head B10 nem no ridge raw B5;
   checkpoints têm tensores idênticos aos históricos.
2. **A, métrica:** o quadro raw versus engineered mistura ridge probe e head
   supervisionado. O ridge do engineered seed 42 não é positivo, confirmando que o
   head, não a linearidade do embedding, sustenta os números publicados.
3. **B:** +39% é exato apenas como arredondamento da seed 42. A seed 43 dá +45,26%;
   a média de duas seeds dá +41,80%. friction_box não melhora na média.
4. **C:** o checkout não é executável em MuJoCo após o `uv sync` documentado; a
   dependência é ausente e os testes são pulados. +0,28/+0,11 não foi reproduzido.
5. **D:** 0,865/0,006 não foi repetido; a execução atual dá 0,835426/0,006753 e
   coincide com a auditoria anterior.
6. **Testes:** `docs/USAGE.md` diz 195, enquanto esta coleta passa 186 e pula o módulo
   MuJoCo.

## Limitações

- Duas seeds adicionais são seeds de **treino** pareadas sobre datasets gerados com a
  seed histórica 42; não são duas coletas independentes completas.
- O melhor checkpoint é selecionado no mesmo conjunto de validação usado para reportar
  R². Não há avaliação final do head no split `test` documentada no protocolo B10/B11.
- O número de episódios/janelas difere entre random e active apesar do mesmo orçamento
  de transições; isso é parte do efeito da política, mas também altera a quantidade de
  exemplos de treino/validação.
- A ablação raw versus engineered troca mais que features: head, loss, InfoNCE e, no
  baseline histórico, janela. O controle raw/window 40 reduz, mas não elimina, o
  confounding.
- Ridge probes em 42/63 episódios são instáveis; não devem substituir o head
  supervisionado sem múltiplos splits/seeds.
- MuJoCo não foi executado. Artefatos históricos não substituem reprodução atual.
- Os ambientes são simulados; nenhum resultado sustenta claim de robótica física,
  visão, segurança ou controle em produção.
- Os checkpoints são unsigned; foram carregados apenas no contexto local de auditoria.

## Recomendação editorial conservadora

1. Pode-se dizer internamente que **o pacote belief v2 e a vantagem active/random foram
   reproduzidos de forma determinística na seed 42**, incluindo igualdade tensor a
   tensor com os checkpoints históricos.
2. Para A, evitar “features sozinhas provam identificabilidade”. Preferir: “o pacote B10
   (features físicas + head distribucional + contraste) tornou o head supervisionado
   positivo; uma ablação feature-only ainda falta”.
3. Para B, usar “+38% a +45% em duas seeds de treino; média +41,8%”, declarar que
   friction_box não melhorou e incluir a falha de obstáculos não observáveis.
4. Manter C como experimento interno `EM_VALIDAÇÃO` até `mujoco` ser pinado, os contratos
   rodarem e COAST/PUSH produzir dataset/checkpoint/métricas novos em saída isolada.
5. Para D, publicar o resultado atual como execução datada e condicionada
   (**0,8354/0,00675**), preservando o 0,865/0,006 como número histórico não reconciliado.
6. Não atualizar `README.md`, `WIKI.md`, `docs/RESULTS.md` ou `ARTICLE_MAP.md` sem revisão
   humana deste relatório e dos artefatos.
