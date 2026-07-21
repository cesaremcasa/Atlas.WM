# The Recipe, Not the Data: An Audited Ledger of Physics Identifiability in a Structured World Model

**Cesar Augusto** · Atlas.WM v4.x · living research corpus (2026-07-21)

> Este é o corpus vivo do artigo. A arquitetura narrativa, as fontes de verdade e os
> gates da versão final estão em [`FINAL_VERSION_ROADMAP.md`](FINAL_VERSION_ROADMAP.md).
> O apêndice público de erros, retratações e correções está em
> [`../docs/ERRORS_AND_CORRECTIONS.md`](../docs/ERRORS_AND_CORRECTIONS.md).
> Claims, evidências e gates editoriais estão em
> [`CLAIMS_EVIDENCE.md`](CLAIMS_EVIDENCE.md).

## Abstract

Atlas.WM is a small, executable research system for studying when physical
parameters can be inferred from partial trajectories. This paper records its
construction as a laboratory artifact: the architecture, experiments that
failed, the red-team review that invalidated earlier conclusions, the rebuilt
evidence protocol, and the released product. Its question is not whether one
model can estimate physics, but which conditions make an identifiability claim
inspectable.

The first conclusion about `friction_agent` was retracted. Its oracle was not
versioned, least-squares evaluation was unstable under collision outliers, and
the simulator allowed passive bodies to leave the arena. The corpus therefore
separates historical results from re-executed evidence. In the 2026-07-21
audit, a committed robust oracle reached R² = 0.835426 and MAE = 0.006753 on
its recorded run; this supports qualitative identifiability but does not yet
reconcile the historical R² = 0.865 value. A reproduced B10 configuration also
showed positive held-out scores for gravity (0.4286), `friction_agent` (0.2172)
and `friction_box` (0.0928). MuJoCo results remain historical until its
dependency and protocol are reproduced.

The contribution is an auditable method and record, rather than a claim of
general physical understanding. Every final quantitative statement will be tied
to a dated artifact and a public correction ledger.

## 1. The question before the model

World models are often evaluated as if a low loss established that a latent
state captured the physical variables that matter. Atlas starts from a narrower
proposition: a parameter may be present in the data, absent from a
representation, or made invisible by the measurement protocol. These are
different failures and must not share a conclusion.

The project uses partial observations, episode-level physics randomization and
controlled actions to ask when gravity, friction and mass can be recovered. It
does not claim real-world robotics, visual understanding or general physical
reasoning. Its value is the preserved chain from architectural decision to
executable artifact to qualified conclusion.

We classify claims as verified, partial, under validation or retracted. A
correction is part of the result, not an editorial inconvenience.

## 2. The first architecture

The v3 architecture separates what must remain unchanged from what can vary.
Its latent interfaces distinguish immutable, slow, dynamic and controllable
components; action routing is explicit rather than inferred by an adversarial
critic; checkpoints, configuration and exports are part of the research
contract. These are engineering hypotheses, not proof that a representation
carries semantic physics.

CruelGridworld places three bodies on a bounded plane with per-episode gravity
and friction randomization, process noise and discrete force actions.
Observations are positions only. The later MuJoCo point-mass environment
extends the inquiry to contact dynamics, but remains a separate validation tier.

The belief model consumes same-episode windows of observations, differences and
actions. It can receive raw sequences or engineered physical statistics. The
protocol uses episode-grouped splits, declared seeds, held-out R² per parameter
and preserved configs/checkpoints. Grouping by episode is essential: overlapping
row-level windows can leak episode identity and manufacture apparent skill.

## 3. The first failure and the red-team review

The first implementation produced a suspiciously perfect validation loss in an
environment simple enough to support memorization. That result was not carried
forward as evidence of generalization. The later review found a more serious
problem in the negative conclusion about `friction_agent`: the supporting oracle
was not versioned, its least-squares objective was dominated by collision
outliers, and passive boxes could cross boundaries and remove signal needed by
other parameters.

The response was reconstruction, not cosmetic repair. The environment gained
containment tests; normalization and data fingerprints became explicit; windows
and splits were reviewed; the oracle was committed as a robust median-based
estimator with excitation and collision filters; and deprecated claims were
marked as rebased or retracted. The full record is maintained in
[`ERRORS_AND_CORRECTIONS.md`](../docs/ERRORS_AND_CORRECTIONS.md).

The historical run reported R² = 0.865 and MAE = 0.006 for the robust oracle.
The later recorded audit reached R² = 0.835426 and MAE = 0.006753 on a defined
400-episode execution. The figures support the same qualitative lesson - useful
friction signal exists in this regime - but are not interchangeable. Until the
episode population and exact protocol are reconciled, the historical figure is
not presented as a current reproduced metric.

## 4. Step 1 - the information exists

The robust oracle is the first evidence that useful friction signal is present
in the recorded toy regime. The historical ledger reports R² = 0.865, while
the dated 2026-07-21 rerun reports R² = 0.835426 and MAE = 0.006753. This is a
reference estimator, not a learned world model and not evidence of broad
generalization.

The historical raw-GRU comparison remains part of the investigation, but its
exact metric is not promoted until the matching dataset, split and checkpoint
are reconciled with the audit. The durable conclusion is narrower: an available
signal does not guarantee that a generic learned representation uses it.

## 5. Step 2 - features unlock learning

The B10 audit reproduced a configuration combining physical statistics,
distributional prediction and contrastive training. At window 40 it produced
held-out R² values of 0.4286 for gravity, 0.2172 for `friction_agent` and 0.0928
for `friction_box`. The engineered inputs include decay-ratio summaries,
excitation, distances, acceleration projections and aligned actions.

This is evidence for the reproduced configuration, not an isolated feature
ablation. The raw comparison changes more than the feature set, so it cannot
establish that features alone caused the gain. The next experiment is a
factorial ablation. Recorded failure modes also remain part of the chapter:
misaligned action indexing can nullify a ratio statistic, and short windows can
fall below the estimator's signal-to-noise scale.

## 6. Step 3 - collection amplifies

The active-versus-random audit measures collection as part of the experimental
system. With the recorded collection base and two training seeds, the aggregate
improvement was 38.42% and 45.26%, for a mean of 41.80%. `friction_box` did not
show a corresponding mean gain. This is a contextual result, not a universal
active-learning claim; additional independent collections are required.

The first deterministic policy also exposed a methodological limit. When its
fixed trajectory met an unobserved obstacle, it repeated the collision and
contaminated its own evidence. A rosette-style rotation was introduced to avoid
that pattern. The lesson is retained as a design requirement: an informative
policy must be robust to variables it does not observe.

## 7. MuJoCo contact physics: a validation frontier

MuJoCo extends the question to contact dynamics, but it is not currently a
reproduced result. The repository environment used in the 2026-07-21 audit does
not declare or pin the MuJoCo dependency. Consequently, historical values for
friction and mass are retained only as provenance and are not presented as final
evidence.

The analysis remains useful as a hypothesis for the next validation: slide
joints can absorb gravity for the actuated body, box dynamics can make the
product μ·g more directly observable than gravity alone, and contact geometry
can confound mass estimation. COAST/PUSH collection and aligned-impulse features
must be rerun in a pinned environment before this chapter can advance from
validation frontier to result.

## 8. Related work

Belief-style system identification: RMA (Kumar et al. 2021), VariBAD
(Zintgraf et al. 2020), CRAFT (2025). Active system identification: ASID
(Memmel et al. 2024), SPI-Active (2025) - our policies replace the Fisher
objective with measured information-rate proxies. Self-predictive
stability: BYOL/SimSiam analyses, Ni et al. (2024); our world-model recipe
uses VICReg-style regularization plus prediction grounding. Contrastive
identifiability: DYSCO (2026). Our contribution is not a new algorithm
but an audited, fully reproducible account connecting these pieces, with
retraction, failure ledger, and structural identifiability analysis.

## 9. Limitations

Two synthetic environments; no visual observations; policies are designed
(greedy on measured proxies), not learned; the world model still sits
2.3× above its linear information ceiling on MuJoCo; long-horizon error is
floored by process noise. Mass identification (+0.11) remains weak -
contact events are sparse even under PUSH phases.

## 10. Reproducibility

Everything in this paper maps to a committed artifact:
github.com/cesaremcasa/Atlas.WM (v4.0.0 tag + post-release line): scripts,
policies, feature builders, trainers, seeded configs and tests. The 2026-07-21
audit passed 186 tests on Python 3.11; MuJoCo tests were unavailable in that
environment. `docs/RESULTS.md`, the reproduction report and the correction
ledger are the claim-by-claim evidence record.
