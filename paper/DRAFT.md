# The Recipe, Not the Data: An Audited Ledger of Physics Identifiability in a Structured World Model

**Cesar Augusto** · Atlas.WM v4.x · draft v0.1 (2026-07-04)

## Abstract

We present an end-to-end audited study of physical-parameter
identifiability in a small structured world model, across a toy 2D
environment and a MuJoCo contact-physics environment. Our starting point
is a retraction: the project's previous conclusion — that a friction
parameter was "not identifiable" under random exploration — was wrong,
caused by an uncommitted evaluation oracle whose MSE objective was
destroyed by heavy-tailed collision outliers, compounded by a simulator
bug that silently voided the physics signal. Re-verifying every claim
produced a three-step thesis, each step measured under episode-held-out
evaluation: (1) **the information exists** — a robust closed-form
estimator recovers friction at R² = 0.87 from the same noisy data where a
GRU belief encoder scores −0.49; (2) **features unlock learning** —
handing the estimator's sufficient statistics to the same GRU flips all
three parameters positive (gravity +0.43); (3) **collection amplifies** —
an information-seeking policy improves belief quality +39% over random
data with the model held fixed. The thesis transfers to MuJoCo contact
physics (friction +0.28, mass +0.11 from a zero baseline), where analysis
also yields a structural identifiability result (only the μ·g product
enters box dynamics) and a cautionary finding: deterministic
information-seeking policies require *anti-fragility to unobservables* —
a fixed-pattern policy repeatedly struck unseen obstacles and silently
poisoned its own dataset. All claims map to committed scripts, seeded
runs, and regression tests.

## 1. Introduction

Negative identifiability results in world-model research are commonly
attributed to the *data regime* — "the parameter is not observable under
this policy." We document a case where that conclusion was published in a
project's model card and was wrong three ways at once: the supporting
oracle was never committed (unreproducible), its objective was
statistically fragile (MSE under heavy-tailed outliers), and the
environment itself had a bug that removed the signal being measured.

Rather than merely fixing the record, we use the retraction as a method:
every subsequent claim is (i) reproduced by a committed script, (ii)
evaluated with episode-grouped splits (row-level splits leak episode
labels through overlapping windows — we measure the leak at R² ≈ 0.3 on
pure noise), and (iii) guarded by a regression test. The result is a
compact, fully auditable account of *what makes physics learnable* in a
latent world model, on two environments.

## 2. Setup

**Environments.** (a) *CruelGridworld*: 3 bodies on a bounded plane,
nonlinear inter-object attraction (G/d², active 1<d<10), per-episode
gravity/friction randomization, process noise σ=0.05, 8 discrete force
actions, 6-D position-only observations. (b) *MujocoPointMass*: an
actuated ball and two passive boxes on a bounded MuJoCo plane; episode
latents are sliding friction, box masses and gravity; same discrete
action interface (+ a no-op).

**Belief model.** A GRU over K-step same-episode windows (RMA/VariBAD
lineage) with a heteroscedastic Gaussian head; optionally a half-window
InfoNCE episode-contrastive term. Inputs are either raw
(obs, Δobs, action) sequences or engineered per-step dynamics features
(below).

**Protocol.** All splits are grouped by episode. Supervised held-out R²
per parameter is the headline metric; ridge probes on frozen
representations corroborate. Training is fully seeded; a canary test
asserts bit-identical loss traces.

## 3. A retraction as a methodology lesson

The prior model card claimed friction_agent unidentifiable (oracle
"R² < 0"). A committed median-of-ratios estimator on position-only
observations — per-step decay ratios ⟨v', v+αu⟩/‖v+αu‖², gated by
excitation, boundary-bounce and interaction filters, aggregated by the
median — recovers it at **R² = 0.865 (MAE 0.006)** on the *same* noisy
random-policy data. The original oracle failed because wall/obstacle
bounces create heavy-tailed ratio outliers: a least-squares fit of the
identical quantity scores R² = −35. Separately, a containment bug let
passive boxes exit the arena, silently removing the interaction signal
for the other two parameters. Lesson: *negative identifiability claims
inherit every fragility of their oracle and their simulator.*

## 4. Step 1 — the information exists

With the environment fixed and pipelines made consistent, the closed-form
estimator sets the reference: friction R² 0.865. The learned baseline —
GRU on raw windows — scores **−0.49** on the same data, while its training
loss decreases: it memorizes episodes rather than learning the
dynamics→parameter map. The gap is not capacity; a linear model on one
engineered feature outperforms the full pipeline.

## 5. Step 2 — features unlock learning

We hand the GRU the estimator's sufficient statistics: gated decay ratio
and its *running median*, excitation, distances, box-acceleration
projections onto attractor directions, inverse-square regressors, aligned
actions (27 dims). Same model, same data: gravity **+0.43**,
friction_agent **+0.22**, friction_box **+0.09** (from −0.05/−0.49/−0.10).
Two measured failure modes en route: (i) a single mis-aligned action index
nullifies every ratio (and a test fixture can bake in the same bug); (ii)
an *evidence-length bound*: the target's std (0.025) lies below the
optimal estimator's window-level error at 18 steps (0.043) — SNR < 1
means no learner can go positive at that window; 38-step windows reach
signal scale.

## 6. Step 3 — collection amplifies

An information-seeking policy (oscillating dash for excitation; box-stir
for interaction signal) raises belief quality **+39%** (mean R² 0.246 →
0.341; gravity 0.67) with the model and training held fixed. The first
policy version dashed along a fixed line; when that line crossed an
*unobservable* obstacle it struck it repeatedly, corrupting the episode's
median — episode-level R² collapsed to 0.23 with near-unchanged MAE, a
heavy-tail signature — and belief training on that data silently failed.
Random walks do not repeat their mistakes; deterministic
information-seekers do. A golden-angle rosette rotation restores
robustness. We name the requirement *anti-fragility to unobservables*.

## 7. The thesis transfers: MuJoCo contact physics

Random policy + generic statistics: R² ≈ 0 on all parameters (332
episodes). Analysis explains why and what is identifiable at all: the
actuated agent rides slide joints (gravity absorbed by the joint; normal
force ≈ 0 → it slides nearly frictionless), so only the *boxes* feel
friction; Coulomb friction decelerates linearly (ratio features are the
wrong model class); and gravity enters box dynamics only through the μ·g
product — *gravity alone is structurally unidentifiable*, an exclusion we
make with a measurement rather than by assertion. A COAST/PUSH policy
(no-op actions manufacture free sliding; box-directed pushes manufacture
contacts) plus aligned-impulse features (head-on contacts only; glancing
hits confound geometry with mass at R² −0.27) yields learned beliefs of
**friction +0.28, mass +0.11** at 1048 episodes.

## 8. Related work

Belief-style system identification: RMA (Kumar et al. 2021), VariBAD
(Zintgraf et al. 2020), CRAFT (2025). Active system identification: ASID
(Memmel et al. 2024), SPI-Active (2025) — our policies replace the Fisher
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
floored by process noise. Mass identification (+0.11) remains weak —
contact events are sparse even under PUSH phases.

## 10. Reproducibility

Everything in this paper maps to a committed artifact:
github.com/cesaremcasa/Atlas.WM (v4.0.0 tag + v4.1 line): scripts (oracle,
policies, feature builders, trainers), seeded configs, 196 tests including
regression locks on the oracle score, the information advantage, and the
training canary. `docs/RESULTS.md` is the claim-by-claim evidence ledger.
