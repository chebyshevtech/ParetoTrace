# ParetoTrace: reviewer repository for the author response

This repository accompanies the author response to the ICLR 2027 reviews of
*ParetoTrace: Minimax Recovery of MGDA Directions from Partial Gradient Geometry*. It
contains:

- the code for every new experiment in the response;
- the results of those experiments;
- scripts that re-derive every number the response states.

## Check the response in about two minutes (CPU only)

```bash
pip install numpy scipy
python3 audits/replay_query_rules.py   # exact replay of Stream acquisition on stored Grams (about 90 s)
python3 audits/check_claims.py         # re-derives every number in the response, prints PASS/FAIL per claim
```

`check_claims.py` exits with status 0 only if every check passes. It currently runs 508
checks.

`CLAIMS.md` lists every claim in the response with its source file. It also marks the
claims that quote the submitted PDF itself (Tables 8, 9, 15, 18, 25, 26 and 27, and
App. G); these are not re-derived.

## Layout

| Path | Contents |
|---|---|
| `paretotrace/stream/` | **ParetoTrace-Stream.** See the table below. |
| `paretotrace/direct_language/` | **ParetoTrace-Direct for transformers.** Activation–adjoint factor capture, exact factor contraction (token-pair and parameter-row orders), and an online variant that retires factors. |
| `paretotrace/direct_celeba/` | Shared-input Direct: one batched reverse pass collects all task adjoints. |
| `paretotrace/direct_qm9/` | Direct for the QM9 graph network. |
| `paretotrace/controls/` | Matched controls: adapted structured-NTK contraction orders and the GhostSuite column adapter. |
| `experiments/` | The drivers, exactly as executed, for each study below. |
| `results/` | Outputs of each study, the stored Grams used for replay, and the historical run records cited in the response. |
| `audits/` | Independent re-analysis: claims checker, replay, event-level summaries. |

The Stream modules:

| File | Contents |
|---|---|
| `stable_recovery.py` | Original query rule. |
| `stable_recovery_v2.py` | Target-aware query rule. |
| `v46_baseline_frontier.py` | Wolfe simplex QP. |
| `shared_native_actions.py` | Autodiff Gram actions. |

## Studies and where they appear in the response

| Study | Response | Driver | Results |
|---|---|---|---|
| CelebA-40 training, 20 paired seeds | P8RF W1/Q1, 6wXk Q1 | `experiments/celeba_training/` | `results/celeba_training/` |
| QM9, seven methods, three seeds | P8RF Q1, 6wXk Q1 | `experiments/qm9/` | `results/qm9/` |
| Matched recovery benchmark on Pythia-160M and Pythia-1B, plus the 1B training-update costs | P8RF Q8, 6wXk Q2, ZNzr Q1 | `experiments/language_benchmark/`, `paretotrace/` | `results/language_benchmark/` |
| **Pythia-1B training** (six rules, three paired seeds, frozen protocol, 36 runs) | 6wXk Q2, P8RF Q1, General response (3/3) | `experiments/language_training_1b/` | `results/language_training_1b/` |
| Live CelebA recovery at fixed states | P8RF Q2/Q3/Q7, ZNzr Q2 | `experiments/recovery_study/` | `results/recovery_study/` |
| Live CelebA training, sharp vs scalar endpoint, 12 runs | P8RF Q7, 6wXk Q3 | `experiments/sharp_training/` | `results/sharp_training/` |
| **Target-aware Stream** (live timing and replay) | All threads | `experiments/query_rule/`, `audits/replay_query_rules.py` | `results/query_rule/` |
| **Target-aware Stream, live training** (3 seeds × 300 updates, paired with the 12-run study) | P8RF Q7, 6wXk Q3, ZNzr Q2 | `experiments/target_aware_training/` | `results/target_aware_training/` |
| **Per-update guarantee studies** (CelebA-40: study A with frozen heads, study B with six arms × 3 seeds) and uniform's responses along training | General response (1/3, 2/3), P8RF Q2/Q4, 6wXk Q1 | `experiments/certified_step/`, `audits/uniform_along_training.py` | `results/certified_step/` |
| **Pythia-1B under the verified step** (study C: five arms and one companion × 3 seeds, frozen protocol with two logged amendments, 18 reporting and 27 development runs) | General response (1/3, 2/3), P8RF Q1/Q4, 6wXk Q2 | `experiments/certified_1b/` | `results/certified_1b/` |

## The per-update guarantee

**The guarantee comes from the verified step.**
- It rests on Proposition 1, with the exact margin of Theorem 2 and Corollary 4, and the acceptance test of
  Equation (27): the test is tried only when every exact first-order response is positive.
- Any weighting can be audited this way for one Gram action. MGDA's role is narrower: it is the weighting the test
  never has to refuse for lack of a descent direction, because in exact arithmetic its margin is positive at every
  non-stationary state.
- Scope: the guarantee covers each update's minibatch, in floating point. It bounds no held-out or population
  loss and does not cover momentum or Adam.

Studies A and B below use CelebA-40, three seeds, 300 plain-SGD updates per arm, and the data order of the earlier studies.
- *Fixed step:* 0.02, the learning rate selected for the momentum-trained CelebA study, which is untuned for
  plain SGD.
- *Certified step:* it starts at 0.02 and halves, up to 20 trials, until Equation (27) holds for all 40
  attributes. Otherwise the parameters are held.
- *Primary endpoint* (fixed in each protocol): certified exact MGDA minus certified uniform, final worst-attribute
  training BCE.

**Study A: private heads frozen, three seeds.** A uniform-trained model is fine-tuned, so every attribute depends
only on the shared trunk. Driver: `experiments/certified_step/protected_finetune_v2.py`. Results:
`results/certified_step/protected_v2/`, summarized by `certified_step_summary_v2.py` and
`analyze_certified_step_v2.py`. Starting worst-attribute training BCE: 0.587.

| Arm | Refused | Updates raising a loss | Worst-attribute train BCE | Worst-attribute test accuracy |
|---|---|---|---|---|
| Uniform, fixed | n/a | 899/900 | 0.566 | 68.1% |
| Exact MGDA, fixed | n/a | 0/900 | 0.569 | 68.6% |
| Uniform, certified | 892/900 | 0 | 0.586 | 68.5% |
| Exact MGDA, certified | 0/900 | 0 | 0.569 | 68.6% |
| Stream, certified | 0/900 | 0 | 0.569 | 68.6% |
| MGDA-UB, certified | 894/900 | 0 | 0.587 | 68.5% |

- **Exact MGDA.** It kept the full step at 878 of 900 updates and ended within 0.0002 BCE of its fixed-step run.
- **Endpoint and audits.** The prespecified endpoint is −0.018 [−0.021, −0.014], with exact MGDA lower on 3 of 3 seeds. All 54
  audits passed. In this version the head first-order term is zero everywhere, including inside the audit.
- **Held-out losses.** No arm protects them: 7 to 12 of 40 attributes per seed end with a higher validation BCE than
  at the start.
- **Environment.** This study ran on a fresh runtime after the first one was reclaimed. The CelebA tensors were
  rebuilt from the pinned dataset revision and are byte-identical, with every split SHA-256 and the manifest hash
  matching. The uniform development model was retrained with the frozen code and seed. Its validation
  worst-attribute BCE is 0.638, against 0.636 for the original.
- **Interrupted first run.** An earlier version (`protected_finetune.py`, results in
  `results/certified_step/protected_seed62001/`) completed four arms of seed 62001 before that runtime was lost.
  It agrees: uniform certified was refused at every update, exact MGDA's fixed step raised no loss, and uniform's
  raised one at 299 of 300. Its recorded audit flags test head terms that the frozen design never uses, which is
  why version 2 corrects the audit.

**Study B: private heads trained from a fresh initialization, three seeds.**
Driver: `experiments/certified_step/certified_step_training.py`.

| Arm | Refused | Updates raising a loss | Worst-attribute train BCE | Worst-attribute test accuracy |
|---|---|---|---|---|
| Uniform, fixed | n/a | 777/900 | 0.604 | 65.3% |
| Exact MGDA, fixed | n/a | 900/900 | 1.778 | 43.2% |
| Uniform, certified | 2/900 | 0 | 0.640 | 61.5% |
| Exact MGDA, certified | 0/900 | 0 | 0.653 | 59.8% |
| Stream, certified | 0/900 | 0 | 0.653 | 59.8% |
| MGDA-UB, certified | 0/900 | 0 | 0.655 | 59.6% |

- **Uniform certifiability.** The private heads' gradients turned a negative trunk response positive at 634 of
  uniform's certified updates, so uniform stayed certifiable.
- **Primary endpoint:** +0.013 [+0.002, +0.023], with uniform lower on 3 of 3 seeds.
- **Audits:** all 54 passed.
- **Stream:** certified Stream matched exact MGDA to within 0.0002 BCE, at 0.58 GiB against 8.04 GiB, with no
  fallback.

**Uniform's first-order responses along training.**
Code and data: `audits/uniform_along_training.py`, `audits/extract_uniform_responses.py` and
`results/certified_step/uniform_responses_along_training.npz`.
- **Training runs.** Every Stream recovery spends its first Gram action on the uniform weighting. Over the 2,700
  updates of the nine Stream training runs, the uniform shared-trunk direction would have raised at least one
  attribute's loss to first order at 2,690, a median of 8 of 40.
- **Stored Grams.**
  - CelebA, 33 distinct states: it would raise one at all 33.
  - CIFAR-10 at M = 100, the 8 stored Grams in this repository: at all 8.
  - The 103 recorded Grams: at 2.

**Environment.** Studies A and B shared the A100 with a Pythia-1B job, so their timings are not compared with any
other study.

**Study C: Pythia-1B trained under the verified step, three seeds.** Every parameter is shared by the eight tasks,
so there are no private heads. Code: `experiments/certified_1b/` (`certified_language_1b.py` runs one arm and seed,
`certified_1b_driver.py` runs the stages and the freeze, `run_certified_1b.py` is the VM runner,
`analyze_certified_1b.py` the analysis). Results: `results/certified_1b/`, with `protocol.json` and `analysis.json`.
- *Setup.* Plain SGD on all parameters, batch 32, 200 updates per arm, with the seeds (49201 to 49203) and batch
  sequences of the Pythia-1B campaign below. The 14 campaign sources the study imports are byte-identical to the
  frozen campaign files.
- *Arms.* Uniform at a fixed step (unprotected), and four rules under the verified step: uniform, exact MGDA
  (Direct), audit-first (uniform when its one-action audit shows every response positive, otherwise exact MGDA,
  after Corollary 4 and Algorithm 1) and FAMO.
- *Verified step.* If every exact first-order response is positive, take the largest η0·2^-k, k < 20, with
  L_i(after) ≤ L_i(before) − c·η·s_i for all eight tasks, otherwise hold. Equation (27) uses c = 1/2, and any
  0 < c < 1 keeps strict decrease on every task.
- *Selection.* On development seed 49100 (100 updates, validation split only), each arm chose η0 from 1e-3, 3e-3
  and 1e-2, and each certified arm c from 1/2 and 1e-4, by final validation worst-task answer cross-entropy (ties
  to c = 1/2, then the smaller η0). The protocol records every candidate, the selection and two amendments, both
  logged after development runs and before any reporting run. The first added c = 1e-4. The second
  (`extension_plan.json`) extended the fixed step's grid from 1e-3, the campaign's tuned rate, to 3e-3 and 1e-2, and
  the fixed step chose 1e-2, the largest. A companion run of audit-first at c = 1/2 (its best η0, 1e-3) gives the
  descriptive contrast in c.
- *Primary endpoint* (fixed in the protocol): certified audit-first minus certified uniform, final held-out
  worst-task answer cross-entropy, paired by seed.

| Arm | c, η0 | Refused | Held | Updates raising a loss | Worst-task test CE | Mean test CE | Seconds per update |
|---|---|---|---|---|---|---|---|
| Uniform, fixed | n/a, 1e-2 | n/a | n/a | 577/600 | 0.685 | 0.503 | 2.01 |
| Uniform, certified | 1e-4, 3e-3 | 448/600 | 1 | 0 | 1.069 | 0.543 | 2.34 |
| Exact MGDA, certified | 1e-4, 3e-3 | 0/600 | 0 | 0 | 0.927 | 0.455 | 3.13 |
| Audit-first, certified | 1e-4, 3e-3 | 0/600 | 1 | 0 | 1.050 | 0.477 | 3.84 |
| FAMO, certified | 1e-4, 3e-3 | 394/600 | 0 | 0 | 1.523 | 0.666 | 2.34 |
| Audit-first, certified (companion) | 1/2, 1e-3 | 0/600 | 3 | 0 | 1.044 | 0.446 | 3.87 |

Held-out answer cross-entropy, means of three seeds. Medians of the update time, which includes the step rule and
excludes the audits. The fixed step's time includes the loss evaluations that record each task's change (0.8 s of
its 2.0 s), which a plain uniform update (1.20 s, in the benchmark above) skips. Paired differences, with descriptive Student-t 95% intervals (df = 2):

- **Primary endpoint:** −0.019 [−0.931, +0.892], audit-first lower on 2 of 3 seeds, not separated. Uniform passed
  the audit at 128 of 600 updates.
- **Prespecified secondaries:** exact MGDA minus uniform, both certified, −0.142 [−0.426, +0.143], exact MGDA lower
  on 3 of 3. FAMO minus uniform +0.454 [+0.048, +0.860].
- **Descriptive, not prespecified:** audit-first minus exact MGDA +0.123 [−0.511, +0.757], and the two bullets below
  on the fixed step.
- **Price of the guarantee:** for the same direction, certified uniform minus the fixed step +0.383 [+0.044, +0.723]
  (higher on 3 of 3). Certified exact MGDA minus the fixed step, +0.241 [−0.201, +0.684] in worst-task CE
  (higher on 3 of 3) and −0.048 [−0.231, +0.135] in mean CE. Outside the 45 audited updates the certified arms
  peaked at 15.7 GiB per update, as Direct's plain update does, against 12.7 GiB for the fixed step. The untimed
  audits peaked at 25.8 GiB.
- **The constant** (prespecified as descriptive): audit-first at c = 1e-4 minus c = 1/2, +0.006 [−0.531, +0.542]. On the development seed,
  c = 1e-4 gave the lower validation worst-task loss at all 12 pairs of rule and starting step. On the reporting
  seeds, audit-first's validation worst-task loss after 100 updates was 0.805 at c = 1e-4 against 1.066 at c = 1/2
  (lower on 3 of 3), and its median accepted step 0.0015 against 0.00025.
- **Exact MGDA:** all 1,477 exact-MGDA updates (Direct and audit-first's fallbacks) took a verified step. Every
  hold was on a uniform-weighted update.
- **The unprotected step** raised some task's minibatch loss at 577 of 600 updates. The largest single-task rise
  per such update had median 0.40 and exceeded 1 at 106 updates (83 after update 10), and the largest of all took a
  task from 14.5 to 51.6 (seed 49203, update 2).
- **Held-out losses** are not protected: in the four certified arms of the table, 32 of 96 per-task validation
  losses rose between updates 100 and 200.
- **Audits:** 45 independent autodiff audits of the responses that decide the gate (one task gradient at a time),
  all passed, with maximum relative error 8e-6.
- **Against the campaign** (descriptive, same seeds and batches, different step rules): certified exact MGDA minus
  the campaign's unprotected Direct (rate 1e-3, the top of that grid), −0.100 [−0.226, +0.027], lower on 3 of 3.
- **Provenance.** Three driver versions ran, each released: `v1/` for the smoke and c = 1/2 development runs, `v2/`
  for the c = 1e-4 development runs, and the top-level driver for the extension and every reporting run. The study
  script changed once, to take c as an argument, and `check_claims.py` confirms that diff line by line. Every
  reporting run records the frozen protocol's hash and the hashes of its sources. No run failed.

## The improvement: target-aware Stream

On conflicting tasks, the original rule's cost is set by its 0.95 response gate (a descent
target), not by the energy tolerance ε. In the 12-run training study, the gate decided 887
of 889 accepted stops. Target-aware Stream changes two things:

1. **It decouples the two targets.** Energy is certified by the sharp completion bound
   L_sk. Common descent is certified by the exact margin, with strict descent as the
   default target.
2. **It changes the query rule once energy is certified.** After the energy test passes,
   it queries the coordinate of the task with the smallest exact response.

The lower bound is valid for any transcript, so changing queries cannot affect validity.
With `query_rule='completion'`, `stable_recovery_v2.py` reproduces the original rule
exactly: same action counts, bit-identical weights.

**Live results.** On CelebA-40, an A100 at four held-out states with independent native
audits:

- Recovery time falls from 1.85 s to 1.08 s.
- Memory is unchanged at 0.54 GiB.
- Every direction passes the audit, every task descends, and the energy stays within 2.1%
  of optimal.

**Live training results.** Over 900 CelebA-40 updates, paired by seed with the 12-run study:

- 894 updates certified before full rank, at a mean 19.5 of 40 actions.
- 1.14 s per update, against 1.79 s for the original rule.
- The energy test alone decided 611 stops, against none under the 0.95 gate.
- All nine audits passed, with every task descending.
- Final test accuracy shows no quality difference that three seeds can resolve.

**Replay results.** On stored Grams, the replay reproduces all 60 live action counts. Under
the strict-descent target, the sharp endpoint needs 37%, 59% and 29% fewer actions than
the scalar endpoint, on CelebA, CIFAR-10 groups and the paper's 103 recorded Grams
respectively. The scalar endpoint gains nothing from the relaxed target: at ε = 0.05 it is
equivalent to the margin test m ≥ 0.976·U.

## Pythia-1B training (new)

Six training rules on Pythia-1B-deduped with eight tasks: Direct (exact MGDA), uniform, CAGrad, MGDA-FA, periodic
(held weights refined by one damped exact step every five updates) and FAMO. Plain SGD without momentum, weight decay
or clipping, batch 32, full-parameter training. The protocol (`results/language_training_1b/protocol.json`) was frozen
before the campaign. Each rule received the same learning-rate search: 1e-4, 3e-4 and 1e-3 on a separate development
seed for 100 updates, selected by validation worst-task cross-entropy with no test access. The selected rate was then
run for 200 updates on three paired reporting seeds, which share their batch sequences across rules. All 36 runs are
reported.

| Rule | Selected rate | Final worst-task CE | Final mean CE | Validation-selected worst-task CE |
|---|---|---|---|---|
| Uniform | 1e-3 | 0.774 | 0.375 | 0.680 |
| Periodic | 1e-3 | 0.781 | 0.391 | 0.734 |
| MGDA-FA | 3e-4 | 0.818 | 0.534 | 0.848 |
| CAGrad | 1e-3 | 0.846 | 0.389 | 0.769 |
| FAMO | 1e-3 | 0.878 | 0.378 | 0.796 |
| Direct | 1e-3 | 1.026 | 0.438 | 1.026 |

Held-out answer cross-entropy, means of three seeds. The frozen primary endpoints are paired differences in final
held-out worst-task cross-entropy, with descriptive Student-t 95% intervals (df = 2):

- Direct minus uniform: +0.252 [+0.167, +0.337], uniform lower on 3 of 3 seeds.
- Periodic minus MGDA-FA: -0.037 [-0.464, +0.390], periodic lower on 1 of 3 seeds, not separated.

Five rules chose the largest rate in the grid. Direct's validation-selected checkpoint was its last (update 200) on
every seed, and uniform's was update 100 on every seed. The study measures bounded-budget adaptation, not convergence.
It did not use the verified step, so it speaks to training quality only. Study C (above) trains the same model
under the verified step with the same seeds and batches.

**The guarantee at 1B.** Every parameter is shared across the eight tasks, and the Direct runs logged the exact
8×8 Gram of the task gradients at every update. Along the three confirmatory Direct runs, uniform's first-order
response was negative for some task at 401 of 600 updates (311 by more than 1% of the mean Gram diagonal), so the
verified step's gate would refuse uniform there. Exact MGDA's margin was positive at all 600 (below 1e-4 of that
scale at 3 near-stationary updates). These are first-order checks along Direct's trajectory, not runs of the
verified step, which study C runs. The records are in `results/language_training_1b/runs/*direct*/updates/`, hash-bound to each
run's `metrics.json`.

**Provenance.** `experiments/language_training_1b/` holds the 26 frozen campaign sources byte for byte (their hashes
are in `results/language_training_1b/campaign_sources.json`). The campaign driver validated every run, including full
verification of each final model archive. The frozen analyzer then refused to run, because it pinned
`language_training_methods.py` to the version before the CAGrad solver fix. The campaign ran, and froze, the fixed
version, which differs only in `cagrad_weights`. The analysis was produced by
`experiments/language_training_1b/corrected_analyzer/analyze_language_training.py`, which differs from the frozen
analyzer only in that one hash. Every other check ran as frozen, against the unmodified campaign directory, and no
statistic or endpoint changed. `audits/check_claims.py` confirms the one-line difference, re-derives every number above
from the 36 `runs/*/metrics.json` files and compares them with `campaign_analysis.json`.

## What is not included

- **Datasets and model checkpoints.** The drivers download the data at pinned revisions,
  with hashes recorded in the protocols.
- **Model checkpoints of study C**, which saves none.
- **The 36 trained Pythia-1B models** (about 4 GB each). The analyzer verified them on the training
  machine, so it cannot be re-run from this repository. The checker re-derives every stated number from the
  per-run metrics instead.
- **Upstream third-party code (GhostSuite, LibMTL, FAMO, CAGrad).** It is pinned by commit
  inside the drivers.

## Notes

- All GPU studies ran on one A100-SXM4-40GB.
- Hashes recorded inside result files refer to the files as produced. In two benchmark
  summaries, a local filesystem path was replaced by a repository-relative one.
- Floating-point stopping decisions are checked by independent native-gradient audits.
  They are not outward-rounded certificates for neural autodiff.
