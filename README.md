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

`check_claims.py` exits with status 0 only if every check passes. It currently runs 293
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
| Live CelebA recovery at fixed states | P8RF Q2/Q3/Q7, ZNzr Q2 | `experiments/recovery_study/` | `results/recovery_study/` |
| Live CelebA training, sharp vs scalar endpoint, 12 runs | P8RF Q7, 6wXk Q3 | `experiments/sharp_training/` | `results/sharp_training/` |
| **Target-aware Stream** (live timing and replay) | All threads | `experiments/query_rule/`, `audits/replay_query_rules.py` | `results/query_rule/` |
| **Target-aware Stream, live training** (3 seeds × 300 updates, paired with the 12-run study) | P8RF Q7, 6wXk Q3, ZNzr Q2 | `experiments/target_aware_training/` | `results/target_aware_training/` |
| **Certified-step study** (CelebA-40, six arms × 3 seeds × 300 updates) and uniform's responses along training | General response (1/3), P8RF Q2/Q4, 6wXk Q1 | `experiments/certified_step/`, `audits/uniform_along_training.py` | `results/certified_step/` |

## The per-update guarantee

The response's main argument is that ParetoTrace supplies a certificate, at every update, that no objective's
loss rises on that update's minibatch. Held-out averages, where uniform weighting often wins, cannot supply it.
The theory is Proposition 1 with the exact margin of Theorem 2 and Corollary 4. Equation (27) is the test that
finds and verifies the step.

**Certified-step study** (`experiments/certified_step/`, results in `results/certified_step/`). CelebA-40, three
seeds, 300 plain-SGD updates per arm, with the initialization and data order of the sharp-training and
target-aware studies. Every response includes the private heads' own gradients.

- *Fixed step:* 0.02, the learning rate selected for the CelebA training study, at every update.
- *Certified step:* starts at 0.02 and halves, up to 20 trials, until Equation (27) holds for all 40 attributes.
  It is taken only when every exact first-order response is positive. Otherwise the parameters are held.

| Arm | Updates raising some attribute's minibatch loss | Worst-attribute training BCE | Worst-attribute test accuracy |
|---|---|---|---|
| Uniform, fixed step | 777/900 | 0.604 | 65.3% |
| Exact MGDA (Direct), fixed step | 900/900 | 1.778 | 43.2% |
| Uniform, certified step | 0/900 | 0.640 | 61.5% |
| Exact MGDA (Direct), certified step | 0/900 | 0.653 | 59.8% |
| Target-aware Stream, certified step | 0/900 | 0.653 | 59.8% |
| MGDA-UB, certified step | 0/900 | 0.655 | 59.6% |

- **Without a verified step.** Uniform's fixed-step updates raised at least one attribute's minibatch loss at 777 of 900 updates. Exact MGDA's direction descended on every attribute to first order at every update, yet its fixed step raised one at all 900 updates.
- **With the certified step.** No attribute's minibatch loss rose at any of the 3,598 accepted updates of the four certified arms. The certificate refused 2 uniform updates, whose exact response was not positive, and held the parameters there. The median accepted step was 0.00125.
- **Cost of the guarantee.** For uniform it cost 3.7 points of worst-attribute test accuracy. For exact MGDA it was a gain of 16.6 points.
- **Primary endpoint** (fixed before reporting): certified exact MGDA minus certified uniform in worst-attribute training BCE was +0.013, with uniform lower on 3 of 3 seeds. Certified Stream matched exact MGDA to within 0.0002, at 0.58 GiB against 8.04 GiB.
- **Uniform's full first-order response.** Along its own fixed-step trajectory, private heads included, it was not positive at 703 of 900 updates.

**Uniform's first-order responses along training** (`audits/uniform_along_training.py`,
`audits/extract_uniform_responses.py`, `results/certified_step/uniform_responses_along_training.npz`). Every
Stream recovery spends its first Gram action on the uniform weighting. Over the 2,700 updates of the nine Stream
training runs, the uniform shared-trunk direction would have raised at least one attribute's loss to first order
at 2,690, a median of 8 of 40. Every executed Stream direction had a positive certified margin (2,673 updates),
and the other 27 updates used exact recovery. The same computation on the stored Grams gives the following:

- CelebA (33 distinct states): uniform's direction raises some attribute at 33.
- CIFAR-10 groups (8 states): it raises some group at all 8.
- The 103 recorded Grams: it raises a task at only 2.

The driver `certified_step_training.py` imports the frozen target-aware training module and its retained
sources. On the A100 it ran in a directory assembled from `experiments/target_aware_training/`, with the protocol
in `results/certified_step/certified_step_protocol.json`. It shared the GPU with a Pythia-1B job, so its timings
are not compared with any other study.

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

## What is not included

- **Datasets and model checkpoints.** The drivers download the data at pinned revisions,
  with hashes recorded in the protocols.
- **Upstream third-party code (GhostSuite, LibMTL, FAMO, CAGrad).** It is pinned by commit
  inside the drivers.

## Notes

- All GPU studies ran on one A100-SXM4-40GB.
- Hashes recorded inside result files refer to the files as produced. In two benchmark
  summaries, a local filesystem path was replaced by a repository-relative one.
- Floating-point stopping decisions are checked by independent native-gradient audits.
  They are not outward-rounded certificates for neural autodiff.
