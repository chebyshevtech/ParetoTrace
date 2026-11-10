# Claims in the author response and where they are checked

Run `python3 audits/check_claims.py` from the repository root. Each line it prints names a
check. The groups below map the response's statements to those checks and to their source
files.

## Re-derived from files in `results/` (all checked by `audits/check_claims.py`)

| Response statement | Source | Check names (prefix) |
|---|---|---|
| CelebA-40 arm means, paired differences, 95% t-intervals, seed counts, seed SDs, 13× and 14× cost ratios | `results/celeba_training/celeba_paired_analysis.json` | `celeba ...` |
| QM9 normalized MAE, seconds per update, primary difference and CI, seed counts, 380 A100-hours, systems-pilot costs, 110,831 training graphs | `results/qm9/paired_analysis.json`, `results/qm9/joint_preflight.json` | `qm9 ...` |
| Pythia-160M matched benchmark: medians, memory, the 1.19× and 1.22× ratios, the 2.5×/9.4× and 2.2×/5.7× ratios against native serial, native batched, pass counts on the stricter check | `results/language_benchmark/pythia160m_summary.json` | `160M ...` |
| Pythia-1B benchmark and training-update costs (1.92, 1.99, 4.51, OOM, 1.20, 1.60, 1.6×, 1.2×) | `results/language_benchmark/pythia1b_summary.json`, `training_preflight.json` | `1B ...` |
| Pythia-1B training: six rule means (final worst-task CE, mean CE, validation-selected worst-task CE), the two primary contrasts with 95% t-intervals and seed counts, selected rates (five at the largest), validation-selected checkpoints, frozen-source hashes, the one corrected analyzer pin | `results/language_training_1b/campaign_analysis.json`, `results/language_training_1b/runs/*/metrics.json`, `experiments/language_training_1b/` | `1B training ...` |
| Pythia-1B first-order responses along the Direct runs (supplementary, the response now cites study C instead): uniform negative for some task at 401 of 600 updates (311 beyond 1% of the Gram scale), exact MGDA's margin positive at all 600 | `results/language_training_1b/runs/*direct*/updates/*.json` | `1B first-order ...` |
| Live CelebA recovery study: q\* range, uniform trunk responses, transfer factors, 120 accepted runs, actions per state, worst factor, memory, Armijo checks | `results/recovery_study/results.json` | `q* ...`, `uniform ...`, `transfer ...`, `adaptive ...`, `Armijo ...` |
| Live CelebA training: primary endpoint (sharp Stream minus each arm, means and 95% t-intervals), 889 and 890 early stops, 11 and 10 fallbacks (all below 0.1% of the median energy bound), Stream peaks of 8.08 GiB on the seeds with 1 and 10 fallbacks and 0.58 GiB on the third, mean actions, energy-first medians, gate-decided stops, 1.8–3.5% and 2.4–2.8% savings, 36 audits, test accuracies, time and memory | `results/sharp_training/*.json` | `sharp ...`, `scalar ...`, `training ...`, `stream ...` |
| Target-aware Stream, live: 1.85 → 1.08 s, 41% by medians, 37% by paired geometric mean, actions, audits, descent, 2.1%, 0.54 GiB, about 2× Direct | `results/query_rule/query_rule_summary.json` (from `query_rule_study.json`) | `live ...` |
| Target-aware Stream, replay: 7%/24%/3% fewer actions; sharp vs scalar under strict descent 37%/59%/29%; 33 distinct CelebA states; all 60 live action counts predicted | `results/query_rule/replay_results.json` (from `audits/replay_query_rules.py`) | `replay ...` |
| Target-aware Stream, live training: 894 early stops, 6 fallbacks, 19.5 actions, 1.14 s vs 1.79 s, 611/211/72 decided stops, 9 audits with descent, 0.58 GiB, test accuracies and seed gaps | `results/target_aware_training/*.json` | `target-aware ...`, `energy test ...`, `descent target ...` |
| Per-update guarantee: uniform's shared-trunk direction raises some attribute at 2,690 of 2,700 live training updates (median 8 of 40), 2,673 executed Stream margins positive, 27 exact fallbacks, stored Grams CelebA 33/33, CIFAR-10 8/8, recorded 2/103 | `results/certified_step/uniform_responses_along_training.npz`, `results/grams/`, `results/recovery_study/`, `results/sharp_training/` | `along training ...`, `stored ...` |
| Study A (frozen heads, three seeds), with the A and B primary endpoints' 95% t-intervals: refusals and uniform's 8 passes, updates raising a loss, worst-attribute train BCE and test accuracy, full steps kept, certified vs fixed within 0.0002, held-out rises 7 to 12, endpoint −0.018, 54 audits, plus the interrupted first run on seed 62001 | `results/certified_step/protected_v2/protected_v2_summary.json`, `results/certified_step/protected_seed62001/*_metrics.json` | `study A ...`, `first run ...` |
| Certified-step study: updates raising an attribute's minibatch loss per arm, status counts (uniform's 898 passes, 3,598 of 3,600 accepted), MGDA-UB's full first-order change descending at all 900 updates, worst-attribute training BCE and test accuracy, accepted step sizes, primary endpoint, audits | `results/certified_step/certified_step_summary.json` | `certified-step ...` |
| Study C, Pythia-1B under the verified step: the five arms' and the companion's final worst-task and mean test CE, refusals and updates raising a loss, the primary and secondary contrasts with 95% t-intervals and seed counts, the 0.24 and 0.38 prices and lower mean CE, audit-first's 128 uniform updates, certified exact MGDA below the six-rule study's Direct on 3 of 3 seeds, the c contrast and median accepted steps, 12 of 12 development pairs and 1.018 → 0.707, 1,477 exact-MGDA updates all verified, uniform's 152 gate passes, 14.5 → 51.6, 32 of 96 validation rises, 3.1 s, 3.8 s and 2.0 s (0.8 s of it loss evaluations), 15.7 and 12.7 GiB outside the audits and 25.8 GiB audited, 0.805 against 1.066 after 100 updates, the 100-update development runs, 45 audits, the grids and two amendments, Equation (27) re-verified at every accepted step, selection reproduced from development validation, sources bound to the frozen protocol | `results/certified_1b/protocol.json`, `results/certified_1b/runs/*/metrics.json` and `updates/*.json`, `results/certified_1b/analysis.json`, `experiments/certified_1b/` | `1B verified step ...` |
| Block-4 uniform controls: descends on all nine panels, objective factor 1.15–4.26, every factor and gain in the table | `results/historical/uniform_block4_controls.json` | `uniform descends ...` |
| App. G.2 fixed-step run: 22 early stops, 4–10 actions, U/L ≤ 1.049, q/q\* ≤ 1.031, no false stop, slack ≤ 2.9% of U (median 0.16%), 16 actions per step including the audit | `results/historical/g2_six_arm_run.json` | `G.2 ...` |

## Quoted from the submitted PDF (not re-derived here)

These numbers come from the submission itself:

- **Table 25:** 59/60 accepted steps, losses 65.91, 5.45, 257.10 and 246.34, median step 5.86×10⁻⁷, 48/60 non-increasing, and the audited uniform arm accepted at all 60 updates.
- **Table 26:** held-out worst-task losses 77.48 (certified step), 6.40 (same direction, tuned step) and 248.33 (audited uniform, certified step).
- **Equation (27):** c = 1/2, and for 0 < c < 1 the test is satisfiable at small η exactly when every response is positive (p. 40).
- **App. G.1 (p. 41):** Table 25's uniform arm was not gated, and one of its accepted steps had a negative response, which Equation (27) then does not protect.
- **Table 9:** TorchJD at 1.49×, OOM at 32 objectives.
- **Tables 8 and 15:** memory of Stream relative to contraction.
- **Table 27 and App. G.3:** 54 vs 79 actions, 2.42 vs 3.81 s, the 1.24× speedup.
- **Table 2:** 55 → 28 actions.
- **Section 6.2:** 14 → 11 actions.
- **App. G:** 19 early stops, 37 replay certifications, median slack 7.1×, energy about 10⁻³ after step 60, 6.24 s.
- **C.5:** a pass costs 126–166 columns, and Equation 24.
- **C.6.6 and C.9.1:** factors 5.07 and 1.257.
- **Table 18:** the two-seed CelebA run.
