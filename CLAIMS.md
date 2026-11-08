# Claims in the author response and where they are checked

Run `python3 audits/check_claims.py` from the repository root. Each line it prints names a
check. The groups below map the response's statements to those checks and to their source
files.

## Re-derived from files in `results/` (all checked by `audits/check_claims.py`)

| Response statement | Source | Check names (prefix) |
|---|---|---|
| CelebA-40 arm means, paired differences, 95% t-intervals, seed counts, seed SDs, 13× and 14× cost ratios | `results/celeba_training/celeba_paired_analysis.json` | `celeba ...` |
| QM9 normalized MAE, seconds per update, primary difference and CI, seed counts, 380 A100-hours, systems-pilot costs | `results/qm9/paired_analysis.json`, `results/qm9/joint_preflight.json` | `qm9 ...` |
| Pythia-160M matched benchmark: medians, memory, the 1.19× and 1.22× ratios, the 2.5×/9.4× and 2.2×/5.7× ratios against native serial, native batched, pass counts on the stricter check | `results/language_benchmark/pythia160m_summary.json` | `160M ...` |
| Pythia-1B benchmark and training-update costs (1.92, 1.99, 4.51, OOM, 1.20, 1.60, 1.6×, 1.2×) | `results/language_benchmark/pythia1b_summary.json`, `training_preflight.json` | `1B ...` |
| Live CelebA recovery study: q\* range, uniform trunk responses, transfer factors, 120 accepted runs, actions per state, worst factor, memory, Armijo checks | `results/recovery_study/results.json` | `q* ...`, `uniform ...`, `transfer ...`, `adaptive ...`, `Armijo ...` |
| Live CelebA training: 889 and 890 early stops, 11 and 10 fallbacks, mean actions, energy-first medians, gate-decided stops, 1.8–3.5% and 2.4–2.8% savings, 36 audits, test accuracies, time and memory | `results/sharp_training/*.json` | `sharp ...`, `scalar ...`, `training ...`, `stream ...` |
| Target-aware Stream, live: 1.85 → 1.08 s, 41% by medians, 37% by paired geometric mean, actions, audits, descent, 2.1%, 0.54 GiB, about 2× Direct | `results/query_rule/query_rule_summary.json` (from `query_rule_study.json`) | `live ...` |
| Target-aware Stream, replay: 7%/24%/3% fewer actions; sharp vs scalar under strict descent 37%/59%/29%; 33 distinct CelebA states; all 60 live action counts predicted | `results/query_rule/replay_results.json` (from `audits/replay_query_rules.py`) | `replay ...` |
| Block-4 uniform controls: descends on all nine panels, objective factor 1.15–4.26 | `results/historical/uniform_block4_controls.json` | `uniform descends ...` |
| App. G.2 fixed-step run: 22 early stops, 4–10 actions, U/L ≤ 1.049, q/q\* ≤ 1.031, no false stop, slack ≤ 2.9% of U (median 0.16%), 16 actions per step including the audit | `results/historical/g2_six_arm_run.json` | `G.2 ...` |

## Quoted from the submitted PDF (not re-derived here)

These numbers come from the submission itself:

- **Table 25:** 59/60 accepted steps, losses 65.91, 5.45, 257.10 and 246.34, median step 5.86×10⁻⁷, 48/60 non-increasing.
- **Equation (27):** c = 1/2.
- **Table 9:** TorchJD at 1.49×, OOM at 32 objectives.
- **Tables 8 and 15:** memory of Stream relative to contraction.
- **Table 27 and App. G.3:** 54 vs 79 actions, 2.42 vs 3.81 s, the 1.24× speedup.
- **Table 2:** 55 → 28 actions.
- **Section 6.2:** 14 → 11 actions.
- **App. G:** 19 early stops, 37 replay certifications, median slack 7.1×, energy about 10⁻³ after step 60, 6.24 s.
- **C.5:** a pass costs 126–166 columns, and Equation 24.
- **C.6.6 and C.9.1:** factors 5.07 and 1.257.
- **Table 18:** the two-seed CelebA run.
