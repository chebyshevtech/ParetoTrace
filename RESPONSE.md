=== POST: General response (1/4), to all reviewers: the per-update guarantee

**In brief.** We thank the three reviewers. All three credit the theory. The shared practical question is whether exact MGDA recovery is worth running when uniform weighting matches or beats MGDA on held-out averages (P8RF W1, 6wXk W1). Our answer is a guarantee rather than an average. Where no objective's loss on the current batch may rise at any update, as for a protected safety or fairness loss, a verified step (the certified arms in our tables) provides that guarantee and a held-out average cannot. MGDA is the weighting the verified step never has to refuse for lack of a descent direction, and ParetoTrace computes or certifies it where native per-task differentiation runs out of memory. Two new studies show when that matters and what it costs.

**The guarantee in theory.**
- For smooth losses, a positive exact margin $m=\min_i(K\lambda)_i$ admits a step that raises no task loss on the minibatch (Proposition 1). Equation (27), tried only when every exact response is positive, finds and verifies such a step or holds the parameters. Any weighting can be audited for one Gram action (Corollary 4), so the guarantee belongs to the verified step, as our §6.3 says.
- MGDA is the weighting it never refuses for lack of a descent direction. For every unit $u$ and $\lambda\in\Delta$, $\min_i g_i^\top u\le(G\lambda)^\top u\le\Vert G\lambda\Vert$, while $g_i^\top G\lambda^\star\ge q^\star$ for every $i$. So $G\lambda^\star$ has the largest worst-task rate of descent per unit step, $\sqrt{q^\star}$, and in exact arithmetic its margin is positive at every non-stationary state. Under one smoothness constant $\beta$ and with $m>0$, the best step this bound licenses guarantees every task a decrease of $m^2/(2\beta q)$, which is largest at MGDA, where it equals $q^\star/(2\beta)$, since $m\le\lambda^{\star\top}K\lambda\le\sqrt{q^\star q}$.
- ParetoTrace reads the margin of its in-span weights at no extra action (Theorem 2). It computes MGDA exactly at Pythia-1B, where native per-task differentiation ran out of memory in every configuration we tried (adapted structured-NTK contraction also fits, 1.03× slower).

**What the new studies show** (CelebA-40, details in 2/4).
- *Private heads frozen*, so every attribute depends only on the shared trunk (three seeds): the verified step refused uniform at 892 of 900 updates and MGDA-UB at 894, so neither could train under it. Exact MGDA was never refused and kept the full 0.02 step at 878 of 900 updates, and even unverified its step raised no attribute's minibatch loss, against 899 of 900 updates for uniform's.
- *Private heads trained*: their own gradients kept uniform certifiable at 898 of 900 updates, so there the direction did not decide certifiability, and uniform made slightly more progress under the guarantee.
- *No verified step*: with trained heads, the fixed step raised some attribute's minibatch loss at 777 of 900 uniform updates, and at all 900 exact-MGDA updates, where it broke training (2/4).
- *In practice*: any weighting's margin costs one action (Corollary 4), so a run can audit uniform first and recover exact MGDA only when uniform would be refused, as Algorithm 1 does for a held weighting. Uniform passed at 8 and 898 of 900 updates in the two studies.

**Scope.** The guarantee is per update, on that minibatch, and in floating point. It bounds no held-out or population loss. It chains across updates only for an objective scored on the same fixed set at every update, which then never rises during the run. It does not cover momentum or Adam. None of our training-quality studies used it. Its price can be large: in Table 26 the certified arm's held-out worst-task loss was 77.48, against 6.40 for the same direction at the tuned step, though audited uniform under the same test reached 248.33. We have not run a safety benchmark, so we claim the property, not a safety outcome. Code, logs and a checker for every number are at https://github.com/chebyshevtech/ParetoTrace.

=== POST: General response (2/4), the two guarantee studies (new)

Both studies use CelebA-40, three seeds, 300 plain-SGD updates per arm and the data order of our other training studies. The trunk moves along $-G\lambda$. Each private head moves along its own unweighted gradient unless frozen. The fixed step is 0.02, the learning rate selected for our momentum-trained CelebA study, so it is untuned for plain SGD. The certified step is tried only when every exact first-order response (private heads included) is positive. It takes the first of $0.02\cdot2^{-k}$, $k<20$, that satisfies Equation (27) for all 40 attributes, and otherwise holds. An accepted step cannot raise a loss by construction, so these studies measure refusals and cost. Each fixed its primary endpoint in advance: certified exact MGDA minus certified uniform in final worst-attribute training BCE.

*A. Private heads frozen (three seeds).* A model trained with uniform weighting (1200 updates) is fine-tuned, so every attribute depends only on the shared trunk. Starting worst-attribute training BCE 0.587.

| Arm | Refused | Updates raising a loss | Worst-attr. train BCE | Worst-attr. test acc. |
|---|---|---|---|---|
| Uniform, fixed | n/a | 899/900 | 0.566 | 68.1% |
| Exact MGDA, fixed | n/a | 0/900 | 0.569 | 68.6% |
| Uniform, certified | 892/900 | 0 | 0.586 | 68.5% |
| Exact MGDA, certified | 0/900 | 0 | 0.569 | 68.6% |
| Stream, certified | 0/900 | 0 | 0.569 | 68.6% |
| MGDA-UB, certified | 894/900 | 0 | 0.587 | 68.5% |

Of the arms we ran, only the MGDA directions could train under the guarantee: uniform's and MGDA-UB's trunk directions would raise some attribute's loss at nearly every update. For exact MGDA the guarantee cost almost nothing. It kept the full step at 878 of 900 updates and ended within 0.0002 BCE of its fixed-step run. Unprotected uniform reached a slightly lower worst attribute by ignoring the guarantee. No arm protected held-out losses: 7 to 12 of 40 attributes per seed ended with a higher validation BCE than at the start. The prespecified endpoint was −0.018, exact MGDA lower on 3 of 3 seeds, and all 54 audits passed. The starting model was retrained on a fresh runtime with the frozen code, seed and byte-identical data (validation worst-attribute BCE 0.638, against 0.636 originally). An interrupted first run of four arms on one seed agreed.

*B. Private heads trained from a fresh initialization.*

| Arm | Refused | Updates raising a loss | Worst-attr. train BCE | Worst-attr. test acc. |
|---|---|---|---|---|
| Uniform, fixed | n/a | 777/900 | 0.604 | 65.3% |
| Exact MGDA, fixed | n/a | 900/900 | 1.778 | 43.2% |
| Uniform, certified | 2/900 | 0 | 0.640 | 61.5% |
| Exact MGDA, certified | 0/900 | 0 | 0.653 | 59.8% |
| Stream, certified | 0/900 | 0 | 0.653 | 59.8% |
| MGDA-UB, certified | 0/900 | 0 | 0.655 | 59.6% |

The heads' own gradients turned a negative trunk response positive at 634 of uniform's certified updates, so uniform stayed certifiable and made slightly more progress (primary endpoint +0.013, uniform lower on 3 of 3 seeds). Certification moved uniform's worst-attribute test accuracy from 65.3% to 61.5%. Certified exact MGDA reached 59.8%, against 64.7% for momentum-trained Direct at 300 updates (reply to P8RF Q7). The fixed step broke exact MGDA (worst-attribute BCE above $\ln2$), so that row is not a fair baseline. Certified Stream matched exact MGDA within 0.0002 BCE at 0.58 GiB against 8.04 GiB, with no fallback. Means over seeds, differences computed before rounding.

=== POST: General response (3/4), other new results

*Terms.* "Energy" is the MGDA objective $q(\lambda)=\Vert G\lambda\Vert^2$. The "response gate" is a descent target on the exact responses $(K\lambda)_i$. The "sharp endpoint" is the completion lower bound $L_{\rm sk}$ of Corollary 1, and the "scalar endpoint" is the squared-margin bound $\max(m,0)^2/U$. A Stream "fallback" is an exact recovery charged to Stream when the certificate cannot be resolved numerically.

**New results**

- **CelebA-40, 20 paired seeds (P8RF Q1).** Primary endpoint, Direct − MGDA-UB in worst-attribute test accuracy: +0.9 points [−4.1, +5.8], 9/20 seeds, not separated. Direct beats MGDA-UB on mean accuracy (19/20), mean balanced accuracy (19/20) and cross-entropy (20/20), but trails uniform on worst-attribute accuracy (−5.1 [−8.6, −1.7]) and cross-entropy. We retract the ordering of Section 6.1 and Table 18.
- **QM9, seven methods (P8RF Q1).** Over 2 epochs and 3 seeds, exact MGDA beats MGDA-UB on all three seeds (not significant), is level with FairGrad and Aligned-MTL, and trails uniform and FAMO.
- **Pythia-1B training (6wXk Q2).** Six rules, three paired seeds, one learning-rate search each under a frozen protocol. Direct − uniform in held-out worst-task cross-entropy is +0.252 [+0.167, +0.337], uniform lower on 3/3 seeds, so exact MGDA does not train a better 1B model at this budget.
- **Matched cost (P8RF Q8, ZNzr Q1).** Direct is 1.19–1.22× faster than the best adapted structured-NTK contraction on Pythia-160M and 1.03× at 1B. With eight tasks at 1B an exact update costs 1.2× MGDA-FA or FAMO, and native per-task differentiation ran out of memory on a 40 GB A100 in every configuration we tried. With 40 tasks on CelebA's small trunk it costs 13× MGDA-UB.
- **Improved Stream: target-aware acquisition (all reviewers).** The new studies show that on conflicting tasks Stream's cost is set by its descent target, not by ε. So Stream now certifies the energy with the sharp endpoint and common descent with the exact margin, with strict descent ($\min_i(K\lambda)_i>0$) as the default target. Once the energy test passes, it queries the task with the smallest response. Validity is unchanged, since the lower bound holds for any transcript. Live on CelebA-40 (four held-out states, independent native audits), recovery time falls from 1.85 s to 1.08 s at 0.54 GiB (41% by medians, 37% by paired geometric mean over states), with every task descending and energy within 2.1% of optimal. The scalar endpoint cannot exploit this, because at ε = 0.05 it equals the margin test $m\ge0.976U$. In replay on 33 distinct CelebA states, 8 stored CIFAR-10 Grams at $M=100$ and the 103 recorded Grams, the sharp endpoint needs 37%, 59% and 29% fewer actions than the scalar one. Replay reproduces the action counts of all 60 live Stream recoveries exactly. Live training confirms it: over 900 CelebA-40 updates (three paired seeds), target-aware Stream certified 894 before full rank at a mean 19.5 of 40 actions and 1.14 s per update, against 29.7 actions and 1.79 s for the original rule, and every audited update descended. The sharp endpoint now decides most stops: 611 of 894 alone, against none under the 0.95 gate.
- **Live CelebA-40 training (P8RF Q7, 6wXk Q3).** Under the original 0.95 target, Stream certified 889 of 900 updates before full rank (scalar endpoint: 890), with all 36 independent audits passing and all 11 fallbacks at near-stationary states.
- **New theorem (6wXk Q4, ZNzr Q2).** The completion query rule closes the interval in at most $\min(M,\operatorname{rank}R+1)$ exact actions, where $R$ is the geometry not yet contracted (exact rank). Statement and proof sketch are in our reply to 6wXk.

=== POST: General response (4/4), corrections and changes

**Corrections to the submission**

1. **CelebA ordering (§6.1, C.9.2, Table 18).** Retracted. Table 18 used two seeds and a 4k split. On 20 seeds and the full test set, Direct − uniform is −5.1 points [−8.6, −1.7]. The MGDA arms vary far more on this metric (seed SD 7–8 points, against 1.3 for uniform) than the two-seed spread suggested.
2. **Efficiency headline.** The 14.9× figure leaves the abstract, replaced by the matched benchmark.
3. **Table 1 provenance.** Blocks 2 and 6 record their rules in run outputs: the positive-part scalar bound with a 0.95 gate, and completion with a numerical slack. Block 6's arm named Direct acquired every column with identity actions, so the Table 26 caption and G.2's "Direct contracts all sixteen blocks" are withdrawn. Block 5 ran a periodic heuristic with no certificate. For blocks 1 and 4 only source code survives, so their Stream rows will be marked uncertified, not relabeled, and B.5's statement that block 4's gate is checkable in the released record is withdrawn. Block 5's abstract claim is withdrawn, including "all three seeds", which included the tuning seed.
4. **Margin gate (remark after Corollary 1, A.5.7, B.5, C.1).** The positive-part bound is valid for either sign of $m$ and is what the code computes. A.5.7 refutes only the untruncated square of C.1. The gate is a descent target, not a validity condition.
5. **App. G.** Its 19 live early stops occur at non-positive measured energy and are disjoint from the 37 replay certifications. The abstract will say that this run made no early stop at a state float32 resolves, and report the new live results instead.
6. **C.6.6 and C.9.1.** The 5.07 and 1.257 factors score weights optimal for one minibatch on another, not the spread of $q^\star$, which is larger still (about 700× across five CelebA minibatches). C.6.6's conclusion stands with corrected evidence.
7. **Trunk scope.** The MGDA-UB audit, and our new uniform and transfer checks, measure the shared-trunk direction. Private heads also descend on their own tasks. With heads trained, MGDA-UB's full first-order change descended on every attribute at all 900 certified updates of study B. We will state this wherever these results appear, the abstract included, with the full first-order change.
8. **Abstract.** The 60/60 result will appear only beside the tuned-step comparison (Table 25).
9. **Table 25.** Its 16 actions per step include the full-rank audit's columns, which are outside the update.
10. **Algorithm 1.** On CelebA at batch 128 its across-update rule would rarely keep a weighting, so §5's description will be scoped to the agreeing regime.

**Additions.** The new studies, including the certified-step study, target-aware Stream, the residual-rank theorem, an implementation section for Direct and its controls, one decision rule among Direct, native differentiation and Stream, and a map from each main result to the step it justifies and the experiment that tests it (posted in our reply to ZNzr). The introduction and conclusion will state the per-update guarantee, credit it to the verified step, say what MGDA adds, and state its scope.

**Scope.** MGDA optimizes the current minibatch's common-descent direction, not held-out loss. Exact recovery beats MGDA-UB on CelebA's mean metrics, but trails uniform on worst-attribute accuracy and cross-entropy, and on held-out worst-task cross-entropy at Pythia-1B. Direct was the fastest exact route in most of our settings. Native differentiation was cheaper on QM9 and close on CelebA. Stream wins on time only at large M with agreeing gradients, and otherwise trades about 2× Direct's time for 93% less memory when it does not fall back to Direct.

=== POST: Response to Reviewer P8RF (1/6)

**In brief.** We ran the experiments asked for. Exact MGDA beats MGDA-UB on every CelebA mean metric, but the primary worst-attribute endpoint did not separate and uniform is better on it, so we retract the §6.1 ordering. The case for exact recovery is a per-update guarantee rather than an average. A verified step keeps every task's minibatch loss from rising, and MGDA is the weighting it never has to refuse for lack of a descent direction. With frozen private heads on CelebA-40 the step refused uniform at 892 of 900 updates and exact MGDA at none, while with trained heads uniform passed at 898 (general response). The new studies also exposed what sets Stream's cost on conflicting tasks, the descent target. Relaxing it to strict descent, with target-aware queries, cuts live recovery time from 1.85 s to 1.08 s. Every headline the reviewer questioned is withdrawn or relabeled.

**W1 / Q1. Does exact recovery improve a trained model?**

*CelebA-40 (new, 20 paired seeds).*

- **Arms.** ParetoTrace-Direct itself, not 40 serial backward passes, against MGDA-UB and uniform weighting.
- **Setup.** 1200 steps, ImageNet ResNet-18 trunk with 40 heads, batch 128, SGD with momentum 0.9, a 16k official-train subset and the full official test set (19,962 images).
- **Tuning.** Each arm chose its learning rate on a separate development seed from {0.005, 0.02, 0.08} by validation worst-attribute accuracy. All three chose 0.02.
- **Endpoint.** Fixed in advance: paired Direct − MGDA-UB in final worst-attribute test accuracy. All 60 runs are reported.

| Test metric, mean of 20 seeds | Direct | MGDA-UB | Uniform |
|---|---|---|---|
| Worst-attribute accuracy (%) | 62.6 | 61.7 | 67.7 |
| Mean accuracy (%) | 86.9 | 85.6 | 87.6 |
| Mean balanced accuracy (%) | 71.6 | 67.3 | 70.0 |
| Mean cross-entropy | 0.304 | 0.360 | 0.280 |
| Seconds per update | 0.49 | 0.038 | 0.034 |

Paired differences, with 95% t-intervals and seeds favoring the first arm (computed before rounding):

- **Primary, Direct − MGDA-UB, worst-attribute accuracy:** +0.9 points [−4.1, +5.8], 9/20. Not separated.
- **Direct − MGDA-UB, secondary:** mean accuracy +1.3 [+0.8, +1.8], 19/20. Mean balanced accuracy +4.3 [+3.2, +5.5], 19/20. Cross-entropy lower on 20/20.
- **Direct − uniform:** worst-attribute accuracy −5.1 [−8.6, −1.7], 3/20. Mean accuracy −0.7 [−1.0, −0.4], 1/20. Mean balanced accuracy +1.7 [+1.0, +2.3], 18/20.

With the twenty seeds C.9.2 estimated, the primary endpoint still did not separate: the paired standard deviation was about 10 points, far above the two-seed spread C.9.2 assumed. The spread comes from the MGDA arms, which on some seeds push one attribute down to about 40% accuracy (seed SD 7–8 points, against 1.3 for uniform). Uniform has the best worst-attribute accuracy and cross-entropy. That reverses Table 18 (two seeds, 4k split, serial exact MGDA), and we retract its ordering. Exact MGDA beats its cheap surrogate on every mean metric and has the best balanced accuracy of the three. Direct costs 13× more per update than MGDA-UB here.

=== POST: Response to Reviewer P8RF (2/6)

*QM9 (new).* We ran LibMTL's QM9 benchmark with pinned upstream code and split: 11 targets, 110,831 training graphs. Seven methods took part, Direct, MGDA-UB, uniform, Nash-MTL, FairGrad, Aligned-MTL and FAMO, each with LibMTL's default settings and the same Adam learning rate, for two full-data epochs over three paired seeds. Normalized MAE is each target's MAE in units of its standard deviation, averaged over targets. The standard 300 epochs would need about 380 A100-hours for this suite, so this is an early-training comparison, not a convergence ranking.

| QM9, final test | Mean normalized MAE (lower is better) | Seconds per update |
|---|---|---|
| Uniform | 0.303 | 0.056 |
| FAMO | 0.312 | 0.074 |
| Aligned-MTL | 0.327 | 0.275 |
| FairGrad | 0.337 | 0.310 |
| **Direct** | **0.339** | 0.455 |
| Nash-MTL | 0.414 | 0.536 |
| MGDA-UB | 0.477 | 0.070 |

- **Primary, Direct − MGDA-UB:** −0.138 [−0.442, +0.167]. Better on 3/3 seeds, not significant with three seeds.
- **Against Nash-MTL:** better on 2/3 seeds.
- **Against FairGrad and Aligned-MTL:** level (+0.002 and +0.012).
- **Against uniform and FAMO:** worse on 3/3 seeds (+0.036 and +0.027).

On this small graph network a native per-task route is cheaper than Direct (0.32 s and 2.9 GiB per update, against 0.48 s and 31 GiB, in our systems pilot), so QM9 speaks to training quality only.

*What we conclude.* Exact recovery changes the trained model measurably. It beats MGDA-UB on every CelebA mean metric and on all three QM9 seeds (not significant), but MGDA itself is not the best training rule on these benchmarks. This matches C.1's Pythia-160M result and our new matched Pythia-1B study (reply to 6wXk, Q2): in both, uniform has lower worst-task cross-entropy than Direct on all three seeds. So the case for exact recovery is not a better average. It is the per-update guarantee: a verified step, which any weighting can attempt for one Gram action, never has to refuse MGDA for lack of a descent direction, and with frozen private heads on CelebA it refused uniform at 892 of 900 updates (general response). Elsewhere uniform passes: at 898 of 900 updates with trained heads, where it made slightly more progress than MGDA, and at all 60 in Table 25, where it lowered the worst task far less (246.34 against 65.91). When a practitioner needs that guarantee, we show how to compute or certify MGDA and what it costs. That cost depends on the setting. With eight tasks at Pythia-1B an exact update costs 1.2× MGDA-FA or FAMO, and native per-task differentiation ran out of memory (reply to 6wXk, Q2). With 40 tasks on CelebA's small trunk it costs 13× MGDA-UB, and on QM9 native differentiation is cheaper. We will revise the abstract and introduction accordingly, and replace C.6.6's "null training result" with these results.

=== POST: Response to Reviewer P8RF (3/6)

**Q2. Uniform on the block-4 panels, and a conflicting setting where Stream beats Direct.**

From retained records of the block-4 panel family, with the same generator, objective count and seed and mean cosine matching to 5e-5. The native Grams were not stored, so these are reconstructions. Objective factor is $q(\mathbf 1/M)/q^\star$, and gain is $\min_i (K\mathbf 1/M)_i/\sqrt{q(\mathbf 1/M)\,q^\star}$.

| $M$ | seed 0 factor / gain | seed 1 | seed 2 |
|---|---|---|---|
| 512 | 2.90 / 0.29 | 1.15 / 0.97 | 1.16 / 0.96 |
| 1024 | 3.45 / 0.11 | 1.17 / 0.97 | 1.18 / 0.96 |
| 2048 | 4.26 / 0.13 | 1.68 / 0.61 | 1.19 / 0.97 |

Uniform descends on all nine panels, with objective factor 1.15 to 4.26, so there Stream buys fidelity to MGDA, not descent.

Apart from the $M=512$ CIFAR-10 descent result the reviewer cites, we have no conflicting setting where uniform fails to descend and Stream beats Direct on wall-clock time. On CelebA-40 with the private heads frozen, the verified step refused uniform at 892 of 900 updates because its direction would raise some attribute's loss, and on the shared trunk alone uniform's direction would do so at 2,690 of 2,700 updates of our nine live Stream training runs. Stream's advantage there is memory:

| Live CelebA-40 recovery, four held-out states | Seconds | Peak GiB |
|---|---|---|
| Direct | 0.53 | 8.04 |
| Native per-task differentiation, best memory | 0.54 | 2.09 |
| Stream, original rule (0.95 target) | 1.85 | 0.54 |
| Stream, target-aware (strict descent) | 1.08 | 0.54 |

Medians over three timing repeats at the same four states, including all actions, solves and reconstruction. Native differentiation was timed in our first run on these states, the other rows in the second. The abstract will state this boundary.

**Q3. What does a 5% certificate buy on CelebA-40, and does a noise-calibrated ε change Stream's cost?**

A correction first. The 1.257 (C.9.1) and 5.07 (C.6.6) factors are $q_{K_i}(\lambda^\star_{K_j})/q^\star_{K_i}$, weights optimal for minibatch $j$ scored on minibatch $i$. The optimum varies even more. Across five CelebA minibatches at batch 128, $q^\star$ ranges from 0.00055 to 0.38, because some minibatches are nearly Pareto-stationary. Over the twelve pairs of held-out minibatches the median transfer factor is 39 (C.9.1 measured 1.257 at batch 512), and every transferred vector raises some task's loss to first order on the shared trunk.

A 5% certificate therefore does not estimate a population quantity. It certifies the update taken: $\Vert G\lambda-G\lambda^\star\Vert^2\le0.05\,q^\star$ on the current minibatch, a relative direction error of at most $\sqrt{0.05}\approx22\%$. With the descent target, every task also decreases to first order on that minibatch. That is the step stochastic MGDA prescribes.

ε does not set the cost here: actions at the four held-out states are 33, 36, 23 and 31 for every ε from 0.01 to 0.25. The descent target does. Relaxing it from the 0.95 gate to strict descent, with target-aware queries, cuts live recovery time from 1.85 s to 1.08 s, while every audited direction keeps its energy within 2.1% of optimal. C.6.6 saw the same on CIFAR-10 groups, where a looser ε did not help because descent bound.

=== POST: Response to Reviewer P8RF (4/6)

**Q4. The 60/60 safeguard and its cost in progress.**

Agreed, and the abstract will report both numbers together. In Table 25, 59 of 60 updates accepted a step, and one refused and held the parameters. The safeguarded run reached worst-task training loss 65.91, against 5.45 for the same direction at the tuned step, and held-out worst-task loss 77.48 against 6.40 (Table 26). The guarantee is non-increase of every task loss on the current minibatch. It does not cover progress, the population loss or momentum. Under the same test the recovered direction lowered the worst task from 257.10 to 65.91 while audited uniform reached 246.34, so the direction matters there, but the test is costly.

*The same rule on CelebA (new).* Table 25's acceptance test (Equation (27) with $c=1/2$, halving the step), tried only when every exact response is positive and started from 0.02, costs much less on CelebA-40 (general response, 2/4). With private heads trained, the certified arms accepted 3,598 of 3,600 updates at a median step of 0.00125, 16× below the fixed 0.02, and certified uniform reached worst-attribute training BCE 0.640 against 0.604 at the fixed step. With the private heads frozen, it kept exact MGDA's full step at 878 of 900 updates, so the certified run ended within 0.0002 BCE of the fixed-step run, while it refused uniform at 892 of 900. So the cost of the guarantee depends on the setting as much as on the rule.

*An intermediate rule.* We did not test another rule. Any $c\in(0,1)$ keeps strict decrease, and a smaller $c$ accepts at least as large a step on the same grid, possibly much larger than the 2× change in the smoothness bound suggests, since acceptance depends on the realized losses. A non-increase test with a step floor is another candidate. The dataset revision behind Table 25 is no longer available, so we will test both on the maintained revision for the final version and claim nothing about them now.

**Q5. The periodic comparison.**

- **Rule.** Block 5 is MGDA-FA plus one exact Gram action and a damped exact Frank–Wolfe step every fifth update, with no stopping certificate or margin gate. Cadence 5 and damping 0.5 were chosen on one development seed. That is 20 scheduled refreshes per run. The applied-refresh count was not logged.
- **Timing.** 0.206 s per update is the amortized cost of this heuristic at $M=32$. The 6.24 s in App. G is the median per-update cost of Stream's certified recovery at $M=16$ in a separate run. They differ in method, stopping rule and accounting.
- **"Strongest cheap baseline."** Unsupported, given FAMO in block 6 and uniform and CAGrad in block 2. The label and the periodic result leave the abstract.

Our Pythia-1B study tuned MGDA-FA on the same development seed as every other rule and added FAMO, CAGrad and uniform. Its periodic arm holds its weights between refreshes instead of running MGDA-FA's recursion, a close variant of block 5's rule. Periodic − MGDA-FA was −0.037 [−0.464, +0.390], periodic lower on 1/3 seeds, and uniform was best (reply to 6wXk, Q2). The rule is outside the certified method, so we withdraw the claim. Our new comparisons use equal tuning: one learning-rate search for every CelebA arm, and one published learning rate for every QM9 method.

**Q6. Which endpoint and gate ran in each block?**

| Block | Stopping rule | Margin gate | Evidence |
|---|---|---|---|
| 1 | Completion bound in source (geometry replay) | none | source only |
| 2 | Positive-part scalar bound | $m\ge0.95U$ | run output |
| 3 | Full contraction (Direct) | n/a | run output |
| 4 | Completion bound in source, scalar test in the paper's label | $m\ge0.95U$ | source only |
| 5 | Periodic heuristic | none | source only |
| 6 | Completion with numerical slack | full-rank fallback | run output |
| 7 | Completion vs Frank–Wolfe at fixed queries | n/a | replay by design |

Block 5 ran a heuristic with no certificate (Q5). For blocks 1 and 4 the gate cannot be recovered, because no execution transcript was saved, so B.5's statement that block 4's gate is checkable in the released record is withdrawn. As the reviewer proposes, those Stream rows will be marked uncertified, and we do not relabel them. On validity, the positive-part bound $\max(m,0)^2/U$ is valid for either sign of $m$. A.5.7 refutes only the untruncated square written in C.1, so B.5's statement that the gate is required for validity is wrong.

=== POST: Response to Reviewer P8RF (5/6)

**Q7. Does the live certificate ever stop early away from Pareto-stationarity?**

In App. G's run, no, and the reviewer's reading is right. In every other live run we have, yes. App. G's 19 early stops all occur where float32 makes the measured energy non-positive, so they are disjoint from the 37 replay certifications. Its guard subtracted a slack proportional to the Gram scale. The run approached stationarity, with median energy about $10^{-3}$ after step 60, while the Gram scale did not fall, so the slack was a median 7.1× the upper endpoint. Elsewhere:

- **Fixed-step Stream arm, App. G.2.** The algorithm stops before full rank on 22 of 60 updates, after 4 to 10 of 16 actions. Table 25's 16 actions per step also count the full-rank audit's columns. Each stop has positive energy and lower bound and $U/L\le1.049$, and the full-Gram audit confirms each at $q/q^\star\le1.031$, with no false stop. That run stayed far from stationarity, and the same guard's slack was at most 2.9% of $U$ (median 0.16%).
- **Real CIFAR-10 groups, App. G.3 ($M=100$).** Over 15 states, completion needed a median 54 actions and 2.42 s, against 79 and 3.81 s for the scalar endpoint, and was better on 14 of 15.
- **Live CelebA training (new, three paired seeds, 300 updates per arm, primary endpoint fixed in advance).** Four arms shared initialization and data order: Direct, native differentiation, and Stream with the sharp or the scalar endpoint, both under the 0.95 target. Stream stopped before full rank on 889 of 900 updates with the sharp endpoint (scalar: 890), after a mean 29.7 of 40 actions, with 11 fallbacks (scalar: 10), all at near-stationary states. All 36 native audits passed. Mean final worst-attribute test accuracy was 65.5% (sharp), 63.2% (scalar), 64.7% (Direct) and 64.7% (native). The primary endpoint, sharp Stream minus each arm in that accuracy, was +0.8 points [−3.5, +5.2] against Direct, +0.8 [−1.2, +2.9] against native and +2.3 [−2.0, +6.6] against scalar, none separated. Direct and native compute the same direction and differ by up to 2.5 points on a seed, so we claim no ranking.

That study also shows why the sharp endpoint saved only 2–3% there. At ε = 0.05 the scalar test $U\le1.05\max(m,0)^2/U$ is the margin test $m\ge0.976U$, so under a 0.95 gate both endpoints wait for the margin. The sharp endpoint met the energy target at a median 20 actions, but the gate decided 887 of 889 stops. Relaxing the target to strict descent removes that coupling, and then only the sharp endpoint can stop early. In replay on 33 distinct CelebA states it needs 20.2 actions against 32.1 for the scalar endpoint, 37% fewer. On CIFAR-10 groups at $M=100$ the reduction is 59%, and on the 103 recorded Grams 29%. Live, the target-aware rule takes 1.08 s against 1.85 s for the original rule.

- **Target-aware Stream in live training (new, with the same seeds, initialization and data order).** 894 of 900 updates stopped before full rank, after a mean 19.5 of 40 actions. The energy test alone decided 611 stops, the descent target 211, and both together 72. Update time fell to 1.14 s from 1.79 s. All nine audits passed with every task descending, and the 6 fallbacks were near-stationary. Mean final worst-attribute test accuracy was 62.4%: within 2.6 points of the exact arms on two seeds and 4.3–6.8 points below them on the third. With three seeds we claim no quality effect either way.

These are floating-point decisions checked by independent audits, not outward-rounded certificates for neural autodiff.

=== POST: Response to Reviewer P8RF (6/6)

**Q8. Comparison with ghost-Gram and structured-NTK computation.**

We built a matched benchmark of complete recovery (capture, Gram, simplex solve, emitted direction). The controls are both structured-NTK orders (token-pair, parameter-row) and an automatic choice (structured-auto), GhostSuite's product kernels (the code of Wang et al., 2025) at a pinned revision adapted to task-mean columns, and native serial and batched differentiation. All arms share precision, keep tied-weight cross terms, select their configuration on development batches, and are compared only where the direction matched an independent native reference to $2\times10^{-5}$ on every trial.

| Pythia-160M | Direct | Best structured control | GhostSuite adapter | Native serial |
|---|---|---|---|---|
| 8 objectives | 0.376 s, 4.11 GiB | 0.446 s, 4.11 GiB (1.19×) | 0.634 s, 4.11 GiB | 0.957 s, 9.11 GiB |
| 32 objectives | 0.373 s, 4.11 GiB | 0.457 s, 4.11 GiB (1.22×) | 1.712 s, 4.11 GiB | 3.497 s, 23.6 GiB |

These are pooled medians of nine timed trials, and ratios use unrounded medians. Direct beats the best structured control on every batch. Native batched differentiation took 1.006 s and 4.244 s, qualifying on 6 of 9 trials. TorchJD was not rerun: in Table 9 it was 1.49× slower than Direct at 8 objectives and out of memory at 32. This is not a rerun of Table 9. Both sizes use 32 examples, so the 32 objectives are single examples, while Table 9 split a larger batch. Direct's cost depends on the examples, not their grouping, while native differentiation pays one backward pass per objective. The GhostSuite arm computes one column at a time and is not an optimized all-pairs implementation.

A stricter check required agreement with two FP32 native references at once. These references disagree at that tolerance on some trials, and pass rates were erratic: native serial passes 3/9 at 8 objectives, structured-auto 3/9 and 9/9, GhostSuite 9/9 and 0/9, Direct 3/9 at both sizes. We read this as reference noise near the tolerance and make no speed claim under it.

*Pythia-1B (new, 8 objectives).* Direct 1.92 s at 15.7 GiB, best structured order 1.99 s (1.03× from unrounded medians), GhostSuite adapter 4.51 s. Native differentiation is out of memory in every configuration tried. Every contraction route passes every check at 1B, the stricter one included.

**W2. Headline claims.** Agreed. The abstract and Table 1 changes are listed in our general response. Every headline that rests on one seed, a replay, an unrecorded rule or asymmetric tuning is removed or labeled where it is stated. On the sharp endpoint's role: under the original 0.95 target it decided few live stops, as the reviewer suspected. Under the target-aware Stream it decides the stop: it needs 37% fewer actions than the scalar endpoint on CelebA, and in live training it alone decided 611 of 894 stops (Q7).

=== POST: Response to Reviewer 6wXk (1/3)

**In brief.** On our benchmarks exact MGDA is not a better generalization rule than uniform, and we now say so. Its case is a per-update guarantee. A verified step keeps every task's minibatch loss from rising, and MGDA is the weighting it never has to refuse for lack of a descent direction. With frozen private heads on CelebA-40 the step refused uniform at 892 of 900 updates and exact MGDA at none, while with trained heads uniform passed at 898. Exact recovery is affordable at Pythia-1B, where native differentiation ran out of memory. A redesigned safeguard stops early on 889 of 900 live updates, and strict descent with target-aware queries cuts live recovery from 1.85 s to 1.08 s.

**W1 / Q1. Why does more accurate recovery not give better generalization on Pythia-160M?**

MGDA optimizes a different objective. It returns the minimum-norm point of the convex hull of the current minibatch's task gradients, the direction that best guarantees common first-order descent on that minibatch. Nothing in that objective targets held-out loss, so higher fidelity to it need not lower test cross-entropy. Unitary scalarization is also known to be a strong baseline in multi-task learning (Kurin et al., 2022, and Xin et al., 2022). We do not claim to know the cause of the Pythia result. What fidelity to MGDA does buy is a direction a per-update guarantee never has to refuse. A verified step keeps every minibatch loss from rising for any weighting it accepts, and held-out averages provide no such guarantee. Without it, a fixed step raised some attribute's minibatch loss at most updates of our new CelebA-40 studies, for uniform in both and for exact MGDA when the private heads were trained. With it, none rose, and with frozen private heads the step refused uniform at 892 of 900 updates while exact MGDA kept training (general response). The guarantee costs held-out accuracy, which we report there.

New, larger evidence shows that fidelity changes the trained model in a metric-dependent way. On CelebA-40 with 20 paired seeds:

- **Primary endpoint.** Direct − MGDA-UB in worst-attribute accuracy is +0.9 points [−4.1, +5.8], 9/20 seeds, not separated.
- **Against MGDA-UB.** Direct is better on mean accuracy (+1.3 points, 19/20), mean balanced accuracy (+4.3, 19/20) and cross-entropy (20/20).
- **Against uniform.** Direct trails on worst-attribute accuracy (−5.1 points), mean accuracy (−0.7) and cross-entropy, and leads on mean balanced accuracy (+1.7, 18/20).

The full table is in our reply to P8RF, where we also retract the submission's two-seed CelebA ordering. On QM9 (seven methods, 2 epochs, 3 seeds), exact MGDA beats MGDA-UB on all three seeds (not significant), is level with FairGrad and Aligned-MTL, and trails uniform and FAMO. So exact MGDA improves on its cheap surrogate, but on these benchmarks MGDA is not the best training rule. We will say this plainly.

**Q2. Efficiency and training on larger language models.**

*Efficiency at 1B (new).* Pythia-1B, eight tasks, batch 32, one 40 GB A100:

| Pythia-1B | Seconds | Peak GiB |
|---|---|---|
| Exact recovery, Direct | 1.92 | 15.7 |
| Exact recovery, best structured-NTK order | 1.99 | 15.7 |
| Exact recovery, GhostSuite column adapter | 4.51 | 15.7 |
| Exact recovery, native per-task differentiation | out of memory | |
| Training update, Direct / uniform | 1.92 / 1.20 | 15.7 / 12.7 |
| Training update, MGDA-FA / FAMO | 1.60 / 1.60 | 12.7 / 12.7 |

Recovery times are medians over three held-out batches. Update times are means of ten measured updates after warm-up, including the optimizer step, from a separate run on different batches. MGDA-FA's warm start uses one exact recovery and peaks at 15.7 GiB. With eight tasks at 1B an exact update costs 1.6× uniform and 1.2× FAMO or MGDA-FA, because the shared forward and backward dominate. With 40 tasks on CelebA's small trunk it costs 14× uniform.

*Training outcomes at 1B (new).* Six rules, plain SGD, 200 updates, three paired seeds. Each rule's learning rate was chosen from {1e-4, 3e-4, 1e-3} on a separate seed by validation worst-task loss, under a frozen protocol. All 36 runs are in the repository. Final held-out worst-task answer cross-entropy, mean of three seeds: uniform 0.774, periodic 0.781, MGDA-FA 0.818, CAGrad 0.846, FAMO 0.878, Direct 1.026. Primary endpoints: Direct − uniform +0.252 [+0.167, +0.337], uniform lower on 3/3 seeds, and periodic (held weights refined every five updates) − MGDA-FA −0.037 [−0.464, +0.390], periodic lower on 1/3, not separated. Direct also trails uniform on mean cross-entropy (0.438 against 0.375) and at validation-selected checkpoints (1.026 against 0.680). Five rules chose the largest rate, and Direct's validation worst-task loss was still falling at 200 updates in every seed, while uniform's was lowest at 100. So at this budget exact MGDA does not train a better 1B model, as on Pythia-160M (W1). This study did not use the verified step.

=== POST: Response to Reviewer 6wXk (2/3)

**W2 / Q3. Live certification versus replay, and how numerical safeguards affect it.**

The reviewer identifies a real gap, and our accounting in App. G was incomplete. Its 19 live early stops all occur where float32 makes the measured energy non-positive, and they are disjoint from the 37 replay certifications. Its guard subtracted a slack proportional to the Gram scale. The run approached stationarity, with median energy about $10^{-3}$ after step 60, while the Gram scale did not fall, so the slack was a median 7.1× the upper endpoint and withheld every early stop at a resolved state.

*What we changed.* The new guard drops the scale-proportional slack. A stop is accepted only with a positive lower bound and $U\le1.05L$, after checking that the completion reproduces the observed actions to relative residual $2\times10^{-5}$. Any update that fails a numerical check falls back to exact recovery, charged to Stream. Stops remain floating-point decisions, so we audit sampled updates against independently differentiated task gradients.

*What it did.* In new live CelebA-40 training (three paired seeds, 300 updates each), Stream certified 889 of 900 updates before full rank with the sharp endpoint and 890 with the scalar one. The 11 and 10 fallbacks all came at near-stationary states, where the exact MGDA energy was below 0.1% of its median, which is the regime float32 cannot resolve. All 36 independent audits passed (three per run). In App. G.2's run the old guard also passed 22 early stops, because there the energy stayed large and the slack was at most 2.9% of $U$. So the shortfall came from how the safeguard scaled, not from the endpoint.

*How it affects efficiency.* Under the original 0.95 descent target, the descent gate, not the safeguard or the endpoint, decided 887 of 889 live stops. With strict descent as the target and the target-aware query rule (reply to Q4), live recovery at fixed CelebA states takes 1.08 s instead of 1.85 s, with every task descending and energy within 2.1% of optimal. In live training (three seeds, 900 updates) it certified 894 updates early at 1.14 s per update, against 1.79 s for the original rule, with 6 near-stationary fallbacks and all audits passing.

*Next step.* Proposition 3 shows how to tie the guard to action error: if the action errors have operator norm at most $\delta$, both bounds move by computable multiples of $\delta$, and the relative test survives when $U_\delta\le(1+\epsilon)L_\delta$ with $L_\delta>0$. Our runs do not yet use these perturbed bounds, which need a measured bound on action error. We do not claim an outward-rounded certificate for neural autodiff.

=== POST: Response to Reviewer 6wXk (3/3)

**Q4. How should query vectors be chosen in practice?**

Use the rule the theory supplies, plus one improvement the new studies motivated.

*Theorem (residual-rank action bound).* Let $K=C+R$, where $C\succeq0$ is known (the geometry already contracted) and $R\succeq0$ is reached only through exact actions $v\mapsto Rv$, with $r=\operatorname{rank}R$. Initialize by querying $Ru$ for any $u\in\arg\min_{\Delta}u^\top Cu$. For the acquired span $\mathcal S$ with basis $Q$ and $Y=RQ$, let

$$A=C+Y(Q^\top Y)^\dagger Y^\top,\qquad L=\min_{\Delta}\mu^\top A\mu,\qquad U=\min_{\Delta\cap\mathcal S}\lambda^\top K\lambda.$$

While $L<U$, query the normalized off-span component of a lower minimizer $\mu$. With exact arithmetic and solves, $L=U$ after at most $\min(M,r+1)$ actions, including initialization. The returned weights are optimal for every PSD completion of the transcript, and $r$ need not be known.

*Proof sketch.* With $F=R^{1/2}$ and $T=F\mathcal S$, the shorted (Nyström) identity gives $A=C+FP_TF$, as in Theorem 4. If $L<U$ then $\mu\notin\mathcal S$, since otherwise its true and completed values coincide and $U\le L$. Adding $\mu$ either enlarges $T$, or leaves $A$ unchanged while making a weighting of true value $L$ feasible, which closes the interval. So every query that does not finish raises $\dim T$ by one.

*Scope.* The bound is in exact rank. Neural Grams are generically full rank, so on its own it gives $M$, which is vacuous for our live studies ($C=0$). It is informative for partial contraction, where $\operatorname{rank}R$ is at most the number of uncontracted parameters, which we have not run.

*Target-aware queries (new).* The completion rule chases the energy interval. When the descent target is what remains, querying the task with the smallest exact response serves it better. In replay this cuts actions by 7% on 33 distinct CelebA states, 24% on CIFAR-10 groups at $M=100$ and 3% on the 103 recorded Grams at the same 0.95 target, and live recovery time from 1.85 s to 1.72 s. Combined with strict descent as the target, live recovery takes 1.08 s. The rule is sequential, one query per action. A block version would suit the pass costs of C.5, where one pass costs as much as 126–166 columns, so a $P_a$-pass recovery of $k$ actions beats the full Gram only when $(P_a-1)(f/g)+k<M$ (Equation 24).

*Practical rule.* Time acquisition, solve, reconstruction and fallback for each candidate on a few development states, and take the fastest that fits memory (reply to ZNzr, Q2).

=== POST: Response to Reviewer ZNzr (1/3)

**In brief.** We do not claim the contraction identity: Direct is the full-information endpoint of our minimax framework. In a new matched benchmark it is 1.19–1.22× faster than adapted structured-NTK contraction on Pythia-160M and 1.03× at 1B, faster than the GhostSuite adapter, and 2.5–9.4× faster than native serial differentiation. With strict descent and target-aware queries, Stream now runs at about 2× Direct's time with 93% less memory on live CelebA when it does not fall back to Direct. We give one decision rule and post the theory-to-experiment map now. Code: https://github.com/chebyshevtech/ParetoTrace.

**W1 / Q1. What does Direct add beyond structured NTK contractions [1] (Novak et al., 2022) and ghost-gradient products [2] (Wang et al., 2025), and how does it compare?**

*Novelty claim.* We do not claim the contraction identity. Direct uses the same arithmetic as [1] and [2]: for an affine layer, $\langle g_i,g_j\rangle$ is a sum of products of activations and output adjoints. The paper's contribution is the conditional minimax characterization and the certified partial recovery built on it. Direct is the full-information endpoint of that framework (Theorem 4 with nothing left uncontracted).

*Implementation* (a new section in the final version):

1. aggregation by task inside the contraction, giving the $M\times M$ task Gram rather than an example-level kernel
2. a per-parameter choice between token-pair and parameter-row contraction by a FLOP estimate, which our structured-auto control also uses
3. embedding contraction restricted to the vocabulary rows actually observed
4. summation over every occurrence of tied or reused weights before the product, which all arms do
5. an online variant that retires factors after their last use
6. for shared-input tasks such as CelebA, one batched reverse pass that collects all task adjoints.

The controls share items 2 and 4, and we have not ablated which remaining item gives Direct its margin. We present that margin as an implementation result, not as the paper's contribution.

*Matched comparison.* Complete recoveries against both structured-NTK orders and an automatic choice, GhostSuite's product kernels adapted to task-mean columns, and native serial and batched differentiation, all at the same precision. Each arm's configuration was selected on development batches, and only directions matching an independent native reference to $2\times10^{-5}$ on every trial are compared.

| Pythia-160M | Direct | Best structured control | GhostSuite adapter | Native serial |
|---|---|---|---|---|
| 8 objectives | 0.376 s, 4.11 GiB | 0.446 s, 4.11 GiB (1.19×) | 0.634 s, 4.11 GiB | 0.957 s, 9.11 GiB |
| 32 objectives | 0.373 s, 4.11 GiB | 0.457 s, 4.11 GiB (1.22×) | 1.712 s, 4.11 GiB | 3.497 s, 23.6 GiB |

Ratios use unrounded medians. Both sizes use 32 examples, so the 32 objectives are single examples. At Pythia-1B, Direct takes 1.92 s against 1.99 s for the best structured order (1.03×) and 4.51 s for the GhostSuite adapter, and native differentiation runs out of memory. A stricter check against two FP32 native references at once gave erratic pass rates for every arm, including native differentiation (Direct 3/9), so we make no speed claim there (details in our reply to P8RF, Q8). The GhostSuite arm adapts per-column kernels and is not an optimized all-pairs implementation. The code, adapters, harness and run logs are at https://github.com/chebyshevtech/ParetoTrace.

=== POST: Response to Reviewer ZNzr (2/3)

**W2 / Q2. When should a practitioner choose Stream over Direct?**

*Rule.* Time the exact routes on a few development states and take the fastest that fits memory. Across our measurements the patterns are:

- **Direct** was the fastest exact route on CelebA-40 (0.53 s) and on CIFAR-10 groups at $M=100$ (0.14 s against 2.42 s for Stream).
- **Native per-task differentiation** was cheaper than Direct on QM9's small graph network and within 4% of Direct on CelebA-40 at a quarter of its memory (0.54 s, 2.09 GiB). It ran out of memory at Pythia-1B.
- **Stream** wins on time only at large $M$ with agreeing gradients. On the block-4 panels it beat contraction at $M\ge1024$ when the mean pairwise cosine was about 0.8 (1.97× at 2048). At cosine 0.63 it lost at 1024 and won by only 1.09× at 2048. On conflicting CIFAR-10 groups it won only with a strict-descent certificate at $M=512$. Elsewhere its advantage is memory.

*Memory.* Stream's memory is not uniformly lower. On live CelebA-40 it needed 0.54 GiB against Direct's 8.04 GiB, because it never holds all factors at once. On the block-4 panels it needed up to 16% more than contraction (Table 8), and on CIFAR-10 groups 19% more (Table 15). Its fallback should match the memory budget. In our training study the fallback used Direct, so two of three Stream runs peaked at 8.08 GiB (seeds with 1 and 10 fallbacks) and the third at 0.58 GiB. A memory-bound deployment should fall back to chunked native differentiation.

*Target-aware Stream (new).* The new studies showed that on conflicting tasks the descent target sets the cost. Stream now certifies energy with the sharp endpoint and common descent with the exact margin, and queries the task with the smallest exact response once the energy test passes. On live CelebA-40 (four held-out states, independent native audits):

| Method | Seconds | Peak GiB | Actions of 40 |
|---|---|---|---|
| Direct | 0.53 | 8.04 | 0 |
| Native per-task differentiation, best memory | 0.54 | 2.09 | 40 |
| Stream, original rule (0.95 target) | 1.85 | 0.54 | 23–36 |
| Stream, target-aware (strict descent) | 1.08 | 0.54 | 15–25 |

Native differentiation was timed in our first run on these states, the other rows in the second. Every Stream direction passed the audit, every task descended, and energy stayed within 2.1% of optimal. So Stream now costs about 2× Direct's time, down from 3.5×, for 93% less memory when it does not fall back to Direct. In live training it took 1.14 s per update against 0.50 s for Direct and 1.79 s for the original rule, at 0.58 GiB on every update that did not fall back. Our new theorem bounds the actions by $\operatorname{rank}R+1$ in exact rank, which is informative only for partial contraction. In practice the effective rank and the descent target decide the cost. We will replace the regime discussion of Sections 5–6 with this rule and these costs.

=== POST: Response to Reviewer ZNzr (3/3)

**W3. Presentation.** We agree that the submission makes theory, stopping rules and experiments hard to connect. The final version will open its experiments with this map:

| Result | What it gives | Where it is used | Where it is tested |
|---|---|---|---|
| Theorem 1 | Interval $[L_{\rm sk},U_{\rm sk})$ and minimax weights from any transcript | Stream's weights and stop | Block 7, live CelebA studies |
| Theorem 2, Corollary 4 | Exact worst-task response and certified gain at in-span weights | Descent target | New live studies, block 2 |
| Theorem 3 | A method correct on every Gram consistent with the transcript can certify no better than the minimax weights | Why the sharp interval is the right stopping quantity | A necessity result, not an experiment |
| Corollary 1, Equation (21) | Sharp endpoint, dominating the Frank–Wolfe gap and the scalar endpoint | Stopping test | Block 7 replay, live training, target-aware study |
| Theorem 4 | Recovery when part of the geometry is contracted, a single Gram when all of it is | Direct | Block 3, matched benchmark |
| Theorem 5, Proposition 4 | Additive regret when a trace budget bounds the completions | Trace-budget variant of the stop | 103 recorded Grams (C.6) |
| Proposition 1, Equation (27) | A positive margin admits a step that raises no task loss, and a test that finds it | Certified step | Tables 25 and 26, two new CelebA studies (general response, 2/4) |
| Proposition 3 | Bounds under inexact actions | Basis for a numerical guard | Not yet run live |
| New residual-rank theorem | At most $\operatorname{rank}R+1$ actions, exact rank | Completion query rule | Rule used live, bound vacuous at full rank |

The final version will also:

1. give the full specification of each method in one place: query rule, stopping rule, descent target and fallback
2. split Table 1 into recovery cost, recovery fidelity and training outcome, each block with one protocol, and state the executed stopping rule and its evidence for every Stream row
3. move results that rest on a single seed or replay alone out of the abstract
4. define every term on first use and keep the appendix to protocol detail.
