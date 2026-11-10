=== POST: General response (1/2): protecting the losses that matter

**Thank you.** All three reviews credit the theory, and their practical questions led to new results: a new theorem with a full proof (reply to 6wXk), a faster target-aware Stream, a matched recovery benchmark, a 20-seed CelebA study, QM9 against five methods, and three studies of training under the per-update guarantee, the largest on Pythia-1B. Code, logs, proofs and a checker that re-derives every number here are public at https://github.com/chebyshevtech/ParetoTrace.

**Accuracy is not everything.** The shared question is why recover exact MGDA when uniform weighting matches it on held-out averages (P8RF W1, 6wXk W1). Because an average can hide a loss that must not rise. When one objective is protected, such as a safety or fairness loss scored on a fixed set, a rule that wins on average while raising that loss is the wrong rule. The model drifts from what it must preserve and may later need correcting, for instance by further fine-tuning or RL pressure. Our method prevents the rise instead. The *verified step* (Proposition 1, Equation (27)) moves only when every task's exact first-order response is positive, and shrinks the step until every task's minibatch loss provably decreases, otherwise it holds. In exact arithmetic, exact MGDA is the weighting it never refuses: its margin is positive at every non-stationary state, and its direction has the largest worst-task rate of descent per unit step. ParetoTrace computes it exactly at Pythia-1B, where native per-task differentiation runs out of memory.

**Terms.** *Fixed step*: a constant learning rate with no check. *Certified uniform*, *certified exact MGDA*: that weighting under the verified step. *Refused*: the gate failed (some task's first-order response was not positive), so the parameters held. *Worst-task loss*: the largest held-out loss over the tasks.

**Training under the guarantee** (three seeds each, details in 2/2):

| Setting | Fixed-step uniform: updates raising some task's loss | Certified uniform: refused | Certified exact MGDA: refused |
|---|---|---|---|
| CelebA-40, private heads frozen | 899/900 | 892/900 | 0/900 |
| CelebA-40, private heads trained | 777/900 | 2/900 | 0/900 |
| Pythia-1B, every parameter shared | 577/600 | 448/600 | 0/600 |

- Designate any one of Pythia-1B's eight tasks as the protected loss: the fixed step raised it at 235 to 285 of 600 updates. No certified update raised any task's minibatch loss, in any of the three studies.
- When the conflict sits in shared parameters, certified uniform stalls, refused at most updates, while exact MGDA keeps training. At Pythia-1B certified exact MGDA reached held-out worst-task loss 0.927 against 1.069 for certified uniform, lower on 3 of 3 seeds. Our prespecified primary, audit-first (uniform when its audit passes, else exact MGDA) minus certified uniform, was −0.019 [−0.931, +0.892].
- Protection has a price, which we report: fixed-step uniform reached 0.685 at Pythia-1B by letting losses rise. The verified step pays only where a full step would raise a loss. On CelebA with frozen heads, exact MGDA kept the full step at 878 of 900 updates and ended within 0.0002 BCE of its fixed-step run.

**Scope.** The guarantee is per update, on that minibatch, in floating point, for plain SGD, and it does not bound held-out loss. We demonstrate the property in training at scale. We have not run a safety benchmark, so we claim the property, not a safety outcome.

=== POST: General response (2/2): the three guarantee studies and the revision

*Protocol.* Plain SGD. An update is refused unless every exact first-order response (private heads included) is positive. Otherwise it takes the first of $\eta_0\cdot2^{-k}$, $k<20$, that satisfies Equation (27) for every task, and holds if none does. Every primary endpoint was fixed before any run. Means over seeds, differences computed before rounding, 95% t-intervals.

*A. CelebA-40, private heads frozen* (300 updates, $\eta_0=0.02$, $c=1/2$, only the shared trunk moves). Only exact MGDA and Stream trained under the guarantee: uniform and MGDA-UB were refused at 892 and 894 of 900 updates. Primary, certified exact MGDA minus certified uniform in worst-attribute training BCE: −0.018 [−0.021, −0.014], exact MGDA lower on 3 of 3 seeds. All 54 audits passed.

*B. CelebA-40, private heads trained.* The heads' own gradients kept uniform certifiable (898 of 900), so the guarantee was cheap for every weighting: the certified arms accepted 3,598 of 3,600 updates at a median step of 0.00125. Primary: +0.013 [+0.002, +0.023], uniform lower on 3 of 3. Certified Stream matched exact MGDA within 0.0002 BCE at 0.58 GiB against 8.04 GiB. The fixed step 0.02, tuned for momentum, overshot exact MGDA: every attribute descended to first order at all 900 updates, yet the finite step raised one at each. That is exactly what the verified step prevents.

*C. Pythia-1B, eight tasks sharing every parameter* (200 updates, the seeds and batches of our six-rule 1B study, 6wXk Q2). On a separate seed each arm chose its rate or $\eta_0$ from $10^{-3}$, $3\cdot10^{-3}$ and $10^{-2}$, and each certified arm the constant $c$ of Equation (27) from 1/2 and $10^{-4}$. Two amendments, logged before any reporting run, added $c=10^{-4}$ and the fixed rates $3\cdot10^{-3}$ and $10^{-2}$. All certified arms chose $c=10^{-4}$ and $\eta_0=3\cdot10^{-3}$, the fixed step $10^{-2}$.

| Arm | Refused | Updates raising a loss | Worst-task test CE | Mean test CE |
|---|---|---|---|---|
| Uniform, fixed step | n/a | 577/600 | 0.685 | 0.503 |
| Uniform, certified | 448/600 | 0 | 1.069 | 0.543 |
| Exact MGDA, certified | 0/600 | 0 | 0.927 | 0.455 |
| Audit-first, certified | 0/600 | 0 | 1.050 | 0.477 |
| FAMO, certified | 394/600 | 0 | 1.523 | 0.666 |

Certified exact MGDA was never refused or held, was lower than certified uniform on 3 of 3 seeds (prespecified secondary, −0.142 [−0.426, +0.143]), and had the lowest mean CE in the table (exploratory). Audit-first used uniform at 128 of 600 updates. A certified exact-MGDA update took 3.1 s and 15.7 GiB, against 1.20 s and 12.7 GiB for a plain uniform update. All 45 autodiff audits passed.

*Revision.* The abstract and introduction will lead with training under the per-update guarantee, and the three studies join §6.3. The paper gains the 20-seed CelebA table (replacing Table 18), QM9, the matched benchmark (replacing the 14.9× figure), target-aware Stream, the residual-rank theorem and the theory-to-experiment map (reply to ZNzr). Every headline will name its protocol. We also state that the MGDA-UB audit measures the shared-trunk direction, where the verified step refused MGDA-UB at 894 of 900 updates, describe block 6's Direct arm as acquiring columns by identity actions, and scope Algorithm 1's held weighting to agreeing tasks.

=== POST: Response to Reviewer P8RF (1/3)

**We thank the reviewer** for crediting the characterization and the MGDA-UB finding, and for questions that shaped every new study. All requested experiments are done.

**W1. Is exact recovery worth running?** Yes, wherever a loss must not rise, which is the multi-task setting the reviewer asks for. A held-out average is the wrong yardstick for a protected loss: at Pythia-1B the fixed uniform step raised each task's minibatch loss at 235 to 285 of 600 updates. Under the verified step no task's loss rose, and exact MGDA is the weighting it never refused, while uniform was refused at 892 of 900 updates on CelebA with frozen heads and at 448 of 600 at Pythia-1B. So when protection is required, recovery is what lets training proceed (of the weightings we certified, only MGDA was never refused), and it is affordable: a certified exact-MGDA update at Pythia-1B took 3.1 s, where native per-task differentiation ran out of memory. Certified exact MGDA also reached lower held-out worst-task loss than certified uniform on 3 of 3 seeds (0.927 against 1.069, with prespecified primary −0.019 [−0.931, +0.892] for audit-first minus certified uniform). On unprotected averages, uniform is a known strong baseline (Kurin et al., 2022), and beating it there is not our claim.

**W2. Headline claims.** Each headline will name its protocol, and the controls the reviewer asked for now back them. Direct is the fastest exact route in a matched benchmark against structured-NTK and GhostSuite controls (Q8). With target-aware queries, the sharp endpoint alone decides most live stops (611 of 894, Q7). The 14.9× figure is replaced by the matched numbers, and block 5's two-arm comparison by the equally tuned 1B study (Q5). Blocks 1 and 4 did not log their gate, so their Stream rows will be marked uncertified, as the reviewer suggests.

**Q1. CelebA-40 with twenty seeds, and QM9.** Done, with ParetoTrace-Direct itself (1200 steps, ResNet-18, batch 128, momentum 0.9, each arm's learning rate chosen on a development seed, full test set):

| Test, mean of 20 seeds | Direct | MGDA-UB | Uniform |
|---|---|---|---|
| Worst-attribute accuracy (%) | 62.6 | 61.7 | 67.7 |
| Mean accuracy (%) | 86.9 | 85.6 | 87.6 |
| Mean balanced accuracy (%) | 71.6 | 67.3 | 70.0 |
| Mean cross-entropy | 0.304 | 0.360 | 0.280 |

Exact recovery changes the model measurably. Against MGDA-UB, Direct improves mean accuracy (+1.3 points, 19/20 seeds), balanced accuracy (+4.3, 19/20) and cross-entropy (20/20), and it has the best balanced accuracy of the three. The prespecified worst-attribute endpoint, Direct − MGDA-UB, is +0.9 points [−4.1, +5.8], 9/20. This table replaces the two-seed Table 18.

QM9 (pinned LibMTL, seven methods, two full-data epochs, three seeds), mean normalized MAE: uniform 0.303, FAMO 0.312, Aligned-MTL 0.327, FairGrad 0.337, Direct 0.339, Nash-MTL 0.414, MGDA-UB 0.477. Direct beats MGDA-UB on 3/3 seeds (−0.138 [−0.442, +0.167]) and Nash-MTL on 2/3, and is level with FairGrad and Aligned-MTL. These are unprotected averages, and the guarantee studies (general response) show what exact recovery adds when a loss is protected.

=== POST: Response to Reviewer P8RF (2/3)

**Q2. Uniform on the block-4 panels, and Stream where uniform fails.** From the retained block-4 records (same generator, M and seed, mean cosine matching to 5e-5). Objective factor $q(\mathbf 1/M)/q^\star$, gain $\min_i(K\mathbf 1/M)_i/\sqrt{q(\mathbf 1/M)\,q^\star}$:

| $M$ | seed 0 factor / gain | seed 1 | seed 2 |
|---|---|---|---|
| 512 | 2.90 / 0.29 | 1.15 / 0.97 | 1.16 / 0.96 |
| 1024 | 3.45 / 0.11 | 1.17 / 0.97 | 1.18 / 0.96 |
| 2048 | 4.26 / 0.13 | 1.68 / 0.61 | 1.19 / 0.97 |

Uniform descends on all nine panels, so there Stream buys fidelity to MGDA, a factor 1.15 to 4.26 in the objective. Under a verified step that fidelity decides progress: in Table 25 the recovered direction lowered the worst task to 65.91 where audited uniform reached 246.34. Where uniform fails to descend, beyond the M = 512 CIFAR-10 case, Stream's advantage is memory. On CelebA-40 with frozen heads the verified step refused uniform at 892 of 900 updates, and Stream certifies the direction at 0.54 GiB against Direct's 8.04 GiB (1.08 s against 0.53 s, energy within 2.1% of optimal). In training under the guarantee it matched exact MGDA within 0.0002 BCE.

**Q3. What a 5% certificate buys.** It certifies the update actually taken: $\Vert G\lambda-G\lambda^\star\Vert^2\le0.05\,q^\star$ on the current minibatch (relative direction error at most $\sqrt{0.05}\approx22\%$), with every task descending to first order, the step stochastic MGDA prescribes. It is not a population estimate, and the minibatch optimum varies more than C.9.1 suggests: $q^\star$ ranges from 0.00055 to 0.38 across five CelebA minibatches. The 1.257 and 5.07 factors of C.9.1 and C.6.6 score weights optimal for one minibatch on another, and over the twelve ordered pairs of four held-out minibatches their median is 39. ε does not set Stream's cost there: actions at four held-out states are 33, 36, 23 and 31 for every ε from 0.01 to 0.25. The descent target does, and strict descent with target-aware queries cuts live recovery from 1.85 s to 1.08 s.

**Q4. The safeguard's cost, and an intermediate rule.** The abstract will report the 60/60 result beside the tuned-step comparison (worst-task training loss 65.91 against 5.45, held-out 77.48 against 6.40). An intermediate rule exists: any $c\in(0,1)$ in Equation (27) keeps strict decrease, and we tested $c=10^{-4}$ against $c=1/2$ at Pythia-1B. The smaller constant took a 6× larger median step (0.0015 against 0.00025) and lowered validation worst-task loss after 100 updates on all 12 development pairs of rule and starting step, and for audit-first on all three reporting seeds (0.805 against 1.066). After 200 updates the two did not separate (+0.006 [−0.531, +0.542]). The cost depends on the setting as much as on the rule: on CelebA with frozen heads the same test kept exact MGDA's full step at 878 of 900 updates.

**Q5. The periodic comparison.** Block 5 ran MGDA-FA plus one exact Gram action and a damped exact Frank–Wolfe step every fifth update, with no certificate (cadence 5 and damping 0.5 chosen on one development seed, 20 scheduled refreshes, applied count not logged). Its 0.206 s is that rule's amortized cost at M = 32, while 6.24 s is a certified Stream recovery at M = 16 in a separate run, so they measure different things. Under equal tuning at Pythia-1B, with FAMO, CAGrad and uniform added, periodic − MGDA-FA is −0.037 [−0.464, +0.390], and unprotected uniform has the lowest held-out worst-task loss (0.774), as on Pythia-160M. This equally tuned comparison replaces the "strongest cheap baseline" label.

=== POST: Response to Reviewer P8RF (3/3)

**Q6. Endpoint and gate per block.**

| Block | Stopping rule | Margin gate | Evidence |
|---|---|---|---|
| 1 | Completion bound in source (geometry replay) | none | source only |
| 2 | Positive-part scalar bound | $m\ge0.95U$ | run output |
| 3 | Full contraction (Direct) | n/a | run output |
| 4 | Completion bound in source, scalar test in the paper's label | $m\ge0.95U$ | source only |
| 5 | Periodic heuristic | none | source only |
| 6 | Completion with numerical slack | full-rank fallback | run output |
| 7 | Completion vs Frank–Wolfe at fixed queries | n/a | replay by design |

Blocks 1 and 4 kept only source code, so their Stream rows will be marked uncertified, as the reviewer suggests, and block 5 ran no certificate (Q5). The positive-part bound $\max(m,0)^2/U$ is valid for either sign of $m$. A.5.7 refutes only the untruncated square written in C.1, so the gate is a descent target, not a validity condition, and B.5 and the sentence after Equation (21) will say so.

**Q7. Does the live certificate stop early away from stationarity?** Yes, wherever the guard does not interfere. App. G's 19 live stops do fall at non-positive measured energy, as the reviewer notes. Its guard subtracted a slack proportional to the Gram scale, a median 7.1× the upper endpoint near stationarity, which withheld every stop at a resolved state. Elsewhere:

- *App. G.2's fixed-step arm* stops before full rank on 22 of 60 updates, after 4 to 10 of 16 actions, each with $U/L\le1.049$ and confirmed by the full-Gram audit ($q/q^\star\le1.031$, no false stop).
- *CIFAR-10 groups (App. G.3, M = 100)*: completion needs a median 54 actions and 2.42 s against 79 and 3.81 s for the scalar endpoint, better on 14 of 15 states.
- *Live CelebA training (new, three seeds, 300 updates)*: with a guard that drops the slack, Stream stopped before full rank on 889 of 900 updates (mean 29.7 of 40 actions), its fallbacks only near stationarity, and all 36 native audits passed. With target-aware queries it stopped early on 894 of 900 at 1.14 s per update against 1.79 s, the energy test alone deciding 611 stops. Final worst-attribute accuracy did not separate from the exact arms (sharp Stream − Direct +0.8 points [−3.5, +5.2]).

**Q8. Direct against ghost-Gram and structured-NTK computation.** In a matched benchmark (same precision, tied-weight cross terms kept, configurations selected on development batches, directions within $2\times10^{-5}$ of a native reference on every trial), Direct is 1.19× and 1.22× faster than the best structured-NTK control on Pythia-160M at 8 and 32 objectives, faster than the GhostSuite adapter, 2.5–9.4× faster than native serial differentiation, and 1.03× faster at Pythia-1B, where native differentiation ran out of memory. The table is in our reply to ZNzr.

=== POST: Response to Reviewer 6wXk (1/2)

**We thank the reviewer** for crediting the theory, and for questions that led to a new theorem (Q4) and to our largest study (Q2).

**W1/Q1. Why does more accurate recovery not give better generalization on Pythia-160M?** Because MGDA optimizes a different objective: the minimum-norm point of the convex hull of the current minibatch's task gradients, the direction with the best guaranteed common descent on that minibatch. Nothing in it targets held-out loss, and on unprotected averages uniform scalarization is a known strong baseline (Kurin et al., 2022, Xin et al., 2022). What fidelity to MGDA buys is protection, because MGDA is the direction the verified step never refuses (in exact arithmetic). Unprotected, the fixed step raised some task's minibatch loss at 777 to 899 of 900 CelebA updates and at 577 of 600 at Pythia-1B. Verified, no task's loss rose, and only MGDA kept training where uniform was refused (892 of 900 on CelebA with frozen heads, 448 of 600 at Pythia-1B). Fidelity also changes the trained model: with 20 CelebA seeds, exact MGDA beats MGDA-UB on mean accuracy (19/20 seeds), balanced accuracy (19/20) and cross-entropy (20/20) (reply to P8RF).

**Q2. Efficiency and training on larger language models.** *Efficiency at Pythia-1B* (eight tasks, batch 32, one 40 GB A100):

| Pythia-1B | Seconds | Peak GiB |
|---|---|---|
| Exact recovery, Direct | 1.92 | 15.7 |
| Exact recovery, best structured-NTK order | 1.99 | 15.7 |
| Exact recovery, GhostSuite column adapter | 4.51 | 15.7 |
| Exact recovery, native per-task differentiation | out of memory | |
| Training update, Direct / uniform | 1.92 / 1.20 | 15.7 / 12.7 |
| Training update, MGDA-FA / FAMO | 1.60 / 1.60 | 12.7 / 12.7 |

Recovery times are medians over three held-out batches, update times means of ten updates after warm-up. An exact update costs 1.6× uniform and 1.2× FAMO or MGDA-FA, because the shared forward and backward dominate.

*Training at Pythia-1B* (three paired seeds, frozen protocols, held-out worst-task cross-entropy):

- *Unprotected*, six rules with rates chosen from {1e-4, 3e-4, 1e-3} on a separate seed: uniform 0.774, periodic 0.781, MGDA-FA 0.818, CAGrad 0.846, FAMO 0.878, Direct 1.026 (prespecified primary Direct − uniform +0.252 [+0.167, +0.337]). Direct's validation loss was still falling at 200 updates on every seed.
- *Under the verified step* (general response): uniform was refused at 448 of 600 updates and exact MGDA at none. Certified exact MGDA reached 0.927, below certified uniform and below unprotected Direct on 3 of 3 seeds each, with every task's loss protected at every update (prespecified primary, audit-first minus certified uniform, −0.019 [−0.931, +0.892]). Uniform's fixed step, tuned up to $10^{-2}$, reached 0.685 by raising some task's loss at 577 of 600 updates.

=== POST: Response to Reviewer 6wXk (2/2)

**W2/Q3. Live certification versus replay.** The gap came from how the old guard scaled. It subtracted a slack proportional to the Gram scale, and as App. G's run approached stationarity (median energy about $10^{-3}$ after step 60) the slack reached a median 7.1× the upper endpoint, withholding every early stop at a resolved state. The new guard drops it: a stop needs a positive lower bound, $U\le1.05L$, and a completion that reproduces the observed actions to relative residual $2\times10^{-5}$, otherwise the update falls back to exact recovery. In live CelebA-40 training (three seeds, 300 updates) Stream then certified 889 of 900 updates before full rank, its 11 fallbacks all near stationarity (exact energy below 0.1% of its median), and all 36 independent audits passed. With strict descent and target-aware queries (Q4) it certified 894 of 900 at 1.14 s per update against 1.79 s. Proposition 3 gives the route to an outward guard once action error is measured.

**Q4. How should query vectors be chosen?** We thank the reviewer for this question. Answering it gave a new theorem for the completion query rule that Stream uses.

*Theorem (residual-rank action bound).* Let $K=C+R$, where $C\succeq0$ is known (the geometry already contracted) and $R\succeq0$ is reached only through exact actions $v\mapsto Rv$, with $r=\operatorname{rank}R$. Initialize by querying $u\in\arg\min_\Delta u^\top Cu$. For the acquired span $\mathcal S$ with basis $Q$ and $Y=RQ$, let

$$A=C+Y(Q^\top Y)^\dagger Y^\top,\qquad L=\min_{\Delta}\mu^\top A\mu,\qquad U=\min_{\Delta\cap\mathcal S}\lambda^\top K\lambda.$$

While $L<U$, query the normalized off-span component of a lower minimizer $\mu$. In exact arithmetic, $L=U$ after at most $\min(M,r+1)$ actions, and the returned weights are optimal for every PSD completion of the transcript. The rule never needs $r$.

*Proof.* With $F=R^{1/2}$ and $\mathcal T=F\mathcal S$, the shorted identity gives $A=C+FP_{\mathcal T}F$, so $A\preceq K$ and $A$ agrees with $K$ on $\mathcal S$. If $L<U$, then $\mu\notin\mathcal S$, since otherwise $L=\mu^\top K\mu\ge U$. Adding $\mu$ either enlarges $\mathcal T$, or leaves $A$ unchanged while making a weighting of true value $L$ feasible, which closes the interval. Since $\dim\mathcal T\le r$, at most $r+1$ actions are needed, and at most $M$ because each action enlarges $\mathcal S$. $\square$

The full proof, and a numerical check on 608 random instances (bound never exceeded, attained in 550), are at https://github.com/chebyshevtech/ParetoTrace/blob/main/theory/residual_rank_theorem.md. The bound is informative under partial contraction, where $\operatorname{rank}R$ is at most the number of uncontracted parameters, and equals $M$ when nothing is contracted, as in our live studies.

*Target-aware queries (new).* Once the energy test passes, the descent target is what remains, so Stream queries the task with the smallest exact response. In replay this cuts actions by 7%, 24% and 3% (33 CelebA states, CIFAR-10 groups at M = 100, the 103 recorded Grams), and with strict descent as the target live recovery falls from 1.85 s to 1.08 s. *Practical rule:* time the candidate rules on a few development states and take the fastest that fits memory.

=== POST: Response to Reviewer ZNzr (1/2)

**We thank the reviewer** for crediting the theory and our reporting of limitations, and for asking for the comparisons that now anchor our efficiency claims.

**W1/Q1. What Direct adds, and how it compares.** Direct shares the contraction arithmetic of [1] and [2]: for an affine layer, $\langle g_i,g_j\rangle$ is a sum of products of activations and output adjoints. Our contribution is the conditional minimax characterization and the certified partial recovery built on it, and Direct is its full-information endpoint (Theorem 4 with nothing left uncontracted), engineered for multi-task Grams:

1. aggregation by task inside the contraction, giving the $M\times M$ task Gram rather than an example-level kernel
2. a per-parameter choice of token-pair or parameter-row contraction by a FLOP estimate
3. embedding contraction restricted to the vocabulary rows actually observed
4. summation over every use of tied or reused weights before the product
5. an online variant that retires factors after their last use
6. for shared-input tasks such as CelebA, one batched reverse pass that collects all task adjoints.

The controls share items 2 and 4. *Matched comparison*: complete recoveries at the same precision, each arm's configuration selected on development batches, and only directions matching an independent native reference to $2\times10^{-5}$ on every trial compared.

| Pythia-160M | Direct | Best structured control | GhostSuite adapter | Native serial |
|---|---|---|---|---|
| 8 objectives | 0.376 s, 4.11 GiB | 0.446 s, 4.11 GiB (1.19×) | 0.634 s, 4.11 GiB | 0.957 s, 9.11 GiB |
| 32 objectives | 0.373 s, 4.11 GiB | 0.457 s, 4.11 GiB (1.22×) | 1.712 s, 4.11 GiB | 3.497 s, 23.6 GiB |

Pooled medians of nine timed trials, ratios from unrounded medians, and Direct beat the best structured control on every batch. Native batched differentiation, which qualified on only 6 of 9 trials and so is not compared, took 1.006 s and 4.244 s. At Pythia-1B, Direct takes 1.92 s against 1.99 s for the best structured order (1.03×) and 4.51 s for the GhostSuite adapter, and native differentiation ran out of memory. A stricter check against two FP32 native references gave erratic pass rates for every arm, native serial included, so we make no speed claim under it. The GhostSuite arm adapts per-column kernels and is not an optimized all-pairs implementation. Code, adapters, harness and logs: https://github.com/chebyshevtech/ParetoTrace.

=== POST: Response to Reviewer ZNzr (2/2)

**W2/Q2. When to choose Stream over Direct.** *Rule:* time the exact routes on a few development states and take the fastest that fits memory. Our measurements give the pattern:

- *Direct* is the fastest exact route on CelebA-40 (0.53 s) and on CIFAR-10 groups at M = 100 (0.14 s against 2.42 s for Stream).
- *Native per-task differentiation* is cheaper on QM9's small network, within 4% of Direct on CelebA-40 at a quarter of its memory (0.54 s, 2.09 GiB), and out of memory at Pythia-1B.
- *Stream* wins on time at large M with agreeing gradients (1.97× at M = 2048, mean cosine about 0.8), and on memory where Direct must hold every factor: 0.54 GiB against 8.04 GiB on live CelebA-40. Its memory edge is not universal (up to 16% above contraction on the block-4 panels, Table 8), and its fallback should match the memory budget.

*Target-aware Stream (new).* On conflicting tasks the descent target sets the cost, so Stream now certifies energy with the sharp endpoint and descent with the exact margin, and queries the task with the smallest exact response once the energy test passes. Live CelebA-40, four held-out states, independent native audits:

| Method | Seconds | Peak GiB | Actions of 40 |
|---|---|---|---|
| Direct | 0.53 | 8.04 | 0 |
| Native per-task differentiation, best memory | 0.54 | 2.09 | 40 |
| Stream, original rule (0.95 target) | 1.85 | 0.54 | 23–36 |
| Stream, target-aware (strict descent) | 1.08 | 0.54 | 15–25 |

Every Stream direction passed its audit, with every task descending and energy within 2.1% of optimal. Stream now costs about 2× Direct's time, down from 3.5×, for 93% less memory, and in live training it took 1.14 s per update against 0.50 s for Direct.

**W3. Presentation.** The final version opens its experiments with this map, gives each method's full specification in one place, splits Table 1 into recovery cost, fidelity and training outcome with one protocol per block, and defines every term on first use.

| Result | What it gives | Where it is used | Where it is tested |
|---|---|---|---|
| Theorem 1 | Interval $[L_{\rm sk},U_{\rm sk})$ and minimax weights from any transcript | Stream's weights and stop | Block 7, live CelebA studies |
| Theorem 2, Corollary 4 | Exact worst-task response and certified gain at in-span weights | Descent target | Live studies, block 2 |
| Theorem 3 | No method correct on every consistent Gram certifies better than the minimax weights | Why the sharp interval is the right stop | A necessity result |
| Corollary 1, Equation (21) | Sharp endpoint, dominating the Frank–Wolfe gap and the scalar endpoint | Stopping test | Block 7, live training, target-aware study |
| Theorem 4 | Recovery when part of the geometry is contracted, one Gram when all of it is | Direct | Block 3, matched benchmark |
| Theorem 5, Proposition 4 | Additive regret under a trace budget | Trace-budget stop | 103 recorded Grams (C.6) |
| Proposition 1, Equation (27) | Under smoothness, a positive margin admits a step that raises no task's minibatch loss, and a test that finds it | Verified step | Tables 25 and 26, CelebA and Pythia-1B studies |
| Proposition 3 | Bounds under inexact actions | Numerical guard | Not yet run live |
| Residual-rank theorem (new) | At most $\operatorname{rank}R+1$ actions, exact rank | Completion query rule | Proof and numerical check in the repository |
