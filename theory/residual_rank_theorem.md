# Residual-rank action bound for the completion query rule

This theorem concerns how to choose query vectors for ParetoTrace-Stream. It bounds the number of exact
actions the completion query rule needs to close the certification interval, in terms of the rank of the geometry
that has not yet been contracted. A numerical check is in `audits/verify_residual_rank.py`.

## Setting

There are $M$ objectives with Gram $K = C + R \in \mathbb{S}^M_+$.

- $C \succeq 0$ is known: the part of the geometry already contracted (Theorem 4 of the paper).
- $R \succeq 0$ is reached only through exact actions $v \mapsto Rv$. Let $r = \operatorname{rank} R$.
- $\Delta$ is the probability simplex in $\mathbb{R}^M$, and $q^\star = \min_{\lambda \in \Delta} \lambda^\top K \lambda$.

After $k$ actions the queries form an orthonormal matrix $Q \in \mathbb{R}^{M\times k}$ with span $\mathcal{S}$, and the
transcript is $Y = RQ$. Define the shorted completion and the two bounds:

$$
R_{\mathcal S} = Y\,(Q^\top Y)^{\dagger}\,Y^\top,\qquad A = C + R_{\mathcal S},\qquad
L = \min_{\mu\in\Delta}\mu^\top A\,\mu,\qquad
U = \min_{\lambda\in\Delta\cap\mathcal S}\lambda^\top K\lambda .
$$

$U$ is computable from the transcript, because for $\lambda = Qc$ we have
$\lambda^\top K\lambda = c^\top(Q^\top C Q + Q^\top Y)c$.

**Rule.** Initialize with one action at $u \in \arg\min_{\Delta} u^\top C u$. While $L < U$, take a lower minimizer
$\mu \in \arg\min_\Delta \mu^\top A \mu$ and query its normalized off-span component
$v = (I - P_{\mathcal S})\mu \,/\, \lVert (I - P_{\mathcal S})\mu \rVert$.

## Theorem

Assume exact arithmetic and exact solves. Then:

1. $L \le q^\star \le U$ at every stage. Moreover $L \le \min_\Delta \mu^\top K'\mu$ for every PSD completion $K'$ of the
   transcript, and $U$ is the same for all of them.
2. The rule stops with $L = U$ after at most $\min(M,\ r+1)$ actions, initialization included.
3. At the stop, the in-span minimizer is optimal for every PSD completion of the transcript.

The rule never uses $r$.

A PSD completion is any $K' = C + R'$ with $R' \succeq 0$ and $R'Q = Y$.

## Proof

Let $F = R^{1/2}$ and $\mathcal T = F\mathcal S = \operatorname{range}(FQ)$. Then $\dim\mathcal T \le \operatorname{rank} F = r$.

**Lemma 1 (shorted identity).** $R_{\mathcal S} = F P_{\mathcal T} F$, where $P_{\mathcal T}$ is the orthogonal projector onto
$\mathcal T$. Hence $0 \preceq R_{\mathcal S} \preceq R$. Also $\lambda^\top A\lambda = \lambda^\top K\lambda$ for every
$\lambda\in\mathcal S$.

*Proof.* Let $B = FQ$. Then $Q^\top Y = B^\top B$ and $Y = FB$, so
$R_{\mathcal S} = F\,B(B^\top B)^\dagger B^\top F = F P_{\mathcal T} F$, because $B(B^\top B)^\dagger B^\top$ is the projector
onto $\operatorname{range}B$. Since $0 \preceq P_{\mathcal T} \preceq I$, it follows that $0 \preceq R_{\mathcal S} \preceq R$.
For $\lambda\in\mathcal S$ we have $F\lambda\in\mathcal T$, so
$\lambda^\top R_{\mathcal S}\lambda = \lVert P_{\mathcal T}F\lambda\rVert^2 = \lVert F\lambda\rVert^2 = \lambda^\top R\lambda$. $\square$

**Lemma 2 (bounds over all completions).** Every PSD completion satisfies $R' \succeq R_{\mathcal S}$. Moreover
$\lambda^\top K'\lambda = \lambda^\top K\lambda$ for every $\lambda\in\mathcal S$. This gives part 1.

*Proof.* Fix $x\in\mathbb R^M$. The Gram matrix of $R'^{1/2}[\,Q\ \ x\,]$ is PSD. Using $R'Q = Y$, it equals
$\begin{pmatrix} Q^\top Y & Y^\top x\\ x^\top Y & x^\top R' x\end{pmatrix}$. By the generalized Schur complement,
$x^\top R'x \ge x^\top Y (Q^\top Y)^\dagger Y^\top x = x^\top R_{\mathcal S}x$.

For $\lambda = Qc$ we have $\lambda^\top R'\lambda = c^\top Q^\top Y c = \lambda^\top R\lambda$. So $U$ is the same for every
completion, and $U \ge q^\star$ because $\Delta\cap\mathcal S \subseteq \Delta$. By Lemma 1, $A \preceq K'$ for every
completion, hence $L \le \min_\Delta \mu^\top K'\mu$. With $K' = K$ this gives $L \le q^\star$. $\square$

**Lemma 3 (the lower minimizer leaves the span).** If $L < U$, every lower minimizer $\mu$ lies outside $\mathcal S$.

*Proof.* Suppose $\mu\in\mathcal S\cap\Delta$. Lemma 1 gives $L = \mu^\top A\mu = \mu^\top K\mu$, and since $\mu$ is feasible
for $U$, $\mu^\top K\mu \ge U$. So $L \ge U$. $\square$

**Lemma 4 (each query closes the interval or grows $\mathcal T$).** Let $L < U$, and let $v$ be the query, so that
$\mathcal S' = \mathcal S + \operatorname{span}\{v\}$ contains $\mu$, and let $\mathcal T' = F\mathcal S'$. Then either
$\dim\mathcal T' = \dim\mathcal T + 1$, or $L' = U'$.

*Proof.* $\mathcal T' = \mathcal T + \operatorname{span}\{Fv\}$. Suppose the dimension does not grow, so $\mathcal T' = \mathcal T$.
By Lemma 1, $A$ depends on the transcript only through $\mathcal T$. Hence $A' = A$ and $L' = L$.

Now $\mu\in\mathcal S'\cap\Delta$, so Lemma 1 applied to $\mathcal S'$ gives $U' \le \mu^\top K\mu = \mu^\top A'\mu = L'$.
Lemma 2 gives $L' \le U'$, so $L' = U'$. $\square$

**Counting.** After the initial action, $\mathcal S = \operatorname{span}\{u\}$.

- If $Ru = 0$, then $\mathcal T = \{0\}$ and $A = C$. So $L = u^\top Cu = u^\top Ku \ge U$, and the interval is closed after
  one action.
- Otherwise $\dim\mathcal T = 1$. By Lemma 4, each later query either closes the interval or raises $\dim\mathcal T$,
  which cannot exceed $r$. So at most $r-1$ raising queries and one closing query follow, for at most $r+1$ actions.

Independently, by Lemma 3 every query made while $L<U$ adds a new direction to $\mathcal S$. After $M$ actions
$\mathcal S = \mathbb R^M$. Then $\mathcal T = \operatorname{range}F$ and $A = K$, so $L = q^\star = U$. This gives the bound
$\min(M, r+1)$.

At the stop, the in-span minimizer attains $U = L \le \min_\Delta\mu^\top K'\mu$ for every completion $K'$, and its value
is the same under every completion. So it is optimal for each of them, which is part 3. $\blacksquare$

## Scope

The bound is stated in exact rank. Neural Grams are generically full rank, so with nothing contracted ($C = 0$) the
bound is $M$, which is vacuous for our live studies.

The bound is informative under partial contraction. There $R$ is the Gram of the uncontracted parameter blocks, so
$\operatorname{rank}R$ is at most their number of parameters. This is the regime of Theorem 4, which we have not run live.

The query rule itself is the one Stream uses live. The target-aware rule changes the query only after the energy test
passes, and the lower bound is valid for any transcript.

## Numerical check

`python3 audits/verify_residual_rank.py` runs the rule on 608 random instances:

- $M \in \{5, 8, 12, 20\}$ and $\operatorname{rank} R \in \{1, 2, 3, 4, 6\}$.
- Known parts $C$ of rank $0$, $1$, $M/2$ and $M$, with and without conflicting task gradients.

It checks Lemma 3 (every query made while $L < U$ leaves the span), Lemma 4 (every non-closing query raises
$\dim\mathcal T$ by one) and the bound. All 608 pass, and the bound is attained exactly in 550 of them, so it is tight.
