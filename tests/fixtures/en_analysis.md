<!-- page 3 -->
# Chapter 2. Compactness

Some motivating remarks about open covers precede the formal development. We recall
that a cover is a family of sets whose union contains the given set.

## 2.1 Compact sets

**Definition 2.1** (Compact). A subset $K$ of a metric space is *compact* if every open cover of $K$
has a finite subcover.

**Lemma 2.2.** Every compact set is bounded.

*Proof.* Cover $K$ by the balls $B(p, n)$ for $n \in \mathbb{N}$. By Definition 2.1 finitely many suffice, so
$K$ is bounded. $\blacksquare$

<!-- page 4 -->
**Theorem 2.3** (Heine-Borel). A subset of $\mathbb{R}^n$ is compact if and only if it is closed and bounded.
In particular the identity

$$
\sum_{k=1}^{n} k = \frac{n(n+1)}{2}
$$

holds for the index set.

*Proof.* Suppose $K$ is compact. By Lemma 2.2, $K$ is bounded. Closedness follows since
$$ K^c = \bigcup_{x \notin K} U_x $$
is open.

Conversely, closed and bounded sets lie in a cube and are covered as in Lemma 2.2 and Theorem 2.3 itself. $\square$

**Corollary 2.4.** Every closed interval $[a,b]$ is compact.

This remark has no label and simply explains how Corollary 2.4 will be used later.

**Example 2.5.** The open interval $(0,1)$ is not compact; the cover
$$
\{ (1/n, 1) : n \ge 2 \}
$$
has no finite subcover.

## 2.2 Exercises

**Exercise 2.1.** Prove that a finite union of compact sets is compact. Hint: use Theorem 2.3.

**Exercise 2.2.** Show the converse of Lemma 2.2 fails.
