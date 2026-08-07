# Preface

Linear algebra is the one piece of mathematics that everybody who computes ends
up using, and almost nobody is taught to compute *with*. The course you were
given proved that a symmetric matrix has an orthonormal eigenbasis; it did not
mention that asking a computer for that basis costs about $9n^3$ operations, or
that the answer it returns for a matrix with a repeated eigenvalue depends on
which LAPACK your machine was built against. The course you were given proved
that $A^{-1}$ exists when $\det A \neq 0$; it did not mention that forming
$A^{-1}$ is almost always the wrong thing to do. Neither omission is a
criticism. There is only so much room in a semester, and the theorems come
first. But it leaves a gap exactly where most people will spend their working
lives.

This course lives in that gap. It runs from "what is a vector" to the singular
spectrum of a trained weight matrix, and every single claim along the way is
checked in code against something the calculation did not assume.

```{admonition} This course is being written in public
:class: note
6 of a planned 47 notebooks are live, and chapters are landing in
order. Everything published here is finished work: each notebook executes
end to end on every commit, and every one of its checks passes before it
appears. What you will not find yet is the later chapters — the table of
contents shows what exists rather than what is promised. Forward references in
the prose point at notebooks that are still coming; they are tracked, and they
resolve as each one lands.
```

## What this is

Forty-seven Jupyter notebooks when complete, in nine chapters plus a Prologue
and an Epilogue.
Each notebook opens with a review of the mathematics it needs, then works
through six to ten exercises. Every exercise has three parts: a statement that
names every object you compute with, a reference solution, and a **validation**
that compares your result to an independent truth. That last part is the whole
point, and linear algebra is unusually well suited to it. Physics has to settle
for conserved quantities and asymptotic limits; here the checks are exact
algebraic identities. $\|A - QR\|$ should be a few times machine epsilon.
$\operatorname{tr} A$ should equal $\sum \lambda_i$. The best rank-$k$
approximation error should equal $\sigma_{k+1}$ — not approximately, *equal*.
When something is off by more than the tolerance, something is genuinely wrong,
and finding out what is where the learning happens.

You do not need to have taken a linear algebra course. You do need to be
comfortable writing a Python function and a `for` loop, and to remember what a
derivative is. Everything else is built here.

## What runs through it

Four threads hold the chapters together.

**Exact against floating-point.** Most notebooks compute something twice: once
exactly, in rational arithmetic with SymPy, and once in `float64`. The exact
answer is the ground truth the floating-point answer is measured against, and
the *gap* between them is a recurring lesson rather than an embarrassment. Rank
is the flagship case. In exact arithmetic rank is a theorem. In floating point
it is a threshold, and you have to choose it. The Prologue plants that question
in its last exercise; [§0.2](../00-machine/floating-point.ipynb) gives you the tolerance; Chapter IV finally settles it.

**Cost.** Every algorithm arrives with an operation count, and then a
measurement that confirms the exponent. Why `solve` beats `inv`. Why the order
you write `A @ B @ C` in can change the runtime by a factor of five hundred.
Why one factorization reused across fifty right-hand sides is not fifty
factorizations. By the end you should be able to estimate what a computation
will cost before you start it, which is a more useful skill than any single
algorithm in the book.

**Structure buys speed.** A general $n \times n$ system costs $O(n^3)$. A banded
one costs $O(n)$. A circulant one costs $O(n \log n)$, because the Fourier
matrix diagonalises it. A Kronecker product should never be formed at all.
Chapter VI is about recognising structure and cashing it in.

**And the last thread is the one I did not expect to write.** The course ends
with the linear algebra inside the systems that are currently rewriting how
everyone works. Attention is a product of matrices — three projections, a scaled
Gram matrix, a row-wise softmax, and one more product. Rotary position
embeddings are $2\times 2$ rotation blocks, and they work because
$\langle R_m q, R_n k\rangle$ depends only on $m-n$, which is a fact you can
verify to fourteen digits in about four lines. Low-rank adaptation works because
the *update* to a trained weight matrix has a rapidly decaying singular
spectrum, which is a claim we measure rather than repeat. None of this is
mysterious. It is Chapter II with better marketing, and Chapter VIII is where the
course says so, carefully and with the checks attached.

Chapter VIII is not a machine-learning course. I am not qualified to teach one
and it would be a different book. What it is, is an honest account of the linear
algebra that machine learning is made of — and [§8.7](../08-learning/where-linear-algebra-stops.ipynb) is a deliberate boundary
marker, the notebook where the course says *here is where linear algebra stops*
and shows you exactly what a nonlinearity buys that a matrix cannot.

## How to read it

In order, ideally. The chapters depend on each other: Chapter V asks what the
algorithms of Chapters I and II actually cost, Chapter VIII is written in the
index notation of Chapter VII, and the Epilogue only makes sense once all five
factorizations are in hand. That said, Chapters VI and VIII are more independent
than the rest, and a reader who already knows the fundamentals can start at
Chapter IV without much pain.

Every notebook runs in the browser: the launch buttons at the top of each page
open it on Binder or in Colab, with nothing to install. The reference solutions
are hidden on the public site — you get the figures and the check results, not
the answers. That is deliberate, and it is what makes the exercises exercises.
If you are teaching from this and want the worked solutions, get in touch.

A few notebooks flag a cell with **"Write this one yourself — the
implementation is the lesson."** Those are the ones where writing the code *is*
the understanding: the elimination loop, the Householder reflector, the
conjugate-gradient iteration, the TT-SVD sweep, the backward pass, the attention
block. Everywhere else, silence is permission. Use whatever help you like; the
mathematics is the subject, not the typing. Each notebook also carries at most
one suggestion for working alongside an assistant, and every one of those ends
with a check you run yourself. That rule does not bend.

## Acknowledgements and companions

This is the fourth of my open notebook courses, and it shares its machinery with
the others: the same validation library, the same diagram engine with its
collision-free label placement, the same insistence that a figure must depict
exactly the object the text names. Readers of *Elementary Computational
Physics* {cite}`ecp` will recognise §0.4 and §0.5 of its Chapter 0 as the
compressed ancestors of Chapters I through V here. That compression was the right
call there, where linear algebra is a tool. Here it is the subject, so it gets
the room.

What this course owes the literature is large and specific. Strang {cite}`strang2023`
for the five-factorization spine and the four subspaces; Trefethen and Bau
{cite}`trefethen1997` for the numerical arc and for insisting the SVD comes
first; Golub and Van Loan {cite}`golub2013` for depth on everything; Higham
{cite}`higham2002` for the backward-error stance; Saad {cite}`saad2003` for the
iterative methods; Axler {cite}`axler2024` for the abstract spine; Boyd and
Vandenberghe {cite}`boyd2018` for the applied register; Martinsson and Tropp
{cite}`martinsson2020` for the randomized methods; Kolda and Bader
{cite}`kolda2009` and Oseledets {cite}`oseledets2011` for the tensors.

Corrections are welcome and will be fixed. This is a living document, and it
will be wrong somewhere.

```{bibliography}
:filter: docname in docnames
```
