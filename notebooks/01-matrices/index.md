# Chapter I — Matrices, Elimination, and Subspaces

This is the chapter that most linear algebra courses *are*, done with the
computer in the room.

We start where Strang starts, with the columns. A matrix times a vector is a
combination of the columns of the matrix, and once that sinks in, the column
space, the rank, and the reason $Ax=b$ sometimes has no solution all follow from
the same picture. [§1.1](four-ways-to-multiply.ipynb) multiplies matrices four different ways — inner products,
a sum of outer products, blocks, and index notation — because each way makes a
different theorem obvious, and because one of them will turn out to be how
attention is written down in Chapter VIII.

Then elimination. [§1.2](elimination-lu.ipynb) is the exemplar notebook of the course: you write
Gaussian elimination with partial pivoting yourself, you watch it produce $L$ and
$U$, you check that $PA = LU$ to thirteen digits, and then you find the matrix
where leaving the pivoting out costs eight digits of the answer. Everything
after that — the inverse we refuse to form, the rank we cannot quite pin down,
the four subspaces, the determinant that is conceptually central and
computationally almost useless — is elimination's consequence.

Two notebooks in this chapter are more abstract than the rest, deliberately.
[§1.5](vector-spaces-coordinates.ipynb) asks what a vector space is when the vectors are polynomials or matrices
rather than columns of numbers, and [§1.6](linear-maps-change-of-basis.ipynb) separates a linear map from the matrix
that represents it, which is the distinction the whole idea of a *change of
basis* depends on. The payoff is not aesthetic. Half of Chapter III is the search
for a basis in which a matrix looks simple, and you cannot search for something
you have no word for.

One warning, which [§1.3](inverses-rank-cr.ipynb) makes concrete and Chapter IV finally resolves. In exact
arithmetic the rank of a matrix is a number. In floating point it is a decision.
We will keep running into that, and it is not a defect of the machine.
