# Chapter VI — Structure, Graphs, and Fast Algorithms

A general dense $n \times n$ system costs $O(n^3)$ to solve. Almost no matrix
anybody actually cares about is general. This chapter is about recognising what a
matrix *is* and cashing that recognition in.

Two of the notebooks are about matrices that come from graphs. [§6.1](graphs-laplacian.ipynb) builds the
incidence matrix of a network and discovers that $L = B^{\top}B$, the Laplacian,
knows the graph's connectivity in its null space and its cluster structure in
its second eigenvector. Kirchhoff's laws turn out to be a statement about the
four fundamental subspaces, which is a satisfying thing to find out in Chapter VI
about something you drew in Chapter I. [§6.2](markov-perron-pagerank.ipynb) makes the graph's matrix stochastic
instead and gets Markov chains, the Perron–Frobenius theorem, and PageRank —
including the two failure modes (dangling nodes, disconnected components) that
Google's damping factor exists to repair.

The other three are about algebraic structure. [§6.3](circulant-toeplitz-fft.ipynb) shows that a circulant
matrix is diagonalised by the Fourier matrix — every circulant, always, with the
eigenvalues literally the FFT of the first column — which turns convolution into
multiplication and an $O(n^2)$ solve into $O(n\log n)$. [§6.4](kronecker-vec-separable.ipynb) does the same
service for Kronecker products: a two-dimensional Laplacian is
$I \otimes L + L \otimes I$, its eigenvalues are pairwise sums of
one-dimensional ones, and the operator should never be formed at all, only
applied as two matrix multiplications. At $n=200$ that distinction is a factor
of a thousand in memory.

[§6.5](kernels-gram-matrix.ipynb) closes with positive definite kernels, which is where the chapter's two
halves meet: a Gram matrix is a matrix of inner products in a feature space you
never construct, positive definiteness is exactly the condition that makes such
a space exist, and the Nyström approximation is Chapter IV's low-rank story
arriving in a new costume. It is also the bridge to Chapter VIII.

[§6.6](kalman-recursive-least-squares.ipynb) adds the structure that is not in
the matrix but in the *arrival of the data*. Refitting a least-squares problem
from scratch after every new observation is waste; the Sherman–Morrison
identity of [§1.3](../01-matrices/inverses-rank-cr.ipynb) turns it into a
rank-one update, and the result — recursive least squares — is exactly the
batch answer on every prefix, gated as an identity rather than an
approximation. Give the unknown permission to move between observations and the
same algebra becomes the Kalman filter, which is how a tracked object's
position is estimated from noisy sightings, in orbit and on the road.
