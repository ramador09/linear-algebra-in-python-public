# Volume IV — The Singular Value Decomposition

Every matrix, without exception, factors as $A = U\Sigma V^{\top}$ with $U$ and
$V$ orthogonal and $\Sigma$ diagonal and nonnegative. No symmetry required, no
squareness required, no invertibility required, no conditions of any kind. It is
the only decomposition in this course with no hypotheses, and that is why it has
quietly become the most important one.

Geometrically it says something simple enough to draw, which [§4.1](svd-geometry.ipynb) does: a matrix
takes the unit sphere to an ellipsoid, and the singular values are the
semi-axes. Rotate, stretch, rotate. Everything else follows. The four
fundamental subspaces are read straight off $U$ and $V$. The $2$-norm is
$\sigma_1$. The condition number is $\sigma_1/\sigma_n$. The rank is the count
of nonzero singular values — with all the trouble that word "nonzero" causes,
which this volume finally settles.

[§4.2](low-rank-eckart-young.ipynb) is the theorem that does the work. Truncating the SVD after $k$ terms gives
the best possible rank-$k$ approximation, in both the spectral and the Frobenius
norm, and the error is *exactly* $\sigma_{k+1}$. Not bounded by, not
asymptotically — equal. Eckart and Young proved it in 1936, and it is the reason
you can compress an image, denoise a measurement, extract topics from a corpus,
and shrink a trained neural network, all with the same three lines of code.
[§4.3](pca-covariance-whitening.ipynb) specialises it to data: centre the matrix first and truncating the SVD is
principal component analysis, with the covariance eigenproblem hiding inside.

[§4.4](randomized-svd-sketching.ipynb) is the modern chapter, and it is the one that surprises people. If you only
want the top few singular vectors of a very large matrix, you do not need to
look at all of it. Multiply by a random Gaussian matrix, orthogonalise the
result, and you have a subspace that provably captures nearly everything — with
the error bounded, not hoped for. Randomized numerical linear algebra is barely
twenty years old, it is ten times faster than the classical algorithm at the
sizes people actually work at, and it almost never appears in an undergraduate
course. It appears in this one.
