# Volume III — Eigenvalues and Spectral Theory

An eigenvector is a direction the matrix does not turn. That is the whole idea,
and almost everything else in this volume is a consequence of asking when enough
such directions exist to describe the map completely.

For symmetric matrices the answer is as good as it could possibly be: real
eigenvalues, orthonormal eigenvectors, and a decomposition $A = Q\Lambda Q^{\top}$
that turns the matrix into a list of numbers and a rotation. [§3.2](spectral-theorem.ipynb) is that
theorem and its consequences — the Rayleigh quotient, the min–max
characterisation, interlacing, and Gershgorin's discs, which localise every
eigenvalue with no computation at all. [§3.3](positive-definite-cholesky.ipynb) adds positive definiteness, which
is what makes a quadratic form an ellipse rather than a saddle, licenses the
Cholesky factorization, and quietly underwrites every covariance matrix and every
kernel in the two volumes that follow.

[§3.4](hermitian-unitary-normal.ipynb) moves to complex vectors, where Hermitian and unitary matrices take over
the roles of symmetric and orthogonal, and where a qubit turns out to be a unit
vector in $\mathbb{C}^2$ and a quantum gate a $2\times 2$ unitary. It is a
short trip from the spectral theorem to the Bloch sphere.

Then the volume turns honest. [§3.5](schur-jordan-nonnormality.ipynb) is about the matrices for which none of this
works: defective matrices with too few eigenvectors, the Jordan form that
describes them beautifully and cannot be computed at all, the Schur
decomposition that can, and **pseudospectra** — the picture that explains why a
matrix with every eigenvalue safely inside the unit disc can still amplify a
vector by a factor of a thousand before it decays. Non-normality is rarely
taught below graduate level and it is the honest answer to a question good
students actually ask.

[§3.6](matrix-functions-exponential.ipynb) closes with functions of matrices, above all $e^{At}$, which is what the
solution of a linear differential equation actually *is*, and which has a
famously large number of dubious ways to compute it.
