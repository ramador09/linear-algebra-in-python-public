# Chapter II — Orthogonality and Least Squares

Perpendicularity is the most computationally useful idea in the subject.

The reason is worth stating plainly at the start. An orthogonal matrix does not
change the length of a vector, so it cannot amplify an error. Every stable
algorithm in numerical linear algebra is built out of orthogonal operations for
exactly that reason, and every unstable one usually got that way by using
something else. Chapter V will make the claim quantitative; this chapter builds
the machinery.

The arc is: project, then factor, then fit. [§2.1](projections-normal-equations.ipynb) projects a vector onto a
subspace and discovers the normal equations by asking for the closest point.
[§2.2](gram-schmidt-qr.ipynb) builds the $QR$ factorization three ways and stages the course's first
genuine numerical disaster — classical Gram–Schmidt losing orthogonality like
$\kappa^2$ while Householder loses none — which is the cleanest available
demonstration that mathematically equivalent algorithms are not computationally
equivalent. [§2.3](least-squares-four-ways.ipynb) then solves the same least-squares problem four ways and finds
the methods separating by four orders of magnitude at degree fourteen, for
reasons the previous notebook explains.

[§2.4](pseudoinverse-regularization.ipynb) handles what happens when the problem is not merely hard but ill-posed:
the pseudoinverse, the minimum-norm solution, and regularization, ending in a
deconvolution that is hopeless without it and routine with it. [§2.5](function-space-bases.ipynb) then takes
the whole apparatus and applies it to functions instead of vectors, which is
where Legendre polynomials, Chebyshev polynomials, and the discrete Fourier
transform turn out to be the same idea wearing three coats.

If you only read one chapter of this course, read this one. Least squares is the
single most-used computation in applied mathematics, and it is where the
distance between "the formula" and "what you should actually type" is widest.
