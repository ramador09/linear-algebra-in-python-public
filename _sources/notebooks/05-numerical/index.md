# Chapter V — Numerical Linear Algebra

Chapters I through IV named the algorithms. This one asks what they cost and
whether you can trust the answers, and the two questions turn out to be the same
question asked twice.

The organising idea is **conditioning**, and it deserves stating carefully
because it is the most misunderstood concept in the subject. A condition number
is a property of the *problem*, not of the method. If $\kappa(A)=10^{10}$, then
perturbing $b$ in its last bit can move the exact solution $x$ in its sixth
digit, and no algorithm, however clever, can prevent that — the information
simply is not there. What a good algorithm promises instead is *backward
stability*: the answer it returns is the exact answer to a problem within a
rounding error of the one you asked. [§5.1](norms-conditioning-stability.ipynb) measures both, on the same matrices,
and shows the residual staying at machine precision while the error climbs with
$\kappa$. Once seen, the distinction is permanent.

[§5.2](eigenvalue-algorithms.ipynb) opens the box on `eig`. There is no formula for eigenvalues past degree
four and the characteristic polynomial is a numerical catastrophe, so what
actually happens is an iteration: power iteration, then inverse iteration, then
the Rayleigh quotient iteration that converges cubically, then reduction to
Hessenberg form and the shifted $QR$ algorithm that is one of the great
algorithms of the twentieth century.

The last three notebooks are about size. [§5.3](sparse-matrices.ipynb) is sparse storage and direct
sparse solvers, where the enemy is fill-in and the weapon is reordering. [§5.4](stationary-and-cg.ipynb)
and [§5.5](krylov-gmres-preconditioning.ipynb) are the iterative methods — Jacobi through SOR, then conjugate
gradients, then the Krylov family: Arnoldi, Lanczos, GMRES, and preconditioning,
which is the art of changing the problem into an easier one that has the same
answer. A preconditioner is a change of basis, which is Chapter I again, and the
convergence rate is governed by $\sqrt{\kappa}$, which is [§5.1](norms-conditioning-stability.ipynb) again. The chapter
is where the course's threads meet.

[§5.6](multigrid.ipynb) then does the thing that should not be possible. Every
method above pays more iterations as the grid refines; multigrid pays the
*same* number, no matter how fine the grid gets, because it stops trying to fix
smooth error on a fine grid and coarsens instead. The result is a solver whose
total cost is proportional to the number of unknowns — optimal, in the strict
sense that you cannot beat reading the problem once — and the chapter ends by
measuring exactly that: iteration counts flat across four grid sizes while
conjugate gradients climbs.
