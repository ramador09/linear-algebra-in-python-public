# Volume VII — Multilinear Algebra and Tensor Networks

A matrix has two indices. Nothing in the world insists on two.

This volume is about arrays with more indices than that, and it is the part of
the course furthest from a standard syllabus. It has three notebooks and they do
three quite different jobs.

[§7.1](einsum-contraction.ipynb) is about **notation**, which sounds like the least interesting of the
three and is probably the most useful. Einstein summation — `np.einsum` — lets
you write any contraction of any collection of arrays as a single string, and
once you can read those strings, a great deal of code that looked like
inscrutable transposes and reshapes becomes one legible line. It also introduces
the fact that makes tensor computation possible at all: the *order* in which you
contract a chain of tensors changes the cost by factors that reach into the
millions, and finding a good order is a real optimisation problem which NumPy
will solve for you if you ask.

[§7.2](unfoldings-tucker-hosvd.ipynb) generalises the SVD. Unfold a three-way tensor along each of its modes,
take the SVD of each unfolding, and you get the higher-order SVD and the Tucker
decomposition. There is a catch, and it is worth the notebook on its own:
truncating the HOSVD is *not* optimal. Eckart–Young does not survive the trip to
three indices. It is merely quasi-optimal, within $\sqrt{3}$, and the honest
treatment of that gap is more instructive than another theorem would have been.

[§7.3](tensor-trains-mps.ipynb) is the one I most wanted to write. A quantum state of $d$ spins is a vector
with $2^d$ entries, which at $d = 50$ is more numbers than there are atoms in
the room. A **matrix product state** writes that vector as a chain of small
three-index tensors, and for the states that actually occur in nature the chain
is short. We build the tensor train decomposition from repeated SVDs, compress a
twenty-qubit state to under one percent of its dense size, evaluate an observable
without ever forming the full vector, and read the entanglement entropy off the
singular values at a bond. The whole apparatus is Volume IV's truncation theorem
applied recursively. That is the entire trick, and it is the foundation of
modern computational quantum many-body physics.
