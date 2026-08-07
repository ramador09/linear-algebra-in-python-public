# Chapter IX — Quantum Information as Linear Algebra

Strip the physics folklore away and quantum information is a linear-algebra
course that already happened: states are unit vectors in $\mathbb{C}^n$, gates
are unitary matrices, measurement is a quadratic form, entanglement is a
statement about the SVD, and a quantum channel is a linear map whose matrix
this course learned to build in [Chapter VI](../06-structure/index.md). This
chapter makes that sentence literal, one identity at a time — and it is the
course's best showcase for the exact-versus-float thread, because the
subject's headline numbers are algebraic: Tsirelson's bound is $2\sqrt2$
exactly, a Bell pair's entropy is $\ln 2$ exactly, and both are gated
symbolically beside their floating-point measurements.

[§9.1](qubits-gates-bloch.ipynb) builds the one-qubit world: the Bloch sphere
as three Pauli expectation values, gates as members of $\mathrm{SU}(2)$, and
the double cover measured — a $2\pi$ rotation really is $-I$, and an
interference experiment can read the sign. [§9.2](tensor-products-entanglement.ipynb)
lets two qubits meet: the tensor product is
[§6.4](../06-structure/kronecker-vec-separable.ipynb)'s Kronecker product,
the Schmidt decomposition *is* the SVD of a reshaped state vector, and
entanglement entropy arrives by the formula
[§7.3](../07-tensors/tensor-trains-mps.ipynb) already earned.
[§9.3](chsh-tsirelson.ipynb) plays the CHSH game: sixteen classical
strategies enumerated exactly, and the quantum optimum $2\sqrt2$ obtained as
the largest eigenvalue of one $4\times4$ Hermitian matrix — verified
symbolically. [§9.4](measurement-channels-choi.ipynb) opens the box of
noise: channels as Kraus maps, complete positivity as the psd-ness of a Choi
matrix, and the transpose map's failure of it detecting entanglement.
[§9.5](quantum-fourier-transform.ipynb) closes the chapter where
[§6.3](../06-structure/circulant-toeplitz-fft.ipynb) pointed all along: the
quantum Fourier transform is the unitary DFT matrix, its circuit is a
factorization theorem, and period finding — the linear-algebra heart of
Shor's algorithm — is one matrix–vector product away.

Nothing in this chapter needs a clock, and nothing asks the reader to believe
anything: every claim is an eigenvalue, an exact enumeration, or an identity
between two routes that share no code.
