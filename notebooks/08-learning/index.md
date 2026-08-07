# Chapter VIII — The Linear Algebra of Learning

Here is the claim this chapter defends: the mathematics inside a modern machine
learning system is, to a first approximation, the mathematics of the preceding
seven chapters, and the parts that are not have a specific name and a specific
job.

That is not a debunking. It is an invitation. A great deal of writing about
neural networks and language models is either breathless or impenetrable, and
both failures come from the same place — treating the linear algebra as
plumbing too dull to examine. It is not dull. Attention is a scaled Gram matrix
followed by a row-wise softmax and one more product, and you can build one from
scratch in thirty lines and check it against an independent implementation to
thirteen digits. Rotary position embeddings are $2\times2$ rotation blocks, and
they encode *relative* position because $\langle R_m q, R_n k\rangle$ depends
only on $m - n$ — an identity you can verify to fourteen decimal places. Low-rank
adaptation works because the update to a trained weight matrix has a fast-decaying
singular spectrum, which [§8.6](low-rank-lora-quantization.ipynb) measures rather than asserts.

The arc runs from the familiar to the current. [§8.1](learning-as-least-squares.ipynb) recasts fitting as least
squares with a regularizer, and reads ridge regression off the SVD as a filter
on singular values — which turns out to explain generalization, effective degrees
of freedom, and the double-descent curve that embarrassed the textbooks in 2019.
[§8.2](gradient-descent-conditioning.ipynb) asks how gradient descent converges on a quadratic and gets the answer
$(\kappa-1)/(\kappa+1)$: conditioning, again, now as a *training* cost. [§8.3](linear-layer-backpropagation.ipynb)
builds a linear layer with batches and derives backpropagation as a product of
Jacobians, gradient-checked against finite differences at the optimal step size
— the U-curve of [§0.2](../00-machine/floating-point.ipynb), returning eight chapters later.

[§8.4](text-to-vectors-embeddings.ipynb) gets from text to vectors: byte-pair encoding, one-hot vectors as a
standard basis, the identity that an embedding lookup *is* a matrix product,
term–document matrices, TF-IDF, latent semantic analysis, and the sense in which
word embeddings are a matrix factorization. [§8.5](attention-matrix-products.ipynb) is attention, in full. [§8.6](low-rank-lora-quantization.ipynb) is
what low-rank structure buys once a model is trained: LoRA, pruning,
quantization, and the perturbation bound that says how much accuracy each costs.

And [§8.7](where-linear-algebra-stops.ipynb) is the boundary. Five linear layers stacked collapse into one matrix —
we verify it — so depth without nonlinearity buys nothing at all. A ReLU network
is a piecewise-linear map, and we enumerate its regions and fit the exact affine
map on each. LayerNorm is a projection followed by a scaling. The residual
stream is a vector space and a probe is a least-squares fit. The chapter ends by
saying precisely where linear algebra stops being the whole story, because a
course that only showed you what its subject *can* do would be selling something.

**This is not a machine learning course.** There is no learning theory here, no
architecture survey, no training-at-scale engineering. What there is, is an
honest account of the linear algebra those things are made of, with every claim
checked.
