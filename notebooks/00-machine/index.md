# Chapter 0 — The Array and the Machine

Before the first matrix, three notebooks about the thing that will hold it.

A vector in a lecture is an element of a vector space. A vector in a Python
session is a contiguous block of bytes with a shape attached, a rule for reading
strides out of that block, and a floating-point format that cannot represent
one tenth. Every result in the remaining eight chapters is produced by that
object, so it is worth knowing what it can and cannot do before asking it to
diagonalise anything.

The chapter is short and it is not optional. [§0.1](arrays-and-vectorization.ipynb) is the array model —
shapes, strides, views versus copies, broadcasting, and the reason a triple loop
is four orders of magnitude slower than one `@`. [§0.2](floating-point.ipynb) is floating-point
reality: machine epsilon, catastrophic cancellation, and the habit that replaces
`==` with a tolerance, which is the habit every validation in this course rests
on. [§0.3](vectors-norms-inner-products.ipynb) is the vector itself — dot products, angles, norms, projections — done
concretely enough that Chapter II's least squares is a short step rather than a
leap.

There is one idea here that the rest of the course keeps cashing in. A
computation has two kinds of error: the error you make because your method is
approximate, and the error the machine makes because it stores 53 bits. The
first shrinks when you work harder. The second does not, and knowing which one
you are looking at is most of numerical analysis. [§0.2](floating-point.ipynb) puts a number on both.

The material is elementary and the pitch is not. A reader who has never met
`np.einsum` will meet it in [§0.1](arrays-and-vectorization.ipynb), and a reader who has used NumPy for years will
probably still find out something uncomfortable about what `A.T @ A` costs.
That is the intended experience: familiar tools seen from the side they are
usually turned away from.
