# Afterword

A course like this makes two kinds of promise, and it is worth saying at the end
whether they were kept.

The first is the forward reference. Whenever a notebook said *we come back to
this* — the tolerance that decides numerical rank, the conditioning that
explains which least-squares method to trust, the theorem behind low-rank
adaptation — a later notebook owed the reader delivery. Those promises are not
kept by good intentions; they are kept by a ledger. `tools/promises.yml` records
each one, and a gate in continuous integration fails the build if a promise is
made and never met. If you found a forward reference in these pages that goes
nowhere, it is a bug, and I would like to hear about it.

The second promise is the larger one: that every claim in this course is checked
against something the calculation did not assume. Not "the code runs". Not "the
figure looks plausible". Checked — against an exact algebraic identity, a closed
form, an independent method, or a theorem with a sharp constant. There are
several hundred such checks across the forty-seven notebooks, they all run on
every commit, and a red one stops publication. That discipline is the reason I
think this course is worth your time rather than one of the many other
introductions to the same material.

It has a cost, and the cost is instructive. Designing an exercise whose answer
can be verified independently is much harder than designing one whose answer you
merely know. Several times while writing this I had a perfectly good exercise
that could not be checked, and the rule is that such an exercise gets redesigned
until it can be, or cut. What survives is material where the mathematics is
sharp enough to be caught being wrong. That is a narrower course than one that
allows itself to gesture, and I think it is a better one.

A word about what is missing. There is no multigrid, no domain decomposition, no
serious treatment of parallel or communication-avoiding algorithms, and only a
glance at structured eigenvalue problems. Interior-point methods and convex
optimization proper are absent; so is the whole of numerical analysis for
differential equations, which lives in the companion physics course. Tensor
networks stop at matrix product states and never reach PEPS or DMRG proper. Each
of those is a book. Naming them is the honest alternative to pretending the
subject ends where the notebooks do.

And a word about the last volume. When I planned this course, Volume VIII was
going to be two notebooks on regression and a nod at neural networks. It grew to
seven because the request that shaped it was right: the linear algebra inside
these systems deserves to be taught properly, at depth, with the checks
attached, by somebody willing to say plainly which parts are linear algebra and
which parts are not. [§8.7](../08-learning/where-linear-algebra-stops.ipynb) exists because the second half of that sentence
matters as much as the first.

The subject is four hundred years old, the algorithms mostly seventy, and the
applications in the last volume mostly seven. The mathematics did not change. It
just kept turning out to be what everyone needed.

*Raymond Amador*
*Zürich, 2026*
