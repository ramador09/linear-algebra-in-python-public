"""
ecp.linalg
==========

Domain helpers for *Linear Algebra in Python*: the course's standard pictures of
a matrix, its named test matrices, its cost model, and its offline datasets.
This module is the Volume-wide analogue of ``ecp.mechanics`` in *Elementary
Computational Physics* — the one home for machinery that recurs across
notebooks, so no notebook hand-rolls a spy plot, a Hilbert matrix, or a
four-subspaces diagram.

Two standing rules govern what belongs here.

**Infrastructure and genuine machinery, never the formula that is the lesson.**
A notebook whose subject is the Cholesky factorization writes the factorization
out in full; it does not call a package wrapper. What lives here is the
*picture* of the result, the *test matrix* it runs on, and the *cost model* it
is measured against.

**Named by what it draws or builds, never by the notebook that needed it
first.** ``singular_value_plot`` rather than ``svd_figure_for_4_2``.

Schematics drawn here route their labels through :func:`ecp.draw.place_label`
and end with :func:`ecp.draw.finish`, so the collision gate applies exactly as
it does to any hand-built diagram.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle, Circle

from . import draw
from .draw import ACCENT, INK, PANEL, SOFT

__all__ = [
    # matrix pictures
    "matrix_heatmap", "sparsity_pattern", "elimination_tableau",
    "factorization_shapes", "big_picture",
    # memory and shape
    "memory_layout", "broadcast_diagram",
    # vector / map geometry
    "plot_vectors", "unit_circle_image", "grid_transform", "subspace_plane",
    # spectra
    "spectrum_plot", "gershgorin", "singular_value_plot", "pseudospectrum",
    # graphs and attention
    "graph_plot", "attention_heatmap",
    # named test matrices
    "hilbert", "wilkinson", "gaussian_blur", "grcar", "poisson_1d", "poisson_2d",
    "random_with_condition", "random_spd", "low_rank_plus_noise",
    # cost model
    "flops", "cost_table",
    # offline datasets
    "toy_corpus", "toy_web", "toy_graph",
]


# ---------------------------------------------------------------------------
# Matrix pictures
# ---------------------------------------------------------------------------

def _diverging_norm(A):
    """Symmetric two-slope norm centred at zero, or None for a constant array."""
    lo, hi = float(np.min(A)), float(np.max(A))
    m = max(abs(lo), abs(hi))
    if m == 0.0:
        return None
    return TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m)


def matrix_heatmap(ax, A, *, title=None, annotate=None, cmap="PuOr_r",
                   signed=True, fmt="{:.2f}", cbar=False, fontsize=9):
    """Draw a matrix as the course's standard colour picture.

    A matrix seen as an image is the fastest way to read its *structure* —
    triangularity, banding, block form, symmetry, rank-one-ness — none of which
    a printed array of numbers makes visible. Signed matrices use a diverging
    map centred at zero so the sign is legible at a glance; nonnegative
    matrices (a Gram matrix, an attention matrix) are drawn sequentially.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    A : array_like, shape (m, n)
        The matrix to draw.
    title : str, optional
        Axes title.
    annotate : bool, optional
        Print each entry inside its cell. Defaults to True when the matrix has
        at most 100 entries, False otherwise.
    cmap : str, default "PuOr_r"
        Colormap; the default is the course's ink/amber-compatible diverging map.
    signed : bool, default True
        Centre the colour scale at zero. Set False for a nonnegative matrix.
    fmt : str, default "{:.2f}"
        Format string for annotations.
    cbar : bool, default False
        Attach a colourbar.
    fontsize : int, default 9
        Annotation font size.

    Returns
    -------
    matplotlib.image.AxesImage
        The image artist, so a caller can attach a colourbar of its own.
    """
    A = np.asarray(A, dtype=float)
    if annotate is None:
        annotate = A.size <= 100
    norm = _diverging_norm(A) if signed else None
    im = ax.imshow(A, cmap=cmap, norm=norm, interpolation="nearest")
    if annotate:
        m = np.max(np.abs(A)) or 1.0
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                v = A[i, j]
                colour = "white" if abs(v) > 0.62 * m else INK
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=fontsize, color=colour)
    ax.set_xticks(range(A.shape[1]))
    ax.set_yticks(range(A.shape[0]))
    ax.set_xticks(np.arange(-0.5, A.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, A.shape[0], 1), minor=True)
    ax.grid(which="minor", color=PANEL, linewidth=0.8)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    if title:
        ax.set_title(title)
    if cbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def sparsity_pattern(ax, A, *, title=None, markersize=3, annotate_nnz=True):
    """Spy plot of a matrix's nonzero pattern, in the course style.

    Where a heatmap shows *values*, a sparsity pattern shows *where the work
    is*: the fill-in a factorization creates, the bandwidth a reordering
    removes, the block structure a Kronecker product replicates. The nonzero
    count and density are annotated because they are the numbers that decide
    whether a sparse solver is worth using.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    A : array_like or scipy.sparse.spmatrix, shape (m, n)
        The matrix.
    title : str, optional
        Axes title.
    markersize : float, default 3
        Marker size for each nonzero.
    annotate_nnz : bool, default True
        Print ``nnz`` and the density in the corner.

    Returns
    -------
    int
        The number of stored nonzeros drawn.
    """
    if hasattr(A, "tocoo"):
        C = A.tocoo()
        rows, cols = C.row, C.col
        shape, nnz = A.shape, C.nnz
    else:
        A = np.asarray(A)
        rows, cols = np.nonzero(A)
        shape, nnz = A.shape, rows.size
    ax.plot(cols, rows, "s", color=INK, markersize=markersize,
            markeredgewidth=0)
    ax.set_xlim(-0.5, shape[1] - 0.5)
    ax.set_ylim(shape[0] - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.grid(False)
    if annotate_nnz:
        density = nnz / (shape[0] * shape[1])
        ax.text(0.98, 0.02, f"nnz = {nnz}\ndensity = {density:.3%}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, color=SOFT)
    if title:
        ax.set_title(title)
    return nnz


def elimination_tableau(ax, A, b=None, *, pivot=None, eliminated=(),
                        title=None, fmt="{:.3g}"):
    """Draw an augmented tableau with the active pivot and cleared entries marked.

    Gaussian elimination is a sequence of tableaux, and the step is much easier
    to follow when the pivot, the pivot row, and the entries just driven to zero
    are visually distinguished. Draw one of these per elimination step to make
    the algorithm's bookkeeping visible.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    A : array_like, shape (m, n)
        The coefficient block at this step.
    b : array_like, shape (m,), optional
        Right-hand side; drawn as an appended column behind a vertical rule.
    pivot : tuple of int, optional
        ``(i, j)`` index of the active pivot, highlighted in amber.
    eliminated : sequence of tuple of int
        Indices just driven to zero, drawn faint.
    title : str, optional
        Axes title.
    fmt : str, default "{:.3g}"
        Entry format string.

    Returns
    -------
    None
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    cols = n + (1 if b is not None else 0)
    elim = set(map(tuple, eliminated))
    for i in range(m):
        for j in range(cols):
            if b is not None and j == n:
                v = float(np.asarray(b, dtype=float)[i])
            else:
                v = A[i, j]
            is_pivot = pivot is not None and (i, j) == tuple(pivot)
            is_elim = (i, j) in elim
            if is_pivot:
                ax.add_patch(Rectangle((j - 0.45, i - 0.4), 0.9, 0.8,
                                       facecolor=ACCENT, alpha=0.28,
                                       edgecolor=ACCENT, linewidth=1.4, zorder=1))
            ax.text(j, i, fmt.format(v), ha="center", va="center", zorder=3,
                    fontsize=11,
                    color=SOFT if is_elim else INK,
                    alpha=0.45 if is_elim else 1.0,
                    fontweight="bold" if is_pivot else "normal")
    if b is not None:
        ax.plot([n - 0.5, n - 0.5], [-0.6, m - 0.4], color=SOFT, linewidth=1.2)
    # Bracket the tableau like a matrix.
    for x, d in ((-0.62, +1), (cols - 0.38, -1)):
        ax.plot([x, x], [-0.6, m - 0.4], color=INK, linewidth=1.4)
        ax.plot([x, x + 0.14 * d], [-0.6, -0.6], color=INK, linewidth=1.4)
        ax.plot([x, x + 0.14 * d], [m - 0.4, m - 0.4], color=INK, linewidth=1.4)
    ax.set_xlim(-1.0, cols - 0.1)
    ax.set_ylim(m - 0.2, -0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)


def factorization_shapes(ax, kind, *, m=6, n=4, r=3, title=None):
    """Draw the block-shape schematic of one of the five factorizations.

    The course's signature figure. Each factorization is, before it is anything
    else, a statement about *shapes*: which factor is tall, which is square,
    which is triangular, which is diagonal. Drawing the shapes makes the
    dimension bookkeeping — and the difference between the full and economy
    forms — impossible to misremember.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    kind : {"CR", "LU", "QR", "QLQt", "SVD"}
        Which factorization to draw.
    m, n : int, default 6, 4
        Row and column counts of A, in arbitrary drawing units.
    r : int, default 3
        Rank, used by "CR" and "SVD".
    title : str, optional
        Axes title; defaults to the factorization's equation.

    Returns
    -------
    None
    """
    specs = {
        "CR":   ("$A = CR$",        [("A", n, m, None), ("C", r, m, None), ("R", n, r, "upper")]),
        "LU":   ("$PA = LU$",       [("A", n, n, None), ("L", n, n, "lower"), ("U", n, n, "upper")]),
        "QR":   ("$A = QR$",        [("A", n, m, None), ("Q", n, m, None), ("R", n, n, "upper")]),
        "QLQt": (r"$S = Q\Lambda Q^{\!\top}$",
                 [("S", n, n, "sym"), ("Q", n, n, None), (r"$\Lambda$", n, n, "diag"), (r"$Q^{\!\top}$", n, n, None)]),
        "SVD":  (r"$A = U\Sigma V^{\!\top}$",
                 [("A", n, m, None), ("U", m, m, None), (r"$\Sigma$", n, m, "diag"), (r"$V^{\!\top}$", n, n, None)]),
    }
    if kind not in specs:
        raise ValueError(f"unknown factorization {kind!r}; expected one of {sorted(specs)}")
    eq, blocks = specs[kind]
    unit, gap = 0.36, 0.9
    x = 0.0
    for k, (name, w, h, pattern) in enumerate(blocks):
        W, H = w * unit, h * unit
        y = -H / 2
        # White ground first, then the pattern on top of it, then the outline:
        # drawing the pattern beneath an opaque face hides it entirely.
        ax.add_patch(Rectangle((x, y), W, H, facecolor="white",
                               edgecolor="none", zorder=1))
        if pattern == "upper":
            ax.add_patch(plt.Polygon([(x, y + H), (x + W, y + H), (x + W, y)],
                                     facecolor=ACCENT, alpha=0.34, edgecolor="none", zorder=2))
        elif pattern == "lower":
            ax.add_patch(plt.Polygon([(x, y + H), (x, y), (x + W, y)],
                                     facecolor=ACCENT, alpha=0.34, edgecolor="none", zorder=2))
        elif pattern == "sym":
            ax.add_patch(Rectangle((x, y), W, H, facecolor=ACCENT, alpha=0.18,
                                   edgecolor="none", zorder=2))
        elif pattern == "diag":
            d = min(w, h)
            for i in range(d):
                ax.add_patch(Rectangle((x + i * unit, y + H - (i + 1) * unit),
                                       unit, unit, facecolor=ACCENT, alpha=0.60,
                                       edgecolor="none", zorder=2))
        ax.add_patch(Rectangle((x, y), W, H, facecolor="none",
                               edgecolor=INK, linewidth=1.6, zorder=3))
        ax.text(x + W / 2, 0, name, ha="center", va="center", fontsize=15,
                color=INK, zorder=4)
        x += W
        if k == 0:
            ax.text(x + gap / 2, 0, "=", ha="center", va="center",
                    fontsize=17, color=SOFT, zorder=3)
            x += gap
        elif k < len(blocks) - 1:
            x += gap * 0.35
    ax.set_xlim(-0.4, x + 0.4)
    lim = max(m, n) * unit / 2 + 0.6
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title if title is not None else eq)


def big_picture(ax, m, n, r, *, title=None):
    """Draw Strang's four-fundamental-subspaces diagram with its dimensions.

    The single most useful picture in elementary linear algebra: the row space
    and null space partitioning the input space, the column space and left null
    space partitioning the output space, each pair orthogonal, and the map
    carrying the row space isomorphically onto the column space. Dimensions are
    labelled from the given ``m``, ``n``, ``r`` so the picture always matches
    the matrix under discussion.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    m, n : int
        Row and column counts of the matrix.
    r : int
        Its rank.
    title : str, optional
        Axes title.

    Returns
    -------
    None
    """
    def diamond(cx, cy, w, h, label, dim, colour):
        pts = [(cx, cy + h), (cx + w, cy), (cx, cy - h), (cx - w, cy)]
        ax.add_patch(plt.Polygon(pts, facecolor=colour, alpha=0.16,
                                 edgecolor=colour, linewidth=1.6, zorder=2))
        ax.text(cx, cy + 0.10, label, ha="center", va="center", fontsize=12,
                color=INK, zorder=3, gid="_nocheck")
        ax.text(cx, cy - 0.22, dim, ha="center", va="center", fontsize=11,
                color=SOFT, zorder=3, gid="_nocheck")

    diamond(-1.55, 0.80, 0.78, 0.46, r"row space $C(A^{\!\top})$", f"dim $= r = {r}$", INK)
    diamond(-1.55, -0.80, 0.78, 0.46, r"null space $N(A)$", f"dim $= n - r = {n - r}$", SOFT)
    diamond(1.55, 0.80, 0.78, 0.46, r"column space $C(A)$", f"dim $= r = {r}$", ACCENT)
    diamond(1.55, -0.80, 0.78, 0.46, r"left null space $N(A^{\!\top})$", f"dim $= m - r = {m - r}$", SOFT)

    # The arrow labels sit at x = 0, clear of both diamonds (which start at
    # |x| = 0.77). Keep them SHORT: a long gloss here is wide enough in data
    # units to reach into the subspace labels, which is a collision.
    ax.annotate("", xy=(0.70, 0.80), xytext=(-0.70, 0.80),
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.8))
    ax.text(0.0, 0.90, r"$A$", ha="center", va="bottom",
            fontsize=13, color=INK, gid="_nocheck")
    ax.annotate("", xy=(0.70, -0.95), xytext=(-0.70, -0.80),
                arrowprops=dict(arrowstyle="-|>", color=SOFT, linewidth=1.4,
                                linestyle=":"))
    ax.text(0.0, -1.06, r"$A\mathbf{x} = \mathbf{0}$", ha="center", va="top",
            fontsize=11, color=SOFT, gid="_nocheck")

    ax.plot([-1.55, -1.55], [0.28, -0.28], color=SOFT, linewidth=1.0, linestyle="--")
    ax.plot([1.55, 1.55], [0.28, -0.28], color=SOFT, linewidth=1.0, linestyle="--")
    ax.text(-1.42, 0.0, r"$\perp$", fontsize=13, color=SOFT, va="center", gid="_nocheck")
    ax.text(1.68, 0.0, r"$\perp$", fontsize=13, color=SOFT, va="center", gid="_nocheck")

    ax.text(-1.55, 1.62, rf"$\mathbb{{R}}^{{n}},\; n = {n}$", ha="center",
            fontsize=12, color=INK, gid="_nocheck")
    ax.text(1.55, 1.62, rf"$\mathbb{{R}}^{{m}},\; m = {m}$", ha="center",
            fontsize=12, color=INK, gid="_nocheck")

    ax.set_xlim(-2.7, 2.7)
    ax.set_ylim(-1.75, 1.95)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)


# ---------------------------------------------------------------------------
# Vector and linear-map geometry
# ---------------------------------------------------------------------------

def plot_vectors(ax, vecs, labels=None, *, origin=None, colors=None,
                 lim=None, title=None, grid=True):
    """Draw 2-D vectors from a common origin with collision-free labels.

    The default picture of a vector in this course: an arrow from the origin,
    labelled beside its head, over a faint coordinate grid so the components are
    readable off the axes rather than guessed.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    vecs : array_like, shape (k, 2)
        The vectors, one per row.
    labels : sequence of str, optional
        Label for each vector; ``None`` skips labelling.
    origin : array_like, shape (2,), optional
        Common tail; defaults to the origin.
    colors : sequence of str, optional
        One colour per vector; defaults to alternating ink and amber.
    lim : float, optional
        Symmetric axis limit; derived from the data when omitted.
    title : str, optional
        Axes title.
    grid : bool, default True
        Draw the faint ruled coordinate grid.

    Returns
    -------
    None
    """
    V = np.atleast_2d(np.asarray(vecs, dtype=float))
    o = np.zeros(2) if origin is None else np.asarray(origin, dtype=float)
    if colors is None:
        colors = [INK if i % 2 == 0 else ACCENT for i in range(len(V))]
    if lim is None:
        lim = 1.25 * max(1.0, float(np.max(np.abs(np.vstack([V + o, o])))))
    if grid:
        step = 1.0 if lim <= 6 else round(lim / 5)
        draw.grid(ax, (-lim, lim), (-lim, lim), step=step, labels=("x", "y"))
    for k, v in enumerate(V):
        lab = None if labels is None else labels[k]
        draw.vector(ax, o, o + v, label=lab, color=colors[k])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)


def unit_circle_image(ax, A, *, n=400, show_singular=True, show_circle=True,
                      title=None):
    """Draw the unit circle and its image under a 2x2 matrix.

    Every 2x2 matrix maps the unit circle to an ellipse, and the ellipse's
    semi-axes are the singular values while its axis directions are the left
    singular vectors. This is the SVD made visible, and it is the picture the
    course returns to whenever a matrix has to be understood geometrically.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    A : array_like, shape (2, 2)
        The matrix.
    n : int, default 400
        Number of samples around the circle.
    show_singular : bool, default True
        Draw the image semi-axes ``sigma_i u_i``.
    show_circle : bool, default True
        Draw the source unit circle.
    title : str, optional
        Axes title.

    Returns
    -------
    numpy.ndarray
        The singular values, in descending order.
    """
    A = np.asarray(A, dtype=float)
    t = np.linspace(0, 2 * np.pi, n)
    circle = np.vstack([np.cos(t), np.sin(t)])
    image = A @ circle
    if show_circle:
        ax.plot(circle[0], circle[1], color=SOFT, linewidth=1.2, linestyle="--",
                label="unit circle")
    ax.plot(image[0], image[1], color=ACCENT, linewidth=2.2, label="image $A\\,S^1$")
    U, s, Vt = np.linalg.svd(A)
    if show_singular:
        for i in range(2):
            axis = s[i] * U[:, i]
            ax.annotate("", xy=tuple(axis), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.8))
            # Push the label past the arrowhead and align it AWAY from the
            # origin, so it never sits on the arrow, the ellipse, or the frame.
            tip = 1.16 * axis
            ax.text(tip[0], tip[1], rf"$\sigma_{{{i+1}}}\mathbf{{u}}_{{{i+1}}}$",
                    fontsize=11, color=INK,
                    ha="left" if axis[0] >= 0 else "right",
                    va="bottom" if axis[1] >= 0 else "top")
    ax.set_aspect("equal")
    ax.axhline(0, color=SOFT, linewidth=0.8)
    ax.axvline(0, color=SOFT, linewidth=0.8)
    # Room for the labels: the ellipse reaches sigma_1, the labels reach past it.
    pad = 1.38 * float(s[0])
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    if title:
        ax.set_title(title)
    return s


def grid_transform(ax, A, *, n=11, extent=1.0, title=None, show_source=True):
    """Draw a reference lattice and its image under a 2x2 linear map.

    A linear map is completely determined by what it does to a grid: straight
    lines stay straight, parallel lines stay parallel, and the origin stays
    fixed. Drawing the deformed grid beside the original shows all three at
    once, and makes shear, rotation, and singularity immediately legible.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    A : array_like, shape (2, 2)
        The matrix.
    n : int, default 11
        Number of grid lines in each direction.
    extent : float, default 1.0
        The lattice covers ``[-extent, extent]`` in both directions.
    title : str, optional
        Axes title.
    show_source : bool, default True
        Draw the undeformed lattice faintly beneath the image.

    Returns
    -------
    None
    """
    A = np.asarray(A, dtype=float)
    ts = np.linspace(-extent, extent, n)
    fine = np.linspace(-extent, extent, 200)
    for t in ts:
        for pts in (np.vstack([fine, np.full_like(fine, t)]),
                    np.vstack([np.full_like(fine, t), fine])):
            if show_source:
                ax.plot(pts[0], pts[1], color=SOFT, linewidth=0.6, alpha=0.35)
            q = A @ pts
            ax.plot(q[0], q[1], color=ACCENT, linewidth=0.9)
    for k, colour in ((0, INK), (1, INK)):
        e = np.zeros(2)
        e[k] = 1.0
        ax.annotate("", xy=tuple(A @ e), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=colour, linewidth=2.0))
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    if title:
        ax.set_title(title)


def subspace_plane(ax, basis, *, lim=1.6, color=ACCENT, alpha=0.25, label=None):
    """Draw a line or plane through the origin spanned by the given basis.

    Subspaces are the objects the four-subspaces theorem is about, and in three
    dimensions they are drawable: a one-dimensional subspace is a line, a
    two-dimensional one a plane, both through the origin. Use on a 3-D axes to
    show a column space beside its orthogonal left null space.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        Target 3-D axes.
    basis : array_like, shape (3, k)
        Columns spanning the subspace; ``k`` is 1 (line) or 2 (plane).
    lim : float, default 1.6
        Half-extent of the drawn patch.
    color : str, default the course accent
        Colour.
    alpha : float, default 0.25
        Face alpha for a plane.
    label : str, optional
        Legend label.

    Returns
    -------
    None
    """
    B = np.asarray(basis, dtype=float)
    if B.ndim == 1:
        B = B[:, None]
    k = B.shape[1]
    Q, _ = np.linalg.qr(B)
    if k == 1:
        t = np.array([-lim, lim])
        pts = np.outer(t, Q[:, 0])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=2.6,
                label=label)
    elif k == 2:
        s = np.linspace(-lim, lim, 2)
        S, T = np.meshgrid(s, s)
        P = (S[..., None] * Q[:, 0] + T[..., None] * Q[:, 1])
        ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], color=color,
                        alpha=alpha, shade=False)
        if label:
            ax.plot([], [], color=color, linewidth=6, alpha=alpha, label=label)
    else:
        raise ValueError("subspace_plane draws a line (k=1) or a plane (k=2)")


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------

def spectrum_plot(ax, eigvals, *, unit_circle=False, title=None, label=None,
                  color=None, marker="o"):
    """Plot a spectrum on the complex plane.

    Where the eigenvalues sit answers most of the questions one asks about a
    matrix: on the real axis for Hermitian, on the unit circle for unitary, in
    the left half-plane for a stable flow, inside the unit disc for a
    convergent iteration. One picture, several theorems.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    eigvals : array_like
        The eigenvalues (real or complex).
    unit_circle : bool, default False
        Draw the unit circle for reference.
    title : str, optional
        Axes title.
    label : str, optional
        Legend label.
    color : str, optional
        Marker colour; defaults to the course accent.
    marker : str, default "o"
        Marker style.

    Returns
    -------
    None
    """
    lam = np.atleast_1d(np.asarray(eigvals))
    ax.scatter(lam.real, lam.imag, s=52, marker=marker,
               color=ACCENT if color is None else color,
               edgecolor=INK, linewidth=0.9, zorder=4, label=label)
    if unit_circle:
        t = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(t), np.sin(t), color=SOFT, linewidth=1.1, linestyle="--",
                zorder=1)
    ax.axhline(0, color=SOFT, linewidth=0.8, zorder=0)
    ax.axvline(0, color=SOFT, linewidth=0.8, zorder=0)
    ax.set_xlabel(r"$\mathrm{Re}\,\lambda$")
    ax.set_ylabel(r"$\mathrm{Im}\,\lambda$")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)


def gershgorin(ax, A, *, title=None, show_eigs=True):
    """Draw the Gershgorin discs of a matrix with its true eigenvalues.

    Gershgorin's theorem localises every eigenvalue inside a disc centred on a
    diagonal entry with radius the absolute row sum of the off-diagonal
    entries — a bound requiring no computation beyond reading the matrix off
    the page. Drawing the discs with the true spectrum inside shows both that
    the theorem holds and how loose it is.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    A : array_like, shape (n, n)
        The matrix.
    title : str, optional
        Axes title.
    show_eigs : bool, default True
        Overlay the true eigenvalues.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        Disc centres (the diagonal) and radii (the off-diagonal absolute row sums).
    """
    A = np.asarray(A)
    centres = np.diag(A).astype(complex)
    radii = np.sum(np.abs(A), axis=1) - np.abs(np.diag(A))
    for c, r in zip(centres, radii):
        ax.add_patch(Circle((c.real, c.imag), r, facecolor=ACCENT, alpha=0.16,
                            edgecolor=ACCENT, linewidth=1.3, zorder=1))
        ax.plot(c.real, c.imag, "x", color=SOFT, markersize=7, zorder=3)
    if show_eigs:
        lam = np.linalg.eigvals(A)
        ax.scatter(lam.real, lam.imag, s=54, color=INK, zorder=4,
                   label="eigenvalues")
    span = float(np.max(np.abs(centres.real) + radii)) + 1.0
    ax.set_xlim(np.min(centres.real - radii) - 0.6, np.max(centres.real + radii) + 0.6)
    ax.set_ylim(-max(radii.max(), 1.0) - 0.6, max(radii.max(), 1.0) + 0.6)
    ax.axhline(0, color=SOFT, linewidth=0.8, zorder=0)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\mathrm{Re}\,\lambda$")
    ax.set_ylabel(r"$\mathrm{Im}\,\lambda$")
    if title:
        ax.set_title(title)
    return np.diag(A).copy(), radii


def singular_value_plot(ax, s, *, k=None, title=None, label=None, color=None,
                        annotate_cliff=False, floor=1e-18):
    """Plot a singular-value spectrum on a log axis, with an optional truncation.

    The shape of this plot is the whole content of low-rank approximation: a
    cliff means the matrix is nearly low rank, a slow decay means it is not, and
    the height of ``sigma_{k+1}`` is *exactly* the error of the best rank-k
    approximation. Marking the truncation makes the Eckart-Young statement
    readable straight off the figure.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    s : array_like
        Singular values, descending.
    k : int, optional
        Truncation rank; a vertical rule is drawn after the k-th value.
    title : str, optional
        Axes title.
    label : str, optional
        Legend label.
    color : str, optional
        Line colour; defaults to ink.
    annotate_cliff : bool, default False
        Mark the largest consecutive ratio drop.
    floor : float, default 1e-18
        Values below this are clipped so the log axis stays finite.

    Returns
    -------
    None
    """
    s = np.maximum(np.asarray(s, dtype=float), floor)
    idx = np.arange(1, s.size + 1)
    ax.semilogy(idx, s, "o-", color=INK if color is None else color,
                markersize=4, label=label)
    if k is not None:
        ax.axvline(k + 0.5, color=ACCENT, linewidth=1.6, linestyle="--")
        ax.text(k + 0.7, s[0], f"keep $k = {k}$", color=ACCENT, fontsize=11,
                va="top")
    if annotate_cliff and s.size > 2:
        drops = s[:-1] / s[1:]
        j = int(np.argmax(drops))
        ax.annotate(f"cliff: $\\sigma_{{{j+1}}}/\\sigma_{{{j+2}}} = {drops[j]:.1f}$",
                    xy=(j + 1.5, np.sqrt(s[j] * s[j + 1])),
                    xytext=(j + 3.0, s[0] * 0.3), color=SOFT, fontsize=10,
                    arrowprops=dict(arrowstyle="->", color=SOFT, linewidth=1.0))
    ax.set_xlabel("index $i$")
    ax.set_ylabel(r"$\sigma_i$")
    if title:
        ax.set_title(title)


def pseudospectrum(ax, A, *, lim=None, n=180, levels=(-1, -1.5, -2, -2.5, -3),
                   title=None, show_eigs=True):
    """Contour the resolvent norm, i.e. draw the pseudospectra of a matrix.

    For a non-normal matrix the eigenvalues are a poor guide to behaviour: a
    matrix can have every eigenvalue well inside the unit disc and still amplify
    a vector by orders of magnitude before decaying. The epsilon-pseudospectrum
    — the set where ``||(zI - A)^{-1}|| > 1/epsilon``, equivalently where the
    smallest singular value of ``zI - A`` is below epsilon — is the set that
    does predict it. For a normal matrix the contours are just discs of radius
    epsilon around each eigenvalue; anything wider is non-normality made visible.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    A : array_like, shape (n, n)
        The matrix.
    lim : tuple, optional
        ``(xmin, xmax, ymin, ymax)``; derived from the spectrum when omitted.
    n : int, default 180
        Grid resolution per axis.
    levels : sequence of float, default (-1, -1.5, -2, -2.5, -3)
        Contour levels as ``log10(epsilon)``.
    title : str, optional
        Axes title.
    show_eigs : bool, default True
        Overlay the eigenvalues.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        The real grid, imaginary grid, and ``log10`` of the smallest singular
        value of ``zI - A`` at each grid point.
    """
    A = np.asarray(A, dtype=complex)
    lam = np.linalg.eigvals(A)
    if lim is None:
        pad = 0.6 * max(1.0, float(np.ptp(lam.real)), float(np.ptp(lam.imag)))
        lim = (lam.real.min() - pad, lam.real.max() + pad,
               lam.imag.min() - pad, lam.imag.max() + pad)
    xs = np.linspace(lim[0], lim[1], n)
    ys = np.linspace(lim[2], lim[3], n)
    X, Y = np.meshgrid(xs, ys)
    I = np.eye(A.shape[0], dtype=complex)
    smin = np.empty_like(X)
    for i in range(n):
        for j in range(n):
            smin[i, j] = np.linalg.svd(complex(X[i, j], Y[i, j]) * I - A,
                                       compute_uv=False)[-1]
    Z = np.log10(np.maximum(smin, 1e-300))
    cs = ax.contour(X, Y, Z, levels=sorted(levels), colors=ACCENT, linewidths=1.3)
    ax.clabel(cs, fmt=lambda v: rf"$10^{{{v:g}}}$", fontsize=9)
    if show_eigs:
        ax.scatter(lam.real, lam.imag, s=48, color=INK, zorder=5,
                   label="eigenvalues")
    ax.set_xlabel(r"$\mathrm{Re}\,z$")
    ax.set_ylabel(r"$\mathrm{Im}\,z$")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return X, Y, Z


# ---------------------------------------------------------------------------
# Graphs and attention
# ---------------------------------------------------------------------------

def graph_plot(ax, edges, pos, *, values=None, labels=None, node_size=420,
               cmap="PuOr_r", title=None, directed=False, edge_width=1.6):
    """Draw a small graph with optional node colouring.

    Volume VI treats a graph as a matrix, and the two representations have to be
    read together: the picture says which nodes are adjacent, the Laplacian says
    what that adjacency implies. Colouring nodes by a computed quantity — a
    Fiedler vector's sign, a PageRank score — puts the spectral answer back on
    the picture it came from.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    edges : sequence of tuple of int
        Edge list as ``(i, j)`` index pairs.
    pos : array_like, shape (n, 2)
        Node coordinates.
    values : array_like, shape (n,), optional
        Per-node scalar; drives node colour (and size, if positive).
    labels : sequence of str, optional
        Node labels; defaults to the node indices.
    node_size : float, default 420
        Base marker area.
    cmap : str, default "PuOr_r"
        Colormap for ``values``.
    title : str, optional
        Axes title.
    directed : bool, default False
        Draw arrowheads.
    edge_width : float, default 1.6
        Edge line width.

    Returns
    -------
    None
    """
    P = np.asarray(pos, dtype=float)
    for i, j in edges:
        if directed:
            ax.annotate("", xy=P[j], xytext=P[i],
                        arrowprops=dict(arrowstyle="-|>", color=SOFT,
                                        linewidth=edge_width,
                                        shrinkA=14, shrinkB=14))
        else:
            ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color=SOFT,
                    linewidth=edge_width, zorder=1)
    if values is None:
        ax.scatter(P[:, 0], P[:, 1], s=node_size, color=PANEL,
                   edgecolor=INK, linewidth=1.6, zorder=3)
    else:
        v = np.asarray(values, dtype=float)
        m = np.max(np.abs(v)) or 1.0
        sizes = node_size * (0.5 + 1.5 * (v - v.min()) / (np.ptp(v) or 1.0))
        ax.scatter(P[:, 0], P[:, 1], s=sizes, c=v, cmap=cmap, vmin=-m, vmax=m,
                   edgecolor=INK, linewidth=1.6, zorder=3)
    names = [str(i) for i in range(len(P))] if labels is None else labels
    for p, name in zip(P, names):
        ax.text(p[0], p[1], name, ha="center", va="center", fontsize=10,
                color=INK, zorder=4, gid="_nocheck")
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)


def attention_heatmap(ax, A, tokens, *, title=None, cmap="PuOr_r", cbar=True,
                      annotate=None):
    """Draw an attention matrix with its token labels on both axes.

    An attention matrix is row-stochastic: row ``i`` says how the query at
    position ``i`` distributes its weight over the keys. Labelling both axes
    with the actual tokens turns an anonymous matrix into a readable statement
    about which position attends to which, and makes a causal mask's strict
    upper triangle of zeros obvious.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    A : array_like, shape (n, n)
        Attention weights; rows should sum to one.
    tokens : sequence of str
        Token strings, one per position.
    title : str, optional
        Axes title.
    cmap : str, default "PuOr_r"
        Colormap.
    cbar : bool, default True
        Attach a colourbar.
    annotate : bool, optional
        Print the weights; defaults to True for at most 64 entries.

    Returns
    -------
    matplotlib.image.AxesImage
        The image artist.
    """
    A = np.asarray(A, dtype=float)
    if annotate is None:
        annotate = A.size <= 64
    im = ax.imshow(A, cmap=cmap, vmin=0.0, vmax=max(1e-12, float(A.max())),
                   interpolation="nearest")
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticklabels(tokens)
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")
    ax.grid(False)
    if annotate:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if A[i, j] > 0.6 * A.max() else INK)
    if cbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title)
    return im


# ---------------------------------------------------------------------------
# Named test matrices
# ---------------------------------------------------------------------------

def hilbert(n):
    """The n-by-n Hilbert matrix, ``H[i, j] = 1/(i + j + 1)``.

    The standard ill-conditioned symmetric positive definite matrix: exactly
    representable in rational arithmetic, spectacularly hard in floating point
    (``kappa`` grows roughly like ``e^{3.5n}``). It is the course's recurring
    demonstration that conditioning is a property of the *problem*, not of the
    algorithm.

    Parameters
    ----------
    n : int
        Size.

    Returns
    -------
    numpy.ndarray, shape (n, n)
        The Hilbert matrix in float64.
    """
    i = np.arange(n)
    return 1.0 / (i[:, None] + i[None, :] + 1.0)


def wilkinson(n):
    """Wilkinson's growth-factor matrix for Gaussian elimination.

    Unit lower triangle of ``-1``, unit diagonal, and a final column of ``+1``.
    Partial pivoting never swaps a row here, yet the growth factor reaches
    ``2^{n-1}`` exactly, which is the standard counterexample to "partial
    pivoting is unconditionally safe".

    Parameters
    ----------
    n : int
        Size.

    Returns
    -------
    numpy.ndarray, shape (n, n)
        The matrix in float64.
    """
    A = np.eye(n)
    A[np.tril_indices(n, -1)] = -1.0
    A[:, -1] = 1.0
    return A


def gaussian_blur(n, width=0.03):
    """The 1-D Gaussian convolution operator on ``n`` equispaced points of [0, 1].

    The course's standard *ill-posed* problem, as opposed to the merely
    ill-conditioned ones above. Row ``i`` samples the kernel
    ``exp(-(t_i - t_j)^2 / 2w^2) / (w sqrt(2 pi))`` at the midpoint grid
    ``t_i = (i + 1/2)/n`` and multiplies by the quadrature weight ``h = 1/n``, so
    ``A @ x`` is the trapezoid-free midpoint rule for the convolution and every
    row sums to 1 to graphical accuracy. The matrix is symmetric, Toeplitz, and
    its singular values decay *exponentially* rather than algebraically, which is
    what separates an ill-posed problem from a hard one: no amount of precision
    recovers the components below the noise, because the forward map genuinely
    destroyed them.

    Parameters
    ----------
    n : int
        Number of grid points; the matrix is ``(n, n)``.
    width : float, default 0.03
        Standard deviation of the kernel, in units of the domain [0, 1]. Wider
        kernels blur more and decay faster, so ``kappa`` grows with ``width``.

    Returns
    -------
    numpy.ndarray, shape (n, n)
        The blur matrix in float64.
    """
    t = (np.arange(n) + 0.5) / n
    d = t[:, None] - t[None, :]
    return np.exp(-0.5 * (d / width) ** 2) / (width * np.sqrt(2.0 * np.pi)) / n


def grcar(n, k=3):
    """The Grcar matrix, a standard highly non-normal Toeplitz example.

    Ones on the diagonal and on ``k`` superdiagonals, ``-1`` on the first
    subdiagonal. Its eigenvalues sit on a curve, but its pseudospectra bulge far
    beyond them, so it is the canonical picture of why eigenvalues alone do not
    predict transient behaviour.

    Parameters
    ----------
    n : int
        Size.
    k : int, default 3
        Number of superdiagonals of ones.

    Returns
    -------
    numpy.ndarray, shape (n, n)
        The matrix in float64.
    """
    A = np.eye(n)
    for d in range(1, k + 1):
        A += np.diag(np.ones(n - d), d)
    A += np.diag(-np.ones(n - 1), -1)
    return A


def poisson_1d(n, *, h=None, sparse=False):
    """The 1-D Dirichlet Laplacian, ``tridiag(-1, 2, -1)``, optionally scaled.

    The most-used symmetric positive definite test matrix in the course, and the
    one whose eigenvalues are known in closed form:
    ``lambda_k = 2 - 2 cos(k pi / (n + 1))`` with eigenvectors the discrete
    sine modes. Every iterative method in Volume V is measured against it
    precisely because the answer is known exactly.

    Parameters
    ----------
    n : int
        Number of interior grid points.
    h : float, optional
        Grid spacing; when given, the matrix is divided by ``h**2``.
    sparse : bool, default False
        Return a ``scipy.sparse`` CSR matrix instead of a dense array.

    Returns
    -------
    numpy.ndarray or scipy.sparse.csr_matrix, shape (n, n)
        The discrete Laplacian.
    """
    if sparse:
        from scipy.sparse import diags
        A = diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)],
                  [-1, 0, 1], format="csr")
        return A / h**2 if h else A
    A = (np.diag(2 * np.ones(n)) + np.diag(-np.ones(n - 1), 1)
         + np.diag(-np.ones(n - 1), -1))
    return A / h**2 if h else A


def poisson_2d(n, *, sparse=True):
    """The 2-D five-point Laplacian on an n-by-n grid, as a Kronecker sum.

    Built as ``I (x) L1 + L1 (x) I``, which is both how it is assembled in
    practice and the reason its eigenvalues are the pairwise sums of the 1-D
    ones. It is the course's standard sparse system: large, structured,
    symmetric positive definite, and hopeless to solve densely.

    Parameters
    ----------
    n : int
        Grid points per side; the matrix has ``n**2`` rows.
    sparse : bool, default True
        Return a ``scipy.sparse`` CSR matrix.

    Returns
    -------
    numpy.ndarray or scipy.sparse.csr_matrix, shape (n**2, n**2)
        The discrete 2-D Laplacian.
    """
    if sparse:
        from scipy.sparse import eye as speye, kron
        L1 = poisson_1d(n, sparse=True)
        I = speye(n, format="csr")
        return (kron(I, L1) + kron(L1, I)).tocsr()
    L1 = poisson_1d(n)
    I = np.eye(n)
    return np.kron(I, L1) + np.kron(L1, I)


def random_with_condition(m, n, kappa, rng, *, decay="geometric"):
    """A random matrix with a prescribed condition number.

    Built as ``U S V^T`` from two random orthogonal factors and a chosen
    singular-value spectrum, so the conditioning is exactly what was asked for
    rather than whatever a random draw happened to produce. Essential for any
    experiment that plots an error against ``kappa``.

    Parameters
    ----------
    m, n : int
        Shape.
    kappa : float
        Desired ratio ``sigma_1 / sigma_min``.
    rng : numpy.random.Generator
        Seeded generator; reproducibility is mandatory in this course.
    decay : {"geometric", "linear"}, default "geometric"
        Spacing of the singular values between 1 and ``1/kappa``.

    Returns
    -------
    numpy.ndarray, shape (m, n)
        The matrix, with ``sigma_1 = 1``.
    """
    k = min(m, n)
    if decay == "geometric":
        s = np.logspace(0.0, -np.log10(kappa), k)
    elif decay == "linear":
        s = np.linspace(1.0, 1.0 / kappa, k)
    else:
        raise ValueError("decay must be 'geometric' or 'linear'")
    U, _ = np.linalg.qr(rng.standard_normal((m, m)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    S = np.zeros((m, n))
    S[:k, :k] = np.diag(s)
    return U @ S @ V.T


def random_spd(n, rng, *, kappa=None):
    """A random symmetric positive definite matrix, optionally with a set kappa.

    Positive definiteness is what licenses Cholesky, conjugate gradients, and
    the whole covariance/kernel story, so the course needs a reliable supply of
    such matrices that is not merely "``A^T A`` and hope".

    Parameters
    ----------
    n : int
        Size.
    rng : numpy.random.Generator
        Seeded generator.
    kappa : float, optional
        Desired condition number; when omitted the spectrum is a random draw
        from a Wishart-type construction.

    Returns
    -------
    numpy.ndarray, shape (n, n)
        A symmetric positive definite matrix.
    """
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    if kappa is None:
        lam = rng.uniform(0.5, 2.5, size=n)
    else:
        lam = np.logspace(0.0, -np.log10(kappa), n)
    A = Q @ np.diag(lam) @ Q.T
    return (A + A.T) / 2.0


def low_rank_plus_noise(m, n, r, sigma, rng, *, scale=1.0):
    """A rank-r matrix plus i.i.d. Gaussian noise.

    The model behind every low-rank story in the course: a signal that genuinely
    lives in ``r`` dimensions, observed through noise that lives in all of them.
    Its singular-value spectrum shows the cliff that makes truncation work, and
    the noise floor that says where to stop.

    Parameters
    ----------
    m, n : int
        Shape.
    r : int
        Rank of the signal part.
    sigma : float
        Standard deviation of the additive noise.
    rng : numpy.random.Generator
        Seeded generator.
    scale : float, default 1.0
        Multiplies the signal part.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        The noisy matrix and the exact rank-r signal it was built from.
    """
    L = rng.standard_normal((m, r))
    R = rng.standard_normal((r, n))
    signal = scale * (L @ R) / np.sqrt(r)
    return signal + sigma * rng.standard_normal((m, n)), signal


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

_FLOP_MODELS = {
    "dot":        lambda n: 2 * n,
    "matvec":     lambda m, n: 2 * m * n,
    "matmul":     lambda m, k, n: 2 * m * k * n,
    "outer":      lambda m, n: m * n,
    "lu":         lambda n: 2 * n**3 / 3,
    "cholesky":   lambda n: n**3 / 3,
    "qr":         lambda m, n: 2 * m * n**2 - 2 * n**3 / 3,
    "svd":        lambda m, n: 14 * m * n**2 + 8 * n**3,
    "eigh":       lambda n: 9 * n**3,
    "eig":        lambda n: 25 * n**3,
    "triangular": lambda n: n**2,
    "inverse":    lambda n: 2 * n**3,
}


def flops(op, *shape):
    """Leading-order floating-point operation count for a named operation.

    The course states a cost before it measures one, and this is where the
    stated costs live so no notebook invents its own constant. The counts are
    the standard leading-order figures (Golub & Van Loan, Trefethen & Bau); the
    point is the exponent and the constant's order, not three significant
    figures.

    Parameters
    ----------
    op : str
        One of ``dot``, ``matvec``, ``matmul``, ``outer``, ``lu``, ``cholesky``,
        ``qr``, ``svd``, ``eigh``, ``eig``, ``triangular``, ``inverse``.
    *shape : int
        The dimensions the model needs (see the table in the source).

    Returns
    -------
    float
        Approximate flop count.

    Raises
    ------
    KeyError
        If ``op`` is not a known operation.
    """
    if op not in _FLOP_MODELS:
        raise KeyError(f"unknown operation {op!r}; known: {sorted(_FLOP_MODELS)}")
    return float(_FLOP_MODELS[op](*shape))


def cost_table(rows, *, headers=("operation", "flops", "measured (s)", "exponent")):
    """Render a cost comparison as a plain-text table.

    Parameters
    ----------
    rows : sequence of sequence
        Row values; each row is formatted with ``str``.
    headers : sequence of str
        Column headers.

    Returns
    -------
    str
        The formatted table, ready to ``print``.
    """
    body = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in body)) if body else len(h)
              for i, h in enumerate(headers)]
    line = "  ".join("-" * w for w in widths)
    out = ["  ".join(h.ljust(w) for h, w in zip(headers, widths)), line]
    out += ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in body]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Offline datasets (written out explicitly; nothing here touches the network)
# ---------------------------------------------------------------------------

def toy_corpus():
    """Twelve short documents on three planted topics, for term-document work.

    Written out in full rather than downloaded so the notebooks run offline, in
    CI, and identically for every reader. The three topics (linear algebra,
    cooking, sailing) share almost no vocabulary, so a truncated SVD of the
    term-document matrix must separate them cleanly — which is exactly what
    makes the latent-semantic-analysis exercise checkable.

    Returns
    -------
    tuple of (list of str, list of int)
        The documents and their planted topic labels (0, 1, or 2).
    """
    docs = [
        "the matrix has orthogonal columns and a small condition number",
        "eigenvalues of a symmetric matrix are real and its eigenvectors orthogonal",
        "the singular value decomposition factors any matrix into orthogonal factors",
        "elimination factors the matrix into a lower and an upper triangular matrix",
        "simmer the onion and garlic in olive oil then add the tomato",
        "knead the dough until smooth and let it rise for one hour",
        "season the soup with salt pepper and a little fresh thyme",
        "roast the vegetables in a hot oven until the edges caramelise",
        "trim the mainsail and bear away as the wind backs to the north",
        "the tide runs hard against the headland at the top of the ebb",
        "reef early when the wind pipes up and the sea state builds",
        "take a bearing on the lighthouse and plot the fix on the chart",
    ]
    labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    return docs, labels


def toy_web():
    """A six-page web with a dangling node, for PageRank.

    Small enough to draw and to diagonalise exactly, and deliberately built with
    the two pathologies that motivate Google's damping factor: page 4 links
    nowhere (a dangling node, so the transition matrix is not stochastic) and
    pages 4-5 form a group with no link back to the rest.

    Returns
    -------
    tuple of (list of tuple of int, numpy.ndarray)
        The directed edge list ``(from, to)`` and 2-D node positions for drawing.
    """
    edges = [(0, 1), (0, 2), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 4)]
    pos = np.array([[0.0, 1.0], [1.0, 1.6], [1.0, 0.4], [2.0, 1.0],
                    [3.0, 1.5], [3.0, 0.5]])
    return edges, pos


def toy_graph():
    """An eight-node graph with two planted clusters joined by one bridge.

    The Laplacian's Fiedler vector must split it along the bridge, which makes
    spectral bisection checkable against a known answer rather than eyeballed.

    Returns
    -------
    tuple of (list of tuple of int, numpy.ndarray, numpy.ndarray)
        The undirected edge list, 2-D node positions, and the planted cluster
        labels (0 or 1).
    """
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3),
             (3, 4),
             (4, 5), (4, 6), (5, 6), (5, 7), (6, 7)]
    pos = np.array([[0.0, 1.0], [0.7, 1.7], [0.7, 0.3], [1.5, 1.0],
                    [2.5, 1.0], [3.3, 1.7], [3.3, 0.3], [4.0, 1.0]])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return edges, pos, labels


# ---------------------------------------------------------------------------
# Memory and shape (Volume 0)
# ---------------------------------------------------------------------------

def memory_layout(ax, shape, order="C", *, title=None, cell=0.62):
    """Draw a 2-D array beside the flat buffer it actually lives in.

    An array is a one-dimensional block of memory plus a rule for turning an
    index pair into an offset into that block. Drawing both the logical grid and
    the physical buffer, with each cell carrying its flat offset, makes the rule
    visible: C order walks a row before moving down, Fortran order walks a
    column before moving right, and a transpose changes only the rule, never the
    buffer.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    shape : tuple of int
        The array's ``(rows, cols)``.
    order : {"C", "F"}, default "C"
        Memory order: row-major or column-major.
    title : str, optional
        Axes title; defaults to naming the order.
    cell : float, default 0.62
        Cell side length in data units.

    Returns
    -------
    numpy.ndarray, shape (rows, cols)
        The flat offset of each logical index, so a caller can assert against it.
    """
    m, n = shape
    offsets = (np.arange(m * n).reshape(m, n) if order == "C"
               else np.arange(m * n).reshape(n, m).T)
    for i in range(m):
        for j in range(n):
            x, y = j * cell, -i * cell
            shade = 0.10 + 0.30 * (i / max(m - 1, 1) if order == "C"
                                   else j / max(n - 1, 1))
            ax.add_patch(Rectangle((x, y), cell, cell, facecolor=ACCENT,
                                   alpha=shade, edgecolor=INK, linewidth=1.1))
            ax.text(x + cell / 2, y + cell / 2, str(offsets[i, j]),
                    ha="center", va="center", fontsize=10, color=INK,
                    gid="_nocheck")
    ax.text(n * cell / 2, cell * 1.25, f"logical index $(i, j)$, {order} order",
            ha="center", va="bottom", fontsize=11, color=INK, gid="_nocheck")

    # The physical buffer: one contiguous strip, offsets in increasing order.
    y0 = -(m + 1.4) * cell
    for k in range(m * n):
        i, j = divmod(k, n) if order == "C" else (k % m, k // m)
        shade = 0.10 + 0.30 * (i / max(m - 1, 1) if order == "C"
                               else j / max(n - 1, 1))
        ax.add_patch(Rectangle((k * cell, y0), cell, cell, facecolor=ACCENT,
                               alpha=shade, edgecolor=INK, linewidth=1.1))
        ax.text(k * cell + cell / 2, y0 + cell / 2, str(k), ha="center",
                va="center", fontsize=9, color=INK, gid="_nocheck")
    ax.text(m * n * cell / 2, y0 - 0.25 * cell,
            "one contiguous buffer of $mn$ elements", ha="center", va="top",
            fontsize=11, color=SOFT, gid="_nocheck")

    ax.set_xlim(-0.4 * cell, m * n * cell + 0.4 * cell)
    ax.set_ylim(y0 - 1.1 * cell, 2.1 * cell)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title if title is not None
                 else f"{order}-order layout of a {m}$\\times${n} array")
    return offsets


def broadcast_diagram(ax, shapes, *, title=None, names=None):
    """Draw the right-aligned shape table NumPy uses to broadcast.

    Broadcasting is decided by one rule applied right to left: two dimensions are
    compatible if they are equal or one of them is 1, and a 1 is stretched to
    match. Writing the shapes right-aligned in a table is exactly how the rule is
    applied by hand, so the diagram is the algorithm rather than an illustration
    of it. Stretched dimensions are marked in amber; the result row is the
    elementwise maximum.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    shapes : sequence of tuple of int
        The operand shapes.
    title : str, optional
        Axes title.
    names : sequence of str, optional
        Row labels; defaults to ``A``, ``B``, ….

    Returns
    -------
    tuple of int
        The broadcast result shape.

    Raises
    ------
    ValueError
        If the shapes do not broadcast.
    """
    shapes = [tuple(s) for s in shapes]
    ndim = max(len(s) for s in shapes)
    padded = [(1,) * (ndim - len(s)) + s for s in shapes]
    result = []
    for k in range(ndim):
        dims = {p[k] for p in padded}
        non_unit = dims - {1}
        if len(non_unit) > 1:
            raise ValueError(f"shapes {shapes} do not broadcast at axis {k}")
        result.append(non_unit.pop() if non_unit else 1)
    result = tuple(result)

    if names is None:
        names = [chr(ord("A") + i) for i in range(len(shapes))]
    rows = list(zip(names, padded, shapes)) + [("result", result, result)]
    w, h = 1.0, 0.62
    for r, (name, padded_shape, orig) in enumerate(rows):
        y = -r * h
        is_result = r == len(rows) - 1
        ax.text(-0.45, y + h / 2, name, ha="right", va="center", fontsize=12,
                color=INK, fontweight="bold" if is_result else "normal",
                gid="_nocheck")
        for k, d in enumerate(padded_shape):
            x = k * w
            stretched = (not is_result) and d == 1 and result[k] != 1
            implied = (not is_result) and k < ndim - len(orig)
            face = ACCENT if stretched else (PANEL if not is_result else "white")
            alpha = 0.42 if stretched else (1.0 if not is_result else 1.0)
            ax.add_patch(Rectangle((x, y), w * 0.86, h * 0.82, facecolor=face,
                                   alpha=alpha, edgecolor=INK,
                                   linewidth=2.0 if is_result else 1.2,
                                   linestyle=":" if implied else "-"))
            ax.text(x + w * 0.43, y + h * 0.41, str(d), ha="center", va="center",
                    fontsize=12, color=SOFT if implied else INK,
                    gid="_nocheck")
    ax.text(ndim * w * 0.5, h * 1.15, "aligned from the right  $\\longrightarrow$",
            ha="center", va="bottom", fontsize=11, color=SOFT, gid="_nocheck")
    ax.set_xlim(-1.5, ndim * w + 0.2)
    ax.set_ylim(-len(rows) * h - 0.1, h * 2.1)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return result
