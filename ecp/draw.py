"""
ecp.draw
========

Schematic-diagram primitives for problem statements. Centralised so a labeled
setup figure (pendulum with θ, ℓ, mg; Atwood machine; spring chain; field
sketch) is ~15 lines in a notebook and inherits the series palette/fonts.

These draw *schematics*, not data plots. Use in a solution-tagged cell whose
OUTPUT (the figure) appears in the public problem statement while the drawing
code stays hidden.

Convention: pass an Axes; helpers draw onto it in data coordinates. Call
ecp.draw.finish(ax) to equalise aspect and strip the frame for a clean diagram.

This module is THE home for schematic drawing across the series; extend it with
new *general* primitives as later volumes need them (springs, charges, field
lines, axes, circuit elements). Name primitives by what they draw, never by the
system that first needed them.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, Circle, Ellipse, Rectangle

INK = "#16213e"; ACCENT = "#c0851a"; SOFT = "#46506b"; PANEL = "#faf7f0"


# ---------------------------------------------------------------------------
# Collision-free label placement (bbox-measured, deferred to finish())
# ---------------------------------------------------------------------------
# A label must never land on an arrow, line, curve, circle, or another label —
# a collision is a build-failing defect, not a cosmetic nit (see
# `assert_no_collisions`). Fixed offsets cannot know what occupies the target
# spot, so placement here is MEASURED: each label's rendered bounding box (read
# via `get_window_extent`) is tested against (i) sample points along every drawn
# geometry artist and (ii) every already-placed label's bbox, trying a ranked
# list of candidate positions until one is clear, falling back to a leader line.
#
# Placement is DEFERRED: the labelling helpers (point, bob, vector, angle_arc,
# dim_label, axes, place_label) RECORD a request and draw their geometry now;
# `finish()` resolves every label last, once all geometry exists and the final
# (equal-aspect) data->display transform is fixed. So every schematic must end
# with `draw.finish(ax)`; that is also where you should call
# `draw.assert_no_collisions(ax)` to enforce the gate.

from matplotlib.transforms import Bbox

_REQUESTS: dict[int, list] = {}   # per-axes deferred label requests
_OBST: dict[int, list] = {}       # per-axes registered obstacle segments (data coords)


def _reset_labels(ax):
    _REQUESTS[id(ax)] = []
    _OBST[id(ax)] = []


def _register_segment(ax, p0, p1):
    """Record a straight obstacle segment (an arrow/line shaft) in data coords.

    Arrows are `FancyArrowPatch`es, whose `get_verts()` returns only a coarse
    bounding quad; registering the true endpoints lets labels avoid the shaft
    exactly. Plotted lines (`ax.plot`) and outline patches are sampled directly.
    """
    _OBST.setdefault(id(ax), []).append((np.asarray(p0, float), np.asarray(p1, float)))


def _densify(pts, step=6.0):
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return pts
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        k = max(2, int(np.hypot(*(b - a)) / step))
        out.append(np.linspace(a, b, k))
    return np.vstack(out)


def _obstacle_points(ax, rend):
    """Display-coord sample points along every drawn geometry artist."""
    out = []
    for ln in ax.lines:
        if ln.get_gid() in ("_leader", "_grid"):
            continue   # leaders and the faint background grid are not obstacles
        d = ax.transData.transform(ln.get_xydata())
        if len(d) == 0:
            continue
        # A marker-only series (e.g. a dotted reference grid) is a set of isolated
        # points, NOT a polyline — densifying between them would fabricate lines.
        if ln.get_linestyle() in ("None", "none", "", " ") or not ln.get_linestyle():
            out.append(np.asarray(d))
        elif len(d) >= 2:
            out.append(_densify(d))
    for p in ax.patches:
        if isinstance(p, FancyArrowPatch):
            continue                      # registered explicitly instead
        try:
            v = np.asarray(p.get_verts())  # already display coords
        except Exception:
            continue
        if len(v) >= 2:
            out.append(_densify(v))
    for p0, p1 in _OBST.get(id(ax), []):
        out.append(_densify(ax.transData.transform(np.vstack([p0, p1]))))
    return out


def _label_bbox(t, rend, pad=2.0):
    b = t.get_window_extent(rend)
    return Bbox.from_extents(b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad)


def _hits_geometry(bb, obst):
    return any(bb.contains(x, y) for pts in obst for x, y in pts)


def _align(d):
    ha = "left" if d[0] > 0.25 else "right" if d[0] < -0.25 else "center"
    va = "bottom" if d[1] > 0.25 else "top" if d[1] < -0.25 else "center"
    return ha, va


def _candidates(normal, clearance, kind, perp):
    """Ranked (direction, distance) candidates. Dimension labels step
    perpendicular to their line (both sides, never along it); other labels try
    the preferred normal first, then rotate outward, at growing distance."""
    c = clearance
    if kind == "dim":
        p = perp if perp is not None else np.array([-normal[1], normal[0]])
        p = p / (np.hypot(*p) or 1.0)
        out = []
        for d in (c, 1.6 * c, 2.4 * c, 3.4 * c):
            out += [(p, d), (-p, d)]
        return out
    # Sweep many angles around the preferred normal so a label can thread a gap
    # between n-fold-symmetric obstacles (e.g. a charge ringed by radial arrows),
    # widening the search outward in distance.
    n = np.asarray(normal, float)
    a0 = np.arctan2(n[1], n[0])
    offsets = [0.0]
    for step in (20, 40, 60, 80, 100, 120, 140, 160, 180):  # ± fan, degrees
        offsets += [np.radians(step), -np.radians(step)]
    dirs = [np.array([np.cos(a0 + o), np.sin(a0 + o)]) for o in offsets]
    return [(dr, d) for d in (c, 1.5 * c, 2.2 * c, 3.0 * c, 4.0 * c) for dr in dirs]


def _draw_leader(ax, anchor, pos, color, z):
    ln, = ax.plot([anchor[0], pos[0]], [anchor[1], pos[1]], "-", lw=0.8,
                  color=color, alpha=0.6, zorder=z - 1)
    ln.set_gid("_leader")


def _place_one(ax, rend, req, obst, placed):
    anchor = req["anchor"]
    n = np.asarray(req["normal"], float)
    n = n / (np.hypot(*n) or 1.0)
    t = ax.annotate(req["text"], (anchor[0], anchor[1]), fontsize=req["fontsize"],
                    color=req["color"], zorder=req["z"])
    if req.get("nocheck"):
        t.set_gid("_nocheck")
    cands = _candidates(n, req["clearance"], req["kind"], req.get("perp"))
    chosen = None
    for dr, dist in cands:
        pos = anchor + dr * dist
        ha = req.get("ha") or _align(dr)[0]
        va = req.get("va") or _align(dr)[1]
        t.set_position(pos)
        t.set_ha(ha)
        t.set_va(va)
        bb = _label_bbox(t, rend)
        if not _hits_geometry(bb, obst) and not any(bb.overlaps(pb) for pb in placed):
            chosen = (pos, bb)
            break
    if chosen is None:
        # Leader-line fallback: search a wide ring for open space.
        for dist in (3.0, 4.0, 5.0, 6.0):
            for k in range(16):
                a = 2 * np.pi * k / 16
                dr = np.array([np.cos(a), np.sin(a)])
                pos = anchor + dr * dist * req["clearance"]
                ha, va = _align(dr)
                t.set_position(pos)
                t.set_ha(ha)
                t.set_va(va)
                bb = _label_bbox(t, rend)
                if not _hits_geometry(bb, obst) and not any(bb.overlaps(pb) for pb in placed):
                    chosen = (pos, bb)
                    break
            if chosen:
                break
        if chosen is None:                       # last resort: farthest preferred
            pos = anchor + n * 4.0 * req["clearance"]
            t.set_position(pos)
            chosen = (pos, _label_bbox(t, rend))
        _draw_leader(ax, anchor, chosen[0], req["color"], req["z"])
    elif req.get("leader"):
        _draw_leader(ax, anchor, chosen[0], req["color"], req["z"])
    placed.append(chosen[1])


def _resolve_labels(ax):
    reqs = _REQUESTS.get(id(ax), [])
    if not reqs:
        return
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    obst = _obstacle_points(ax, rend)
    placed = []
    for req in reqs:
        _place_one(ax, rend, req, obst, placed)
    _REQUESTS[id(ax)] = []


def place_label(ax, text, anchor, normal=(0.0, 1.0), clearance=0.12,
                color=INK, fontsize=13, min_sep=0.13, ha=None, va=None,
                z=5, leader=False, kind="point", perp=None, nocheck=False):
    """Record a label near an anchor, to be placed collision-free by :func:`finish`.

    Placement is deferred: the request is stored now and resolved last, once all
    geometry exists, by measuring the label's rendered bounding box against every
    artist and every other label, so a label can never land on an arrow or a curve.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes the label belongs to.
    text : str
        The label text (math allowed).
    anchor : tuple of float
        ``(x, y)`` point the label annotates.
    normal : tuple of float, optional
        Preferred outward direction from the anchor (default ``(0, 1)``).
    clearance : float, optional
        Starting offset from the anchor in data units (default ``0.12``).
    color : str, optional
        Text colour (default the series ink).
    fontsize : int, optional
        Font size (default ``13``).
    min_sep : float, optional
        Accepted for backward compatibility and ignored.
    ha, va : str, optional
        Explicit horizontal/vertical alignment; chosen automatically if ``None``.
    z : int, optional
        Draw z-order (default ``5``).
    leader : bool, optional
        Force a leader line, e.g. for a label parked in open space (default ``False``).
    kind : str, optional
        ``"point"`` (default) or ``"dim"``; ``"dim"`` steps PERPENDICULAR to the
        annotated line, never along it.
    perp : tuple of float, optional
        Explicit perpendicular direction for a dimension label.
    nocheck : bool, optional
        Exempt this label from the collision gate (default ``False``).

    Returns
    -------
    None
        Records the request; the label is drawn by :func:`finish`.
    """
    _REQUESTS.setdefault(id(ax), []).append(dict(
        text=text, anchor=np.asarray(anchor, float), normal=normal,
        clearance=clearance, color=color, fontsize=fontsize, ha=ha, va=va,
        z=z, leader=leader, kind=kind, perp=perp, nocheck=nocheck))


def point(ax, p, r=0.028, color=INK, label=None, normal=(0.0, 1.0),
          clearance=0.10, z=6):
    """A small filled marked point, a concrete target for dimension lines and leaders.

    Use it for an ellipse centre, an orbit vertex, a focus, or a charge: a
    :func:`dim_label` should terminate on points like these, never float in space.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    p : tuple of float
        ``(x, y)`` location of the point.
    r : float, optional
        Marker radius in data units (default ``0.028``).
    color : str, optional
        Fill colour (default the series ink).
    label : str, optional
        If given, a collision-free label placed via :func:`place_label`.
    normal : tuple of float, optional
        Preferred label direction (default ``(0, 1)``).
    clearance : float, optional
        Label offset in data units (default ``0.10``).
    z : int, optional
        Draw z-order (default ``6``).

    Returns
    -------
    None
        Draws the point (and optional label) onto ``ax`` in place.
    """
    ax.add_patch(Circle(p, r, facecolor=color, edgecolor="none", zorder=z))
    if label:
        place_label(ax, label, p, normal=normal, clearance=clearance,
                    color=color, z=z + 1)


def new_diagram(figsize=(4.2, 4.2)):
    """Create a fresh figure and axes for a schematic, with a clean label register.

    The entry point for every schematic: it makes the figure and clears any deferred
    label requests left on the axes, so :func:`place_label` and :func:`finish` start
    from a known state.

    Parameters
    ----------
    figsize : tuple of float, optional
        Figure size in inches (default ``(4.2, 4.2)``).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The new figure.
    ax : matplotlib.axes.Axes
        Its axes, ready for the drawing primitives.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _reset_labels(ax)
    return fig, ax


def finish(ax, pad=0.15, check=True):
    """Equalise aspect, strip the frame, resolve all deferred labels, then run
    the collision gate.

    Labels are placed LAST so each one can be measured against the final
    geometry and transform. With ``check=True`` (the default) a leftover
    label–geometry or label–label overlap raises immediately, exactly like a
    failed `validate.check`: the cell errors and the build fails, so a label on
    an arrow can never ship. Pass ``check=False`` only for a figure with a
    deliberate on-geometry label that cannot be exempted with ``nocheck``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The schematic's axes.
    pad : float, optional
        Fractional margin added around the content (default ``0.15``).
    check : bool, optional
        Run the blocking collision gate after placing labels (default ``True``).

    Returns
    -------
    None
        Finalises ``ax`` in place.

    Raises
    ------
    AssertionError
        If ``check`` is true and any label overlaps geometry or another label.
    """
    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(pad)
    _resolve_labels(ax)
    if check:
        assert_no_collisions(ax)


def detect_collisions(ax, pad=2.0):
    """Find every label whose rendered box overlaps geometry or another label.

    The inspection half of the diagram QA gate: it draws the figure, reads each
    label's pixel bounding box, and tests it against sample points on every artist
    and every other label.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The schematic's axes.
    pad : float, optional
        Padding in points added around each label box before testing (default ``2.0``).

    Returns
    -------
    list of tuple
        ``(label_text, what_it_hits)`` for each collision, where ``what_it_hits`` is
        ``"geometry"`` or ``"label:<text>"``. An empty list means the figure is clean.
    """
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    obst = _obstacle_points(ax, rend)
    items = [t for t in ax.texts
             if t.get_text().strip() and t.get_gid() != "_nocheck"]
    bbs = [(t, _label_bbox(t, rend, pad)) for t in items]
    issues = []
    for i, (t, bb) in enumerate(bbs):
        if _hits_geometry(bb, obst):
            issues.append((t.get_text(), "geometry"))
        for j, (t2, bb2) in enumerate(bbs):
            if i < j and bb.overlaps(bb2):
                issues.append((t.get_text(), "label:" + t2.get_text()))
    return issues


def assert_no_collisions(ax, pad=2.0):
    """Blocking QA gate: raise if any label overlaps geometry or another label.

    Run automatically by :func:`finish`. A detected collision is treated exactly
    like a failed ``validate.check``: the cell errors and the build fails, so a
    label on an arrow can never ship.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The schematic's axes.
    pad : float, optional
        Padding in points around each label box (default ``2.0``).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If :func:`detect_collisions` finds any overlap, with a message listing each.
    """
    issues = detect_collisions(ax, pad=pad)
    if issues:
        lines = "\n".join(f"  - {lab!r} overlaps {what}" for lab, what in issues)
        raise AssertionError(
            f"Label collision(s) detected in schematic ({len(issues)}):\n{lines}\n"
            "Reposition (perpendicular offset for dimensions, a different normal, "
            "or a leader line) until detect_collisions(ax) is empty.")


def rod(ax, p0, p1, lw=2.2, color=INK, z=2):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "-", lw=lw, color=color, zorder=z,
            solid_capstyle="round")


def bob(ax, p, r=0.07, color=INK, label=None, dx=0.10, dy=0.0, normal=None):
    """A filled bob. The label is placed collision-free along `normal` (default
    the (dx, dy) direction, for backward compatibility); pass `normal` to steer
    it explicitly (e.g. away from a rod)."""
    ax.add_patch(Circle(p, r, facecolor=color, edgecolor="none", zorder=4))
    if label:
        nrm = normal if normal is not None else (dx, dy)
        if np.hypot(*nrm) == 0:
            nrm = (1.0, 0.0)
        clearance = max(np.hypot(dx, dy), r + 0.06)
        place_label(ax, label, p, normal=nrm, clearance=clearance, color=INK)


def pivot(ax, p, r=0.04):
    ax.add_patch(Circle(p, r, facecolor="white", edgecolor=INK, lw=1.6, zorder=5))


def ceiling(ax, x0, x1, y, n=14, h=0.06):
    ax.plot([x0, x1], [y, y], "-", lw=2, color=INK, zorder=2)
    xs = np.linspace(x0, x1, n)
    for x in xs:
        ax.plot([x, x - h], [y, y + h], "-", lw=1, color=SOFT, zorder=1)


def vector(ax, p0, p1, label=None, color=ACCENT, lw=2.2, dx=0.06, dy=-0.06,
           normal=None):
    """An annotated arrow from one point to another (a force, a velocity, a field).

    The shaft is registered as a collision obstacle, so subsequent labels route
    around it; the arrow's own label sits just beyond the tip by default.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    p0, p1 : tuple of float
        Tail and head ``(x, y)`` of the arrow.
    label : str, optional
        Label placed near the head via :func:`place_label`.
    color : str, optional
        Arrow colour (default the series accent).
    lw : float, optional
        Shaft line width (default ``2.2``).
    dx, dy : float, optional
        Fallback label direction when ``p0 == p1`` (defaults ``0.06``, ``-0.06``).
    normal : tuple of float, optional
        Explicit label direction, e.g. to clear a nearby bob or rod.

    Returns
    -------
    None
        Draws the arrow (and optional label) onto ``ax`` in place.
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    ax.add_patch(FancyArrowPatch(tuple(p0), tuple(p1), arrowstyle="-|>",
                                 mutation_scale=16, lw=lw, color=color, zorder=4,
                                 shrinkA=0, shrinkB=0))
    _register_segment(ax, p0, p1)
    if label:
        if normal is not None:
            nrm = normal
        else:
            d = p1 - p0
            nrm = d if np.hypot(*d) > 0 else (dx, dy)
        place_label(ax, label, p1, normal=nrm, clearance=0.12, color=color)


def angle_arc(ax, vertex, start_deg, end_deg, r=0.35, label=None, color=SOFT,
              label_r=0.62):
    """An arc marking an angle at `vertex`. The label sits along the arc's
    bisector at radius `label_r * r` — inside the arc by default (`label_r < 1`),
    which reads cleanly for tight angles — nudged clear of other labels."""
    # Arc draws CCW from theta1 to theta2, so a negative sweep (end < start)
    # would trace the major arc the long way round; order them to get the minor arc.
    t1, t2 = (start_deg, end_deg) if end_deg >= start_deg else (end_deg, start_deg)
    ax.add_patch(Arc(vertex, 2 * r, 2 * r, angle=0, theta1=t1,
                     theta2=t2, color=color, lw=1.6, zorder=3))
    if label:
        mid = np.radians(0.5 * (start_deg + end_deg))
        nrm = (np.cos(mid), np.sin(mid))
        place_label(ax, label, vertex, normal=nrm, clearance=label_r * r,
                    color=color, ha="center", va="center")


def dim_label(ax, p0, p1, label, color=SOFT, off=0.0, mark_ends=False,
              clearance=0.11):
    """A double-headed dimension line measuring the distance between two points.

    The label is offset PERPENDICULAR to the line, never centred on it, so it never
    sits on the arrow; ``mark_ends`` draws a point at each end so the dimension
    visibly *terminates on drawn points* (an ellipse centre and a vertex, say)
    rather than floating.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    p0, p1 : tuple of float
        The two endpoints being measured.
    label : str
        The dimension label (e.g. ``"$a$"``).
    color : str, optional
        Line and label colour (default the series soft grey).
    off : float, optional
        Extra perpendicular offset of the whole dimension line (default ``0.0``).
    mark_ends : bool, optional
        Draw a small point at each endpoint (default ``False``).
    clearance : float, optional
        Perpendicular label offset in data units (default ``0.11``).

    Returns
    -------
    None
        Draws the dimension line and label onto ``ax`` in place.
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    ax.add_patch(FancyArrowPatch(tuple(p0), tuple(p1), arrowstyle="<|-|>",
                                 mutation_scale=10, lw=1.2, color=color, zorder=3,
                                 shrinkA=0, shrinkB=0))
    _register_segment(ax, p0, p1)
    if mark_ends:
        point(ax, p0, r=0.022, color=color, z=4)
        point(ax, p1, r=0.022, color=color, z=4)
    d = p1 - p0
    perp = np.array([-d[1], d[0]])
    if np.hypot(*perp) > 0:
        perp = perp / np.hypot(*perp)
    if off:  # caller-specified perpendicular bias (sign picks the side)
        perp = perp * np.sign(off)
    mid = 0.5 * (p0 + p1)
    place_label(ax, label, mid, normal=perp, clearance=clearance, color=color,
                kind="dim", perp=perp, ha="center", va="center")


# ---------------------------------------------------------------------------
# General extensions. Keep these system-agnostic; name them by what they draw.
# ---------------------------------------------------------------------------

def ground(ax, x0, x1, y, n=14, h=0.06):
    """A solid horizontal support line with hatching below it (the floor)."""
    ax.plot([x0, x1], [y, y], "-", lw=2, color=INK, zorder=2)
    for x in np.linspace(x0, x1, n):
        ax.plot([x, x - h], [y, y - h], "-", lw=1, color=SOFT, zorder=1)


def wall(ax, y0, y1, x, n=14, h=0.06, side="left"):
    """A solid vertical support line with hatching to one side (an anchor wall).

    ``side="left"`` hatches to the left (a wall on the right of the system);
    ``side="right"`` hatches to the right (a wall on the left of the system)."""
    s = -1.0 if side == "left" else 1.0
    ax.plot([x, x], [y0, y1], "-", lw=2, color=INK, zorder=2)
    for yy in np.linspace(y0, y1, n):
        ax.plot([x, x + s * h], [yy, yy + h], "-", lw=1, color=SOFT, zorder=1)


def spring(ax, p0, p1, coils=8, width=0.08, lead=0.15, lw=1.8, color=INK, z=2):
    """A zig-zag spring between p0 and p1, with short straight leads at each end."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    length = np.hypot(*d)
    u = d / length                       # axial unit vector
    perp = np.array([-u[1], u[0]])       # perpendicular unit vector
    a = p0 + u * lead
    b = p1 - u * lead
    seg = np.linspace(0.0, 1.0, 2 * coils + 1)
    pts = [p0]
    for i, s in enumerate(seg):
        base = a + (b - a) * s
        off = 0.0 if (i == 0 or i == len(seg) - 1) else (width if i % 2 else -width)
        pts.append(base + perp * off)
    pts.append(p1)
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], "-", lw=lw, color=color, zorder=z,
            solid_capstyle="round")


def mass_box(ax, center, w=0.3, h=0.3, label=None, color=INK, z=4):
    """A filled rectangular mass with an optional centred (white) label."""
    cx, cy = center
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor=color,
                           edgecolor="none", zorder=z))
    if label:
        # Centred ON the box by design (white on dark) — exempt from the gate.
        a = ax.annotate(label, (cx, cy), fontsize=12, color="white", ha="center",
                        va="center", zorder=z + 1)
        a.set_gid("_nocheck")


def pulley(ax, center, r=0.12, color=INK, z=3):
    """A pulley wheel (rim + axle) for ropes/strings to pass over."""
    ax.add_patch(Circle(center, r, facecolor="white", edgecolor=color, lw=2, zorder=z))
    ax.add_patch(Circle(center, 0.02, facecolor=color, edgecolor="none", zorder=z + 1))


def axes(ax, origin, length, labels=("x", "y"), color=SOFT, z=3):
    """A pair of coordinate axes (two arrows) from an origin, with labelled tips.

    Both shafts register as collision obstacles, and the axis-name labels route
    through the collision-free placer just beyond each arrowhead.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    origin : tuple of float
        ``(x, y)`` foot of both arrows.
    length : float
        Length of each axis arrow.
    labels : tuple of str, optional
        Axis names at the tips (default ``("x", "y")``).
    color : str, optional
        Arrow and label colour (default the series soft grey).
    z : int, optional
        Draw z-order (default ``3``).

    Returns
    -------
    None
        Draws the axes onto ``ax`` in place.
    """
    ox, oy = origin
    ax.add_patch(FancyArrowPatch((ox, oy), (ox + length, oy), arrowstyle="-|>",
                                 mutation_scale=14, lw=1.6, color=color, zorder=z,
                                 shrinkA=0, shrinkB=0))
    ax.add_patch(FancyArrowPatch((ox, oy), (ox, oy + length), arrowstyle="-|>",
                                 mutation_scale=14, lw=1.6, color=color, zorder=z,
                                 shrinkA=0, shrinkB=0))
    _register_segment(ax, (ox, oy), (ox + length, oy))
    _register_segment(ax, (ox, oy), (ox, oy + length))
    place_label(ax, labels[0], (ox + length, oy), normal=(1.0, 0.0),
                clearance=0.07, color=color, z=z + 1)
    place_label(ax, labels[1], (ox, oy + length), normal=(0.0, 1.0),
                clearance=0.07, color=color, z=z + 1)


def grid(ax, xlim, ylim, *, step=1.0, labels=("x", "y"), color="#cdc6b6", z=-5):
    """A faint ruled background grid with labelled axes, for COORDINATE-bearing
    setup figures — so a reader can see a charge sits at, say, (0, 0.55), not
    floating in space.

    Draw this FIRST (before any objects), then the usual primitives, then
    :func:`finish`. Thin, low-contrast lines on a ``step`` lattice span
    ``xlim=(xmin, xmax)`` and ``ylim=(ymin, ymax)`` behind everything; the x- and
    y-axes through the origin are drawn a touch darker and carry the axis-name
    ``labels`` at their positive ends. Grid lines are background, so they are
    EXEMPT from the collision gate (a label may cross a faint gridline); object
    and dimension labels still route through the collision-free placer as usual.

    Use only for figures that assert WHERE things are (charges at stated
    positions, a field point at a coordinate). Leave purely conceptual sketches (a
    generic Gaussian surface, an abstract field-line sketch, a stencil) clean —
    they do not need a grid. See the diagram conventions in NOTEBOOK_STYLE.md.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    xlim, ylim : tuple of float
        ``(min, max)`` extents of the grid in data coordinates.
    step : float, optional
        Spacing between gridlines (default ``1.0``).
    labels : tuple of str, optional
        Axis names placed at the positive ends, e.g. ``("x", "y")``; an empty
        string omits one.
    color : str, optional
        Gridline colour (default a faint parchment grey).
    z : int, optional
        Draw z-order, kept well below the content (default ``-5``).

    Returns
    -------
    None
        Draws the grid and axis labels onto ``ax`` in place.
    """
    xmin, xmax = xlim
    ymin, ymax = ylim
    xs = np.arange(np.ceil(xmin / step) * step, xmax + 0.5 * step, step)
    ys = np.arange(np.ceil(ymin / step) * step, ymax + 0.5 * step, step)
    for x in xs:
        ln, = ax.plot([x, x], [ymin, ymax], "-", color=color, lw=0.6, zorder=z)
        ln.set_gid("_grid")
    for y in ys:
        ln, = ax.plot([xmin, xmax], [y, y], "-", color=color, lw=0.6, zorder=z)
        ln.set_gid("_grid")
    if xmin <= 0 <= xmax:                       # y-axis (x = 0), slightly darker
        ln, = ax.plot([0, 0], [ymin, ymax], "-", color=SOFT, lw=1.1, alpha=0.55,
                      zorder=z + 1)
        ln.set_gid("_grid")
    if ymin <= 0 <= ymax:                       # x-axis (y = 0)
        ln, = ax.plot([xmin, xmax], [0, 0], "-", color=SOFT, lw=1.1, alpha=0.55,
                      zorder=z + 1)
        ln.set_gid("_grid")
    if labels[0]:
        place_label(ax, labels[0], (xmax, 0.0), normal=(1.0, 0.0),
                    clearance=0.06, color=SOFT)
    if labels[1]:
        place_label(ax, labels[1], (0.0, ymax), normal=(0.0, 1.0),
                    clearance=0.06, color=SOFT)


def solenoid(ax, x0, x1, y, radius, n_turns, *, shear=0.65, color=INK, lw=2.0,
             z=2):
    """The conventional solenoid schematic: a wound coil, immediately recognisable
    as a solenoid rather than the row of circles a cross-section would give.

    A helix is drawn along the axis from ``x0`` to ``x1`` at height ``y`` with
    vertical radius ``radius`` and ``n_turns`` loops, projected with a sideways
    ``shear`` so each turn becomes a slanted ellipse and consecutive turns OVERLAP
    — the "slinky" look. The near side of every turn is solid and in front; the
    receding far side is faint and behind, so the eye reads one three-dimensional
    coil. A general primitive for solenoids, inductors, and induction coils
    (reused in 3.7 and 3.11). Overlay an Amperian loop, current, or field with the
    usual helpers and call :func:`finish` after.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    x0, x1 : float
        Start and end of the coil along its axis.
    y : float
        Height of the coil's axis.
    radius : float
        Vertical radius of each loop.
    n_turns : int
        Number of loops.
    shear : float, optional
        Sideways shear that turns each turn into an overlapping ellipse, the
        "slinky" depth cue (default ``0.65``).
    color : str, optional
        Coil colour (default the series ink).
    lw : float, optional
        Line width of the near side (the far side is drawn lighter, default ``2.0``).
    z : int, optional
        Draw z-order (default ``2``).

    Returns
    -------
    None
        Draws the coil onto ``ax`` in place.
    """
    n = max(int(n_turns), 1)
    theta = np.linspace(0.0, 2.0 * np.pi * n, 90 * n)
    axis = x0 + (x1 - x0) * theta / (2.0 * np.pi * n)
    xs = axis + shear * radius * np.sin(theta)  # shear turns each loop into an ellipse
    ys = y + radius * np.cos(theta)
    near = np.sin(theta) >= 0.0                 # near side faces the viewer

    def _runs(mask):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return []
        return np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)

    for run in _runs(~near):                    # far side: faint, behind
        ax.plot(xs[run], ys[run], "-", color=color, lw=lw * 0.8, alpha=0.30,
                zorder=z, solid_capstyle="round")
    for run in _runs(near):                     # near side: solid, in front
        ax.plot(xs[run], ys[run], "-", color=color, lw=lw, alpha=1.0,
                zorder=z + 1, solid_capstyle="round")


def flux_region(ax, center, r, direction="out", n_marks=7, color=ACCENT,
                label=None, normal=(0.0, 1.0), z=2):
    """A region of magnetic flux perpendicular to the page: a shaded disk of
    ⊙ (out of the page) or ⊗ (into the page) symbols.

    The end-on view of a solenoid or flux tube — the standard textbook way to
    draw a field normal to the drawing plane. The disk outline registers as
    real geometry for the collision gate; the ⊙/⊗ glyphs inside are part of
    the prop (exempt, like a card's rank). A general primitive for any
    into/out-of-page field patch (Aharonov–Bohm flux tubes, induction loops,
    Hall-bar fields).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    center : tuple of float
        Centre of the flux region.
    r : float
        Radius of the shaded disk.
    direction : {"out", "in"}, optional
        Field direction relative to the page: ``"out"`` draws ⊙ dots,
        ``"in"`` draws ⊗ crosses (default ``"out"``).
    n_marks : int, optional
        Approximate number of symbols scattered over the disk (default ``7``;
        one central symbol is always drawn).
    color : str, optional
        Outline and symbol colour (default the series accent).
    label : str, optional
        Collision-free label placed outside the disk (e.g. ``r"$\\Phi$"``).
    normal : tuple of float, optional
        Preferred outward direction for the label (default up).
    z : int, optional
        Draw z-order (default ``2``).

    Returns
    -------
    None
        Draws the region onto ``ax`` in place.
    """
    cx, cy = float(center[0]), float(center[1])
    theta = np.linspace(0.0, 2.0 * np.pi, 121)
    ax.fill(cx + r * np.cos(theta), cy + r * np.sin(theta), color=color,
            alpha=0.10, zorder=z - 1, lw=0)
    ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), "-", color=color,
            lw=1.6, zorder=z)
    # symbol sites: centre + a ring at 0.55r (part of the prop -> _nocheck)
    sites = [(cx, cy)]
    m = max(int(n_marks) - 1, 0)
    for k in range(m):
        a = 2.0 * np.pi * k / m + 0.35
        sites.append((cx + 0.55 * r * np.cos(a), cy + 0.55 * r * np.sin(a)))
    glyph = "$\\odot$" if direction == "out" else "$\\otimes$"
    for sx, sy in sites:
        ax.text(sx, sy, glyph, color=color, ha="center", va="center",
                fontsize=11, zorder=z + 1, gid="_nocheck")
    if label is not None:
        # Anchor at the disk edge and ALWAYS use a leader line: the disk
        # interior is full of _nocheck prop glyphs, which the collision gate
        # ignores, so a fallback placement could land "clear" on top of them.
        # The leader keeps the label visibly outside the prop instead.
        edge = (cx + r * normal[0] / np.hypot(*normal),
                cy + r * normal[1] / np.hypot(*normal))
        place_label(ax, label, edge, normal=normal, color=color,
                    clearance=0.55 * r, leader=True)


# ---------------------------------------------------------------------------
# Circuit primitives (IEC 60617 style: rectangle resistor, NOT the ANSI zigzag)
# ---------------------------------------------------------------------------
# Every circuit element draws along a wire segment p0 -> p1: a body centred on the
# segment with straight leads to each endpoint, so elements compose on a clean
# rectangular loop in any orientation (horizontal or vertical). The series-RLC
# notebook (3.11) and any later circuit use these so every schematic in the course
# is the same minimal, language-independent IEC style. See NOTEBOOK_STYLE.md.

def _seg_frame(p0, p1):
    """Unit along/across vectors and centre of a segment (internal helper)."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    L = float(np.hypot(*d))
    u = d / L if L > 0 else np.array([1.0, 0.0])
    n = np.array([-u[1], u[0]])
    return p0, p1, u, n, 0.5 * (p0 + p1), L


def wire(ax, p0, p1, color=INK, lw=2.0, z=2):
    """A straight circuit wire between two nodes.

    The connecting line of a schematic; it registers as a collision obstacle so
    component labels route clear of it.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    p0, p1 : tuple of float
        Endpoints of the wire.
    color : str, optional
        Line colour (default the series ink).
    lw : float, optional
        Line width (default ``2.0``).
    z : int, optional
        Draw z-order (default ``2``).

    Returns
    -------
    None
        Draws the wire onto ``ax`` in place.
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "-", color=color, lw=lw, zorder=z,
            solid_capstyle="round")
    _register_segment(ax, p0, p1)


def junction(ax, p, r=0.04, color=INK, z=4):
    """A filled dot marking a wire junction (IEC connection node).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    p : tuple of float
        Junction location.
    r : float, optional
        Dot radius (default ``0.04``).
    color : str, optional
        Fill colour (default the series ink).
    z : int, optional
        Draw z-order (default ``4``).

    Returns
    -------
    None
        Draws the dot onto ``ax`` in place.
    """
    ax.add_patch(Circle(p, r, facecolor=color, edgecolor="none", zorder=z))


def resistor(ax, p0, p1, label="R", body=0.5, width=0.2, color=INK, lw=2.0, z=3,
             label_side=1.0):
    """An IEC resistor: a plain rectangle on the wire (not the ANSI zigzag).

    The international (IEC 60617) symbol, a clean rectangle with straight leads,
    used course-wide. The label sits clear of the wire via the collision placer.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    p0, p1 : tuple of float
        Endpoints of the wire segment carrying the resistor.
    label : str, optional
        Component label (default ``"R"``); pass ``""`` to omit.
    body : float, optional
        Length of the rectangle body along the wire (default ``0.5``).
    width : float, optional
        Rectangle width across the wire (default ``0.2``).
    color : str, optional
        Colour (default the series ink).
    lw : float, optional
        Line width (default ``2.0``).
    z : int, optional
        Draw z-order (default ``3``).
    label_side : float, optional
        Sign selecting which side of the wire the label sits on (default ``1.0``).

    Returns
    -------
    None
        Draws the resistor onto ``ax`` in place.
    """
    p0, p1, u, n, c, _ = _seg_frame(p0, p1)
    a, b = c - 0.5 * body * u, c + 0.5 * body * u
    rect = [a + 0.5 * width * n, b + 0.5 * width * n,
            b - 0.5 * width * n, a - 0.5 * width * n, a + 0.5 * width * n]
    ax.plot([p[0] for p in rect], [p[1] for p in rect], "-", color=color, lw=lw,
            zorder=z)
    wire(ax, p0, a, color=color, lw=lw, z=z)
    wire(ax, b, p1, color=color, lw=lw, z=z)
    if label:
        place_label(ax, label, c, normal=tuple(label_side * n),
                    clearance=0.5 * width + 0.14, color=color)


def capacitor(ax, p0, p1, label="C", gap=0.13, plate=0.36, color=INK, lw=2.0, z=3,
              label_side=1.0):
    """An IEC capacitor: two short parallel plates across the wire.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    p0, p1 : tuple of float
        Endpoints of the wire segment.
    label : str, optional
        Component label (default ``"C"``).
    gap : float, optional
        Spacing between the two plates along the wire (default ``0.13``).
    plate : float, optional
        Plate length across the wire (default ``0.36``).
    color, lw, z : optional
        Colour, line width, z-order.
    label_side : float, optional
        Side of the wire for the label (default ``1.0``).

    Returns
    -------
    None
        Draws the capacitor onto ``ax`` in place.
    """
    p0, p1, u, n, c, _ = _seg_frame(p0, p1)
    pa, pb = c - 0.5 * gap * u, c + 0.5 * gap * u
    for pp in (pa, pb):
        ends = [pp - 0.5 * plate * n, pp + 0.5 * plate * n]
        ax.plot([ends[0][0], ends[1][0]], [ends[0][1], ends[1][1]], "-",
                color=color, lw=lw, zorder=z, solid_capstyle="round")
    wire(ax, p0, pa, color=color, lw=lw, z=z)
    wire(ax, pb, p1, color=color, lw=lw, z=z)
    if label:
        place_label(ax, label, c, normal=tuple(label_side * n),
                    clearance=0.5 * plate + 0.1, color=color)


def inductor(ax, p0, p1, label="L", body=0.6, n_bumps=4, color=INK, lw=2.0, z=3,
             label_side=1.0):
    """An IEC inductor: a coil of semicircular bumps on the wire.

    The recognizable coil symbol (the international standard also allows a filled
    rectangle; the coil is clearer), echoing the solenoid winding of 3.6.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    p0, p1 : tuple of float
        Endpoints of the wire segment.
    label : str, optional
        Component label (default ``"L"``).
    body : float, optional
        Total length of the coil along the wire (default ``0.6``).
    n_bumps : int, optional
        Number of semicircular loops (default ``4``).
    color, lw, z : optional
        Colour, line width, z-order.
    label_side : float, optional
        Side of the wire for the label (default ``1.0``).

    Returns
    -------
    None
        Draws the inductor onto ``ax`` in place.
    """
    p0, p1, u, n, c, _ = _seg_frame(p0, p1)
    a = c - 0.5 * body * u
    step = body / n_bumps
    rb = 0.5 * step
    th = np.linspace(np.pi, 0.0, 40)
    for k in range(n_bumps):
        ck = a + (k + 0.5) * step * u
        pts = ck[None, :] + rb * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * n)
        ax.plot(pts[:, 0], pts[:, 1], "-", color=color, lw=lw, zorder=z,
                solid_capstyle="round")
    wire(ax, p0, a, color=color, lw=lw, z=z)
    wire(ax, a + body * u, p1, color=color, lw=lw, z=z)
    if label:
        place_label(ax, label, c + rb * n, normal=tuple(label_side * n),
                    clearance=rb + 0.12, color=color)


def ac_source(ax, p0, p1, label="V(t)", radius=0.3, color=INK, lw=2.0, z=3,
              label_side=1.0):
    """An IEC AC source: a circle enclosing a sine symbol, with leads.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    p0, p1 : tuple of float
        Endpoints of the wire segment carrying the source.
    label : str, optional
        Source label (default ``"V(t)"``).
    radius : float, optional
        Circle radius (default ``0.3``).
    color, lw, z : optional
        Colour, line width, z-order.
    label_side : float, optional
        Side of the wire for the label (default ``1.0``).

    Returns
    -------
    None
        Draws the source onto ``ax`` in place.
    """
    p0, p1, u, n, c, _ = _seg_frame(p0, p1)
    ax.add_patch(Circle(tuple(c), radius, fill=False, edgecolor=color, lw=lw,
                        zorder=z))
    s = np.linspace(-0.62 * radius, 0.62 * radius, 60)
    wave = c[None, :] + s[:, None] * u + 0.36 * radius * np.sin(
        s / (0.62 * radius) * np.pi)[:, None] * n
    ax.plot(wave[:, 0], wave[:, 1], "-", color=color, lw=lw * 0.8, zorder=z + 1)
    wire(ax, p0, c - radius * u, color=color, lw=lw, z=z)
    wire(ax, c + radius * u, p1, color=color, lw=lw, z=z)
    if label:
        place_label(ax, label, c, normal=tuple(label_side * n),
                    clearance=radius + 0.14, color=color)


def ellipse(ax, center, width, height, angle=0.0, color=INK, lw=1.8, z=2):
    """A thin ellipse outline (orbit, energy contour, phase-space curve)."""
    ax.add_patch(Ellipse(center, width, height, angle=angle, fill=False,
                         edgecolor=color, lw=lw, zorder=z))


def number_line(ax, x0, x1, y=0.0, ticks=None, labels=None, color=INK, lw=2.0,
                tick_h=0.05, arrow=True, z=2, label_every=1, stagger=False):
    """A horizontal axis line from ``x0`` to ``x1`` at height ``y``.

    Optional ``ticks`` draw vertical tick marks and a matching ``labels`` sequence
    annotates them. For ticks that bunch (a log grid), thin the labels with
    ``label_every=k`` (label every k-th) and/or ``stagger=True`` (alternate
    below/above the axis), so crowded labels never overlap. A general schematic
    primitive for the real line, intervals, and non-uniform grids.
    """
    if arrow:
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                     mutation_scale=14, lw=lw, color=color,
                                     zorder=z, shrinkA=0, shrinkB=0))
    else:
        ax.plot([x0, x1], [y, y], "-", lw=lw, color=color, zorder=z)
    _register_segment(ax, (x0, y), (x1, y))
    if ticks is not None:
        shown = 0
        for i, xt in enumerate(ticks):
            ax.plot([xt, xt], [y - tick_h, y + tick_h], "-", lw=1.3,
                    color=color, zorder=z + 1)
            has = labels is not None and i < len(labels) and labels[i] is not None
            if has and (i % label_every == 0):
                below = not (stagger and shown % 2 == 1)
                yo = (y - tick_h - 0.04) if below else (y + tick_h + 0.04)
                # Tick labels manage their own spacing (label_every/stagger),
                # so they are exempt from the collision gate.
                a = ax.annotate(labels[i], (xt, yo), fontsize=11, color=SOFT,
                                ha="center", va="top" if below else "bottom",
                                zorder=z + 1)
                a.set_gid("_nocheck")
                shown += 1


def spin_grid(ax, spins, *, spacing=1.0, arrow=0.46, up_color=INK,
              down_color=ACCENT, highlight=None, bond_color=ACCENT, z=3):
    """Draw a 2D lattice of Ising-type spins as up/down arrows.

    ``spins`` is a 2D array-like of ``+1`` / ``-1``. Site ``(i, j)`` is drawn at
    ``(j*spacing, -i*spacing)`` so row 0 sits at the top, matching how a spin
    matrix prints. Up spins (``+1``) point up in ``up_color``; down spins
    (``-1``) point down in ``down_color``. Pass ``highlight=(i, j)`` to ring that
    site and draw bonds to its four nearest neighbours — the picture behind a
    single-flip energy change, where flipping site ``i`` costs
    ``ΔE = 2 J σ_i Σ_⟨j⟩ σ_j`` over exactly those bonds. A general primitive for
    lattice-spin and magnetism schematics; call :func:`finish` after.
    """
    s = np.asarray(spins)
    ny, nx = s.shape
    h = arrow / 2.0
    for i in range(ny):
        for j in range(nx):
            x, y = j * spacing, -i * spacing
            up = s[i, j] > 0
            tail = (x, y - h) if up else (x, y + h)
            head = (x, y + h) if up else (x, y - h)
            ax.add_patch(FancyArrowPatch(
                tail, head, arrowstyle="-|>", mutation_scale=12, lw=2.0,
                color=up_color if up else down_color, zorder=z,
                shrinkA=0, shrinkB=0))
    if highlight is not None:
        hi, hj = highlight
        cx, cy = hj * spacing, -hi * spacing
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = hi + di, hj + dj
            if 0 <= ni < ny and 0 <= nj < nx:
                ax.plot([cx, nj * spacing], [cy, -ni * spacing], "-",
                        lw=2.6, color=bond_color, alpha=0.5,
                        zorder=z - 1, solid_capstyle="round")
        ax.add_patch(Circle((cx, cy), arrow * 0.62, facecolor="none",
                            edgecolor=bond_color, lw=1.8, zorder=z + 1))



def field_quiver(ax, X, Y, U, V, *, cmap="cividis", n=22, clip_quantile=97.0,
                 width=0.004, pivot="tail"):
    """Quiver whose arrow length AND colour both encode $|\\mathbf F|$ — a faithful map.

    Unlike ``matplotlib.pyplot.streamplot``, whose line density is a placement
    artefact carrying no physics, every arrow here means something: the field is
    subsampled to about ``n`` arrows per side, and each arrow's length and colour are
    the local magnitude (clipped at the ``clip_quantile`` percentile so a cell beside
    a singularity cannot swamp the rest). The faithful field-map primitive of Volume
    III; state in the caption that length and colour encode field strength.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    X, Y : numpy.ndarray
        2-D coordinate grids, as from ``numpy.meshgrid(xs, ys)`` (default "xy"
        indexing; pass ``.T`` for ``indexing="ij"`` grids).
    U, V : numpy.ndarray
        Field components on the same grid.
    cmap : str, optional
        Colormap for the magnitude (default ``"cividis"``).
    n : int, optional
        Target number of arrows per side after subsampling (default ``22``).
    clip_quantile : float, optional
        Percentile at which the magnitude is clipped, taming singularities
        (default ``97.0``).
    width : float, optional
        Arrow shaft width in axes-fraction units (default ``0.004``).
    pivot : str, optional
        Where each arrow pivots on its grid point (default ``"tail"``).

    Returns
    -------
    matplotlib.quiver.Quiver
        The quiver artist, ready to pass to ``Figure.colorbar``.
    """
    X, Y, U, V = (np.asarray(a, dtype=float) for a in (X, Y, U, V))
    ny, nx = X.shape
    sy, sx = max(1, ny // n), max(1, nx // n)
    Xs, Ys = X[::sy, ::sx], Y[::sy, ::sx]
    Us, Vs = U[::sy, ::sx], V[::sy, ::sx]
    mag = np.hypot(Us, Vs)
    finite = np.isfinite(mag) & (mag > 0)
    ref = float(np.percentile(mag[finite], clip_quantile)) if finite.any() else 1.0
    capped = np.minimum(np.nan_to_num(mag), ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(mag > 0, capped / mag, 0.0)
    Uc, Vc = np.nan_to_num(Us * f), np.nan_to_num(Vs * f)
    cell = float(Xs[0, 1] - Xs[0, 0]) if Xs.shape[1] > 1 else 1.0
    return ax.quiver(Xs, Ys, Uc, Vc, capped, cmap=cmap, angles="xy",
                     scale_units="xy", scale=ref / (0.85 * cell), width=width,
                     pivot=pivot)


def radial_field_lines(ax, center, *, n=16, r0=0.0, r1=1.0, color=ACCENT, lw=1.4,
                       inward=False, heads=True, z=2):
    """Point-charge field lines: ``n`` straight rays at EQUAL ANGULAR spacing.

    Equal angles carry equal flux per line, so the 2-D areal line density falls as
    $1/r$ (and $1/r^2$ in 3-D) on its own, the physically faithful encoding of an
    inverse-square field with none of ``streamplot``'s density artefacts.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    center : tuple of float
        ``(x, y)`` location of the charge.
    n : int, optional
        Number of field-line rays (default ``16``).
    r0, r1 : float, optional
        Inner and outer radii over which each ray is drawn (defaults ``0.0``, ``1.0``).
    color : str, optional
        Line colour (default the series accent).
    lw : float, optional
        Line width (default ``1.4``).
    inward : bool, optional
        If ``True``, arrowheads point inward (a negative charge); default ``False``.
    heads : bool, optional
        Whether to draw arrowheads (default ``True``).
    z : int, optional
        Draw z-order (default ``2``).

    Returns
    -------
    None
        Draws onto ``ax`` in place.
    """
    c = np.asarray(center, dtype=float)
    for k in range(n):
        a = 2.0 * np.pi * k / n
        d = np.array([np.cos(a), np.sin(a)])
        p0, p1 = c + r0 * d, c + r1 * d
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "-", color=color, lw=lw,
                alpha=0.9, zorder=z, solid_capstyle="round")
        if heads:
            rm = r0 + 0.55 * (r1 - r0)
            step = 0.08 * (r1 - r0) * (-1.0 if inward else 1.0)
            base, head = c + rm * d, c + (rm + step) * d
            ax.add_patch(FancyArrowPatch(tuple(base), tuple(head), arrowstyle="-|>",
                                         mutation_scale=11, lw=lw, color=color,
                                         shrinkA=0, shrinkB=0, zorder=z + 1))


# ---------------------------------------------------------------------------
# Spacetime / Minkowski-diagram primitives (Volume IV)
# ---------------------------------------------------------------------------
# A Minkowski diagram plots ct (vertical) against x (horizontal). A massive
# object's worldline is STEEPER than 45 deg (inside the light cone); light is at
# exactly 45 deg; a boost to speed v = beta c tilts the primed axes SYMMETRICALLY
# toward the light line (the ct'-axis has slope 1/beta, the x'-axis slope beta, so
# their slopes multiply to 1), and events slide along the invariant hyperbolae
# -(ct)^2 + x^2 = const. These helpers compose like the rest of draw: call
# spacetime_diagram first for the frame and light cone, overlay worldlines,
# boosted axes, hyperbolae, and event markers, then finish(ax). Reused in 4.4
# (the paradoxes) and the general-relativity capstone.


def spacetime_diagram(ax, lim, *, light_cone=True, labels=("x", "ct"),
                      grid_step=None, cone_color=ACCENT):
    """Set up a Minkowski spacetime diagram: a ruled $ct$-$x$ frame and light cone.

    Draws a faint background lattice spanning $[-\\,\\mathrm{lim},\\,\\mathrm{lim}]$ in
    both $x$ (horizontal) and $ct$ (vertical), with the two axes through the origin a
    touch darker, then the $45^\\circ$ **light cone** $ct=\\pm x$ (the null lines that
    separate timelike from spacelike). Call this FIRST, overlay worldlines, boosted
    axes, hyperbolae, and events with the companion helpers, then :func:`finish`. The
    lattice is background and EXEMPT from the collision gate; the light cone is real
    geometry. Reused in 4.4 and the GR capstone.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    lim : float
        Half-extent of the diagram; the frame spans $[-\\,\\mathrm{lim},\\,\\mathrm{lim}]$
        in both $x$ and $ct$.
    light_cone : bool, optional
        Draw the $ct=\\pm x$ null lines (default ``True``).
    labels : tuple of str, optional
        Axis names at the positive ends (default ``("x", "ct")``); an empty string
        omits one.
    grid_step : float or None, optional
        Spacing of the faint lattice; defaults to ``lim / 4`` when ``None``.
    cone_color : str, optional
        Colour of the light cone (default the series accent).

    Returns
    -------
    None
        Draws the frame and light cone onto ``ax`` in place.
    """
    step = grid_step if grid_step is not None else lim / 4.0
    xs = np.arange(-lim, lim + 0.5 * step, step)
    for v in xs:                                    # faint lattice (exempt from gate)
        for seg in (([v, v], [-lim, lim]), ([-lim, lim], [v, v])):
            ln, = ax.plot(seg[0], seg[1], "-", color="#cdc6b6", lw=0.6, zorder=-5)
            ln.set_gid("_grid")
    for seg in (([0, 0], [-lim, lim]), ([-lim, lim], [0, 0])):   # the two axes
        ln, = ax.plot(seg[0], seg[1], "-", color=SOFT, lw=1.1, alpha=0.55, zorder=-4)
        ln.set_gid("_grid")
    if light_cone:                                  # ct = +x and ct = -x (45 deg)
        ax.plot([-lim, lim], [-lim, lim], "--", color=cone_color, lw=1.5,
                alpha=0.85, zorder=1)
        ax.plot([-lim, lim], [lim, -lim], "--", color=cone_color, lw=1.5,
                alpha=0.85, zorder=1)
    if labels[0]:
        place_label(ax, labels[0], (lim, 0.0), normal=(1.0, 0.0), clearance=0.06,
                    color=SOFT)
    if labels[1]:
        place_label(ax, labels[1], (0.0, lim), normal=(0.0, 1.0), clearance=0.06,
                    color=SOFT)


def worldline(ax, beta, lim, *, through=(0.0, 0.0), color=INK, lw=2.0, ls="-",
              z=3, label=None, normal=(1.0, 0.0)):
    """A worldline at speed $v=\\beta c$ through a given event on a Minkowski diagram.

    In the $ct$-$x$ plane a particle moving at $v=\\beta c$ traces $x=x_0+\\beta(ct-ct_0)$
    through the event ``through`` $=(x_0,ct_0)$: vertical for $\\beta=0$ (at rest),
    $45^\\circ$ for $\\beta=1$ (light), and STEEPER than $45^\\circ$ for any massive
    particle (inside the light cone). The segment spans the diagram's vertical extent.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    beta : float
        Speed as a fraction of $c$ ($|\\beta|\\le1$); the worldline's $dx/d(ct)$ slope.
    lim : float
        Half-extent of the diagram (matching :func:`spacetime_diagram`).
    through : tuple of float, optional
        The event $(x_0, ct_0)$ the worldline passes through (default the origin).
    color, lw, ls : optional
        Line colour, width, and style.
    z : int, optional
        Draw z-order (default ``3``).
    label : str or None, optional
        Optional label routed through the collision-free placer.
    normal : tuple of float, optional
        Preferred label direction (default ``(1, 0)``).

    Returns
    -------
    None
        Draws the worldline onto ``ax`` in place.
    """
    x0, ct0 = through
    cts = np.array([-lim, lim])
    xs = x0 + beta * (cts - ct0)
    ax.plot(xs, cts, ls, color=color, lw=lw, zorder=z, solid_capstyle="round")
    if label is not None:
        anchor = (x0 + beta * (0.6 * lim - ct0), 0.6 * lim)
        place_label(ax, label, anchor, normal=normal, clearance=0.1, color=color,
                    fontsize=9, leader=True)


def boosted_axes(ax, beta, lim, *, color=ACCENT, lw=1.6, labels=("x'", "ct'")):
    """The primed ($S'$) axes of a frame boosted to $v=\\beta c$, on a Minkowski diagram.

    A boost tilts the two axes SYMMETRICALLY toward the $45^\\circ$ light line: the
    $ct'$-axis (the worldline $x'=0$ of the moving origin) has slope $1/\\beta$ in
    $ct$-vs-$x$, and the $x'$-axis (the events $t'=0$ that $S'$ calls simultaneous) has
    slope $\\beta$, so the two slopes multiply to $1$. The closing of these axes toward
    the light line IS the geometric content of time dilation, length contraction, and
    the relativity of simultaneity.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    beta : float
        Boost speed as a fraction of $c$ ($0<|\\beta|<1$).
    lim : float
        Half-extent of the diagram (matching :func:`spacetime_diagram`).
    color, lw : optional
        Colour and width of the primed axes (default the series accent).
    labels : tuple of str, optional
        Names for the $x'$- and $ct'$-axes (default ``("x'", "ct'")``); an empty
        string omits one.

    Returns
    -------
    None
        Draws the primed axes onto ``ax`` in place.
    """
    s = np.array([-lim, lim])
    ax.plot(beta * s, s, "-", color=color, lw=lw, zorder=2)       # ct'-axis: x = beta*ct
    ax.plot(s, beta * s, "-", color=color, lw=lw, zorder=2)       # x'-axis: ct = beta*x
    if labels[1]:
        place_label(ax, labels[1], (beta * 0.85 * lim, 0.85 * lim),
                    normal=(-1.0, 0.3), clearance=0.1, color=color, fontsize=9,
                    leader=True)
    if labels[0]:
        place_label(ax, labels[0], (0.85 * lim, beta * 0.85 * lim),
                    normal=(0.3, -1.0), clearance=0.1, color=color, fontsize=9,
                    leader=True)


def hyperbola(ax, s2, lim, *, color=SOFT, lw=1.3, ls=":", z=1, n=200):
    """An invariant hyperbola $-(ct)^2+x^2=s^2$ on a Minkowski diagram.

    The set of events a Lorentz boost can reach from a fixed starting event: a boost
    SLIDES events along these curves, exactly as an ordinary rotation slides points
    along circles. The sign of ``s2`` sets the orientation: ``s2 > 0`` (spacelike)
    opens left-right, ``s2 < 0`` (timelike) opens up-down. Drawing one through an event
    shows where every other observer sees that same event.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    s2 : float
        The invariant interval $s^2=-(ct)^2+x^2$ labelling the curve. Positive is
        spacelike (opens along $x$), negative is timelike (opens along $ct$).
    lim : float
        Half-extent of the diagram (matching :func:`spacetime_diagram`).
    color, lw, ls : optional
        Curve colour, width, and style (default a faint dotted grey).
    z : int, optional
        Draw z-order (default ``1``).
    n : int, optional
        Samples per branch (default ``200``).

    Returns
    -------
    None
        Draws both branches of the hyperbola onto ``ax`` in place.
    """
    if s2 > 0:                                      # x^2 - (ct)^2 = s2: opens in x
        ct = np.linspace(-lim, lim, n)
        x = np.sqrt(ct**2 + s2)
        for sgn in (1.0, -1.0):
            ax.plot(sgn * x, ct, ls, color=color, lw=lw, zorder=z)
    elif s2 < 0:                                    # (ct)^2 - x^2 = |s2|: opens in ct
        x = np.linspace(-lim, lim, n)
        ct = np.sqrt(x**2 - s2)                     # -s2 = |s2| > 0
        for sgn in (1.0, -1.0):
            ax.plot(x, sgn * ct, ls, color=color, lw=lw, zorder=z)


# ---------------------------------------------------------------------------
# Tensor-network diagrams (Penrose notation) — the Volume VII/VIII family
# ---------------------------------------------------------------------------
# The analogue of the circuit (IEC 60617) and spacetime (Minkowski) families
# above: a small set of primitives so a contraction diagram is a few lines in a
# notebook and looks the same everywhere. Convention follows the tensor-network
# literature — a tensor is a node, each index is a leg, and a leg joining two
# nodes is a summed (contracted) index while a free leg is an open index.

def tensor_node(ax, p, label=None, *, r=0.20, shape="circle", color=INK,
                facecolor=PANEL, lw=1.8, z=4, fontsize=12):
    """Draw one tensor as a node in a tensor-network diagram.

    In Penrose/tensor-network notation a tensor of any order is a single node;
    its order is carried entirely by how many legs leave it. Drawing the node
    without reference to its order is the whole point of the notation, and it is
    why a diagram scales to contractions that would be unreadable in indices.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    p : tuple of float
        Node centre in data coordinates.
    label : str, optional
        Text drawn inside the node (exempt from the collision gate, since it is
        part of the glyph rather than an annotation).
    r : float, default 0.20
        Node radius (half-width for the square shape).
    shape : {"circle", "square"}, default "circle"
        Node glyph. Squares conventionally mark isometric/canonical cores.
    color : str, default the series ink
        Edge colour.
    facecolor : str, default the series panel colour
        Fill colour.
    lw : float, default 1.8
        Edge width.
    z : int, default 4
        Draw order.
    fontsize : int, default 12
        Label size.

    Returns
    -------
    tuple of float
        The node centre, so legs can be attached fluently.
    """
    x, y = float(p[0]), float(p[1])
    if shape == "circle":
        ax.add_patch(Circle((x, y), r, facecolor=facecolor, edgecolor=color,
                            linewidth=lw, zorder=z))
    elif shape == "square":
        ax.add_patch(Rectangle((x - r, y - r), 2 * r, 2 * r, facecolor=facecolor,
                               edgecolor=color, linewidth=lw, zorder=z))
    else:
        raise ValueError("shape must be 'circle' or 'square'")
    if label is not None:
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
                color=color, zorder=z + 1, gid="_nocheck")
    return (x, y)


def tensor_leg(ax, p0, p1, label=None, *, color=SOFT, lw=1.6, z=2,
               normal=None, dim=None, shrink=0.20):
    """Draw one index of a tensor as a leg, optionally labelled with its dimension.

    A leg between two nodes is a contracted index (summed over); a leg with one
    free end is an open index of the result. Labelling a leg with its dimension
    is what turns a diagram into a cost estimate, because the cost of a
    contraction is the product of every dimension involved.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    p0, p1 : tuple of float
        Endpoints in data coordinates.
    label : str, optional
        Index name, placed collision-free beside the leg's midpoint.
    color : str, default the series soft ink
        Line colour.
    lw : float, default 1.6
        Line width.
    z : int, default 2
        Draw order.
    normal : tuple of float, optional
        Preferred outward heading for the label; perpendicular to the leg by
        default.
    dim : int or str, optional
        Dimension of the index, appended to the label as ``label = dim``.
    shrink : float, default 0.20
        Distance trimmed from each end so the leg meets a node's edge rather
        than its centre.

    Returns
    -------
    None
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = p1 - p0
    n = np.hypot(*d)
    u = d / n if n else d
    a, b = p0 + shrink * u, p1 - shrink * u
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=lw, zorder=z,
            solid_capstyle="round")
    _register_segment(ax, a, b)
    if label is not None or dim is not None:
        text = label if label is not None else ""
        if dim is not None:
            text = f"{text} = {dim}" if text else str(dim)
        if normal is None:
            normal = (-u[1], u[0])
        place_label(ax, text, tuple((a + b) / 2), normal=normal, clearance=0.10,
                    color=color)


def tensor_network(ax, nodes, legs, *, node_r=0.20, title=None, finish_ax=True):
    """Draw a whole tensor network from a node list and a leg list.

    The convenience wrapper for the common case: a handful of tensors and the
    indices joining them. Free legs are given by naming ``None`` as one endpoint
    together with a direction, which is how open indices are drawn in the
    literature.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    nodes : dict
        Maps a node name to ``(x, y)`` or to ``((x, y), label)`` or to
        ``((x, y), label, shape)``.
    legs : sequence
        Each item is ``(a, b)`` or ``(a, b, label)`` or ``(a, b, label, dim)``.
        ``a`` and ``b`` are node names; either may instead be an explicit
        ``(x, y)`` point, which is how a free leg is drawn.
    node_r : float, default 0.20
        Node radius.
    title : str, optional
        Axes title.
    finish_ax : bool, default True
        Equalise the aspect, strip the frame, resolve labels, and run the
        collision gate via :func:`finish`.

    Returns
    -------
    dict
        Name to centre, for further composition.
    """
    centres = {}
    for name, spec in nodes.items():
        if isinstance(spec[0], (int, float, np.floating, np.integer)):
            p, label, shape = spec, name, "circle"
        elif len(spec) == 2:
            p, label, shape = spec[0], spec[1], "circle"
        else:
            p, label, shape = spec[0], spec[1], spec[2]
        centres[name] = tensor_node(ax, p, label, r=node_r, shape=shape)

    def resolve(end):
        if isinstance(end, str):
            return centres[end]
        return tuple(float(v) for v in end)

    for leg in legs:
        a, b = resolve(leg[0]), resolve(leg[1])
        label = leg[2] if len(leg) > 2 else None
        dim = leg[3] if len(leg) > 3 else None
        free = not (isinstance(leg[0], str) and isinstance(leg[1], str))
        tensor_leg(ax, a, b, label, dim=dim,
                   shrink=node_r if not free else node_r * 0.9,
                   color=ACCENT if free else SOFT)
    if title:
        ax.set_title(title)
    if finish_ax:
        finish(ax)
    return centres


def hyperplane(ax, normal, point=(0.0, 0.0), *, lim=1.6, color=SOFT, lw=1.4,
               ls="--", z=1, label=None, hatch=False):
    """Draw the hyperplane through a point with a given normal (2-D).

    The mirror of a Householder reflection is a hyperplane, and drawing it makes
    the reflection's geometry — same distance, opposite side, norm preserved —
    visible in a way the algebra ``H = I - 2vv^T/v^Tv`` does not.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    normal : array_like, shape (2,)
        Normal vector; need not be normalised.
    point : array_like, shape (2,), default the origin
        A point on the hyperplane.
    lim : float, default 1.6
        Half-length of the drawn segment.
    color : str, default the series soft ink
        Line colour.
    lw : float, default 1.4
        Line width.
    ls : str, default "--"
        Line style.
    z : int, default 1
        Draw order.
    label : str, optional
        Collision-free label placed at one end.
    hatch : bool, default False
        Draw short ticks on the normal's side, the standard "mirror" mark.

    Returns
    -------
    None
    """
    nvec = np.asarray(normal, dtype=float)
    nvec = nvec / (np.linalg.norm(nvec) or 1.0)
    t = np.array([-nvec[1], nvec[0]])
    p = np.asarray(point, dtype=float)
    a, b = p - lim * t, p + lim * t
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=lw, linestyle=ls,
            zorder=z)
    _register_segment(ax, a, b)
    if hatch:
        for s in np.linspace(-lim * 0.85, lim * 0.85, 9):
            q = p + s * t
            ax.plot([q[0], q[0] + 0.09 * nvec[0]], [q[1], q[1] + 0.09 * nvec[1]],
                    color=color, linewidth=0.9, zorder=z)
    if label:
        place_label(ax, label, tuple(b), normal=tuple(t), clearance=0.10,
                    color=color)
