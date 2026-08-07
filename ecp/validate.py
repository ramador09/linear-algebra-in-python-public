"""
ecp.validate
============

Helpers for the **validation cell** that every exercise must end with.

The course rule: no exercise is "done" until its result is checked against
something independent — an analytic limit, a conservation law, a known
eigenvalue, a symmetry. These helpers make those checks *visible* (a printed
PASS/FAIL line) and *enforceable* (they raise in CI when ``strict=True``),
so the continuous-integration build doubles as a correctness gate rather than
a mere "it executes" gate.

Design goals:
* zero dependencies beyond NumPy;
* readable output in the notebook;
* a non-zero exit / raised AssertionError in CI on failure.
"""
from __future__ import annotations

import json
import os

import numpy as np

# ANSI is ignored by notebook HTML but harmless; we use plain glyphs instead.
_PASS = "\u2713"   # check mark
_FAIL = "\u2717"   # ballot x


def _record_margin(label: str, measured: float, allowed: float) -> None:
    """Append one check's headroom to the margin report, when one is requested.

    The margin is ``allowed / measured``: how many times larger the error could
    have been before the check failed. Repeated CI failures in this course were
    all checks that passed locally with a margin of 2-5x and then met a BLAS
    whose rounding fell the other way \u2014 so `tools/run_notebook.py` sets
    ``ECP_VALIDATE_MARGINS`` to a file path, and every `close`/`conserved`/
    `below` call appends a line here, letting the per-notebook loop surface thin
    margins *before* a push instead of learning about them from a red run.
    Unset (the CI case), this is a no-op.
    """
    path = os.environ.get("ECP_VALIDATE_MARGINS")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"label": label, "measured": float(measured),
                                 "allowed": float(allowed)}) + "\n")
    except (OSError, ValueError):
        pass


def _fmt(x) -> str:
    try:
        return f"{float(x):.6g}"
    except (TypeError, ValueError):
        return str(x)


def check(passed: bool, label: str, detail: str = "", *, strict: bool = True) -> bool:
    """Report a boolean assertion as a visible PASS/FAIL line.

    The primitive behind every validation cell: it both *shows* the result (a ✓/✗
    line that stays on the public page) and *enforces* it (a raise in CI), so the
    build doubles as a correctness gate.

    Parameters
    ----------
    passed : bool
        Whether the check succeeded.
    label : str
        Human-readable description of what is being checked.
    detail : str, optional
        Extra context appended in brackets (e.g. the numbers compared).
    strict : bool, optional
        If ``True`` (default), raise ``AssertionError`` on failure so CI fails.

    Returns
    -------
    bool
        ``passed``, coerced to ``bool``.

    Raises
    ------
    AssertionError
        If ``passed`` is false and ``strict`` is true.
    """
    mark = _PASS if passed else _FAIL
    line = f"{mark}  {label}"
    if detail:
        line += f"   [{detail}]"
    print(line)
    if not passed and strict:
        raise AssertionError(f"validation failed: {label} ({detail})")
    return bool(passed)


def close(
    got,
    expected,
    label: str,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    strict: bool = True,
) -> bool:
    """Check ``got ≈ expected`` elementwise within tolerances.

    The workhorse numerical check: a result matches an independent truth (an
    analytic limit, a closed form, a conserved quantity) to within a relative and
    absolute tolerance, the standard ``numpy.allclose`` criterion.

    Parameters
    ----------
    got : array_like
        The computed value(s).
    expected : array_like
        The reference value(s); broadcast against ``got``.
    label : str
        Description of the check.
    rtol : float, optional
        Relative tolerance (default ``1e-6``).
    atol : float, optional
        Absolute tolerance (default ``1e-9``), the floor for near-zero expected
        values.
    strict : bool, optional
        Raise on failure when ``True`` (default).

    Returns
    -------
    bool
        Whether ``got`` and ``expected`` agree within tolerance.
    """
    ok = bool(np.allclose(got, expected, rtol=rtol, atol=atol))
    g, e = np.asarray(got, dtype=complex), np.asarray(expected, dtype=complex)
    # Worst-element headroom: allowed_i / |err_i| under the allclose criterion
    # |err_i| <= atol + rtol*|expected_i|. This is what the margin report reads.
    err_elem = np.abs(g - e)
    allowed_elem = atol + rtol * np.abs(e)
    worst = float(np.max(err_elem)) if err_elem.size else 0.0
    with np.errstate(divide="ignore"):
        ratios = np.where(err_elem > 0, allowed_elem / np.maximum(err_elem, 1e-300),
                          np.inf)
    i_worst = int(np.argmin(ratios)) if ratios.size else 0
    _record_margin(label, float(err_elem.flat[i_worst]) if ratios.size else 0.0,
                   float(allowed_elem.flat[i_worst]) if ratios.size else 0.0)
    g, e = np.asarray(got), np.asarray(expected)
    if g.size == 1 and e.size == 1:
        detail = f"got {_fmt(g)} vs expected {_fmt(e)} (rtol={rtol:g}, atol={atol:g})"
    else:
        detail = f"max|Δ| = {_fmt(worst)} (rtol={rtol:g}, atol={atol:g})"
    return check(ok, label, detail, strict=strict)


def conserved(
    series,
    label: str,
    *,
    rel_drift: float = 1e-4,
    strict: bool = True,
) -> bool:
    """Check that a conserved quantity barely drifts over a trajectory.

    A conserved quantity (total energy, angular momentum) should hold fixed along
    an integrated trajectory; its relative drift is a sharp, physics-based test of
    an integrator's fidelity, the one used throughout Volumes I–II.

    Parameters
    ----------
    series : array_like
        The time series of the supposedly conserved quantity.
    label : str
        Description of the check.
    rel_drift : float, optional
        Maximum allowed drift relative to the initial value (default ``1e-4``).
    strict : bool, optional
        Raise on failure when ``True`` (default).

    Returns
    -------
    bool
        Whether the maximum relative drift stayed below ``rel_drift``.
    """
    s = np.asarray(series, dtype=float)
    ref = s[0]
    drift = float(np.max(np.abs((s - ref) / ref))) if ref != 0 else float(np.max(np.abs(s)))
    _record_margin(label, drift, rel_drift)
    return check(
        drift < rel_drift,
        label,
        f"max relative drift = {_fmt(drift)} (limit {rel_drift:g})",
        strict=strict,
    )


def below(value, threshold, label: str, detail: str = "", *,
          strict: bool = True) -> bool:
    """Check ``value < threshold``, with the margin recorded for the audit loop.

    The threshold-style boolean check — ``residual < TOL`` — written through
    `check()` is invisible to the margin report, because only the finished bool
    arrives. Routing it through `below` keeps the same ✓/✗ line and raise
    semantics while letting `tools/run_notebook.py` report how close the call
    came, which is what catches an under-margined gate before CI does.

    Parameters
    ----------
    value : float
        The measured quantity (typically a residual).
    threshold : float
        The stated bound it must stay under.
    label : str
        Description of the check.
    detail : str, optional
        Extra context; the measured value and threshold are appended to it.
    strict : bool, optional
        Raise on failure when ``True`` (default).

    Returns
    -------
    bool
        Whether ``value < threshold``.
    """
    value = float(value)
    threshold = float(threshold)
    _record_margin(label, value, threshold)
    tail = f"{_fmt(value)} < {_fmt(threshold)}"
    return check(value < threshold, label,
                 f"{detail} [{tail}]" if detail else tail, strict=strict)


def positive(value, label: str, *, strict: bool = True) -> bool:
    """Check that a scalar is strictly positive.

    Used where the *sign* is the physics, most often a positive largest Lyapunov
    exponent certifying deterministic chaos.

    Parameters
    ----------
    value : float
        The scalar to test.
    label : str
        Description of the check.
    strict : bool, optional
        Raise on failure when ``True`` (default).

    Returns
    -------
    bool
        Whether ``value > 0``.
    """
    return check(float(value) > 0, label, f"value = {_fmt(value)}", strict=strict)
