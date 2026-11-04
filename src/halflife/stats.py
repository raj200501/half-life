"""Survival statistics for constraint half-life (RQ1 to RQ5, gate G7).

The input is opportunity rows from :func:`halflife.analysis.opportunity_rows`: one
mapping per decision point (DP) a session reached, with at least ``session_id``,
``position`` (the tool call index at which the DP subtask starts) and ``violated``,
plus whatever covariates a model names (``compaction_count``, ``constraint_type``,
``placement``, ``mitigation``, ...).

What is here:

- :func:`risk_set` and :func:`fit_hazard`: the discrete-time hazard of a first
  violation, a logistic model per opportunity on the risk set (RQ1, RQ2, RQ4).
- :func:`half_life`: the position at which the predicted hazard reaches 0.5, or
  :class:`NotReached` when it does not within the stated range.
- :func:`cluster_bootstrap` and the ``bootstrap_*`` helpers: percentile intervals
  that resample whole sessions, so the dependence between opportunities in one
  session is kept.
- :func:`kaplan_meier`: time to first violation with Greenwood log-log intervals.
  It is descriptive only, because it depends on how often decision points occur.
- :func:`rate_contrast` and :func:`recall_effect`: contrasts between arms (RQ3,
  RQ5) and the compaction effect on probe recall (RQ2).
- :func:`power_by_simulation`: power for preregistration sizing. Its datasets are
  simulated, and its result is labeled synthetic.
- :func:`wilson_interval` and :func:`normal_quantile`.

Everything is deterministic: every random draw takes an explicit seed and uses
numpy's PCG64 generator, and rows are put in a canonical order before resampling.
Only numpy and the standard library are used.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from halflife.analysis import time_to_first_violation

FloatArray: TypeAlias = npt.NDArray[np.float64]
Row: TypeAlias = Mapping[str, Any]
PositionTransform: TypeAlias = Literal["linear", "log1p"]
Times: TypeAlias = Sequence[float] | npt.NDArray[Any]
Flags: TypeAlias = Sequence[bool | int] | npt.NDArray[Any]
POSITION_TRANSFORMS: tuple[str, ...] = ("linear", "log1p")

DEFAULT_L2 = 1e-6
"""Ridge weight for every logistic fit. It keeps the optimum finite under separation
and moves well-identified estimates by far less than their standard errors."""

SEPARATION_DRIFT = 1.0
"""Separation test threshold in logit units (see :func:`fit_logistic`)."""

DEFAULT_COVARIATES: tuple[str, ...] = ("position", "compaction_count")
DEFAULT_CATEGORICAL: tuple[str, ...] = ("constraint_type",)

KM_NOTE = (
    "Descriptive only: time to first violation depends on how often decision points "
    "occur, so it is not an estimate of the per-opportunity hazard."
)
POWER_NOTE = (
    "Synthetic: power from simulated datasets for preregistration sizing, "
    "not model results or real-world results."
)
NAIVE_NOTE = "Wilson interval that treats opportunities as independent (ignores clustering)."

_MAX_ITER = 200
_TOL = 1e-10
_PERCENTILE_METHOD = "cluster percentile bootstrap (linear interpolation)"

# Acklam's rational approximation to the inverse normal CDF.
_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW = 0.02425


class StatsError(ValueError):
    """Raised for unusable input, or for an estimate that is undefined on the data."""


def _is_real(value: object) -> bool:
    """True for a real number (Python or numpy), False for booleans and everything else."""
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))


def _require_seed(seed: int | None) -> int:
    """The seed as an int; every random draw in this module needs an explicit one."""
    if seed is None:
        raise StatsError("an explicit seed is required (pass seed=<int>)")
    return _count(seed, "seed")


# --------------------------------------------------------------------------- small helpers


@dataclass(frozen=True)
class NotReached:
    """A half-life or quantile that is not reached within ``within`` (tool calls).

    It sorts above every number: an interval bound that is a ``NotReached`` reads
    as ``>within``.
    """

    within: float

    def __post_init__(self) -> None:
        if not (_is_real(self.within) and math.isfinite(self.within)):
            raise StatsError(f"NotReached needs a finite bound, got {self.within!r}")
        if self.within < 0:
            raise StatsError(f"NotReached needs a bound >= 0, got {self.within!r}")

    def label(self) -> str:
        """Display form, for example ``>150``."""
        return f">{_fmt(self.within)}"

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form, the ``not_reached_within`` layout of docs/ARTIFACTS.md."""
        return {"not_reached_within": _json_bound(self.within)}


Value: TypeAlias = float | NotReached
"""A statistic value: a finite number, or a :class:`NotReached` bound."""


def _fmt(value: float, digits: int = 4) -> str:
    """Short display form of a number (integers without a decimal point)."""
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.{digits}g}"


def _fmt_value(value: Value, digits: int = 4) -> str:
    return value.label() if isinstance(value, NotReached) else _fmt(value, digits)


def _json_float(value: float) -> float | None:
    """A float rounded to 12 significant digits (stable JSON), or None when not finite."""
    if not math.isfinite(value):
        return None
    return float(f"{value:.12g}")


def _json_bound(value: float) -> int | float:
    """A tool call bound: an int when integral (the G7 layout wants ``int``)."""
    if float(value).is_integer():
        return int(value)
    return float(f"{value:.12g}")


def _check_level(level: float) -> float:
    if not (_is_real(level) and 0.0 < float(level) < 1.0):
        raise StatsError(f"level must be strictly between 0 and 1, got {level!r}")
    return float(level)


def _count(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise StatsError(f"{what} must be an integer, got {value!r}")
    if value < 0:
        raise StatsError(f"{what} must be >= 0, got {value}")
    return int(value)


def _field(row: Row, name: str, index: int) -> Any:
    """``row[name]``, with a clear error when the field is missing or None."""
    try:
        value = row[name]
    except KeyError:
        raise StatsError(f"row {index} has no field {name!r}") from None
    if value is None:
        raise StatsError(f"row {index} has no value for {name!r}")
    return value


def _number(row: Row, name: str, index: int) -> float:
    value = _field(row, name, index)
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise StatsError(f"row {index}: {name!r} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise StatsError(f"row {index}: {name!r} is not finite ({value!r})")
    if name == "position" and number < 0:
        raise StatsError(f"row {index}: position must be >= 0, got {value!r}")
    return number


def _outcome(row: Row, name: str, index: int) -> int:
    value = _field(row, name, index)
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)
    raise StatsError(f"row {index}: {name!r} must be a boolean, got {value!r}")


def _check_transform(transform: str) -> None:
    if transform not in POSITION_TRANSFORMS:
        raise StatsError(f"position_transform must be one of {list(POSITION_TRANSFORMS)}")


def _position_name(transform: str) -> str:
    """Design column name of the position term."""
    return "position" if transform == "linear" else "log1p(position)"


def _position_term(position: float, transform: str) -> float:
    return position if transform == "linear" else math.log1p(position)


def _sigmoid(eta: FloatArray) -> FloatArray:
    """Logistic function without overflow."""
    return np.asarray(np.exp(-np.logaddexp(0.0, -eta)), dtype=np.float64)


def _scalar_sigmoid(eta: float) -> float:
    if eta >= 0:
        return 1.0 / (1.0 + math.exp(-eta))
    exp_eta = math.exp(eta)
    return exp_eta / (1.0 + exp_eta)


def _safe_exp(value: float) -> float:
    return math.exp(value) if value < 700.0 else math.inf


# --------------------------------------------------------------------------- normal and Wilson


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _tail(q: float) -> float:
    num = ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]
    den = (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
    return num / den


def normal_quantile(p: float) -> float:
    """Inverse of :func:`normal_cdf`.

    Acklam's rational approximation, refined by one Halley step on ``math.erfc``,
    which brings it to about double precision in the central range
    (``normal_quantile(0.975)`` is 1.959963984540054).

    Raises:
        StatsError: unless ``0 < p < 1``.
    """
    if not (_is_real(p) and 0.0 < p < 1.0):
        raise StatsError(f"normal_quantile needs 0 < p < 1, got {p!r}")
    if p < _P_LOW:
        x = _tail(math.sqrt(-2.0 * math.log(p)))
    elif p > 1.0 - _P_LOW:
        x = -_tail(math.sqrt(-2.0 * math.log1p(-p)))
    else:
        q = p - 0.5
        r = q * q
        num = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q
        den = ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
        x = num / den
    if x * x / 2.0 > 700.0:
        # Beyond |x| of about 37 (p below 1e-300) the refinement factor overflows; the
        # rational approximation alone is accurate to about 1e-9 relative there.
        return x
    error = normal_cdf(x) - p
    u = error * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


def _z(level: float) -> float:
    return normal_quantile(0.5 + _check_level(level) / 2.0)


@dataclass(frozen=True)
class Proportion:
    """A binomial proportion ``k / n`` with a Wilson score interval."""

    k: int
    n: int
    value: float
    lo: float
    hi: float
    level: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form (``{value, lo, hi}`` as in docs/ARTIFACTS.md, plus counts)."""
        return {
            "k": self.k,
            "n": self.n,
            "value": _json_float(self.value),
            "lo": _json_float(self.lo),
            "hi": _json_float(self.hi),
            "level": self.level,
            "method": "wilson",
        }


def wilson_interval(k: int, n: int, level: float = 0.95) -> Proportion:
    """Wilson score interval for ``k`` successes in ``n`` trials.

    The formula is the one the ship check uses for detector precision and recall,
    so at ``level=0.95`` the bounds agree with it to rounding.

    Raises:
        StatsError: for ``n = 0``, ``k > n``, or non-integer counts.
    """
    k_i = _count(k, "k")
    n_i = _count(n, "n")
    if n_i == 0:
        raise StatsError("a Wilson interval needs at least one trial (n = 0)")
    if k_i > n_i:
        raise StatsError(f"k ({k_i}) cannot exceed n ({n_i})")
    z = _z(level)
    p = k_i / n_i
    denom = 1.0 + z * z / n_i
    centre = (p + z * z / (2.0 * n_i)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n_i + z * z / (4.0 * n_i * n_i)) / denom
    # At k = 0 (k = n) the lower (upper) bound is exactly 0 (1); rounding would leave ~1e-17.
    lo = 0.0 if k_i == 0 else max(0.0, centre - half)
    hi = 1.0 if k_i == n_i else min(1.0, centre + half)
    return Proportion(k_i, n_i, p, lo, hi, level)


# --------------------------------------------------------------------------- risk set


def risk_set(rows: Iterable[Row], by: Sequence[str] = ("session_id",)) -> list[dict[str, Any]]:
    """The discrete-time hazard risk set for a first violation.

    Per session (or per group of the ``by`` fields), opportunities in position order
    up to and including the first violated one. Ties in position keep their input
    order. Groups come out in sorted key order; rows are copied.

    Raises:
        StatsError: when a row lacks a ``by`` field, ``position``, or ``violated``.
    """
    groups: dict[tuple[str, ...], list[tuple[float, int, Row]]] = {}
    for index, row in enumerate(rows):
        key = tuple(str(_field(row, name, index)) for name in by)
        _outcome(row, "violated", index)
        groups.setdefault(key, []).append((_number(row, "position", index), index, row))
    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        for _, index, row in sorted(groups[key], key=lambda item: (item[0], item[1])):
            out.append(dict(row))
            if _outcome(row, "violated", index):
                break
    return out


# --------------------------------------------------------------------------- logistic fits


@dataclass(frozen=True, eq=False)
class LogisticFit:
    """A ridge-stabilized logistic regression fit.

    Attributes:
        names: Term names, ``intercept`` first when one was added.
        coef: Coefficients (log odds), in ``names`` order.
        se: Standard errors from the inverse of the penalized Hessian.
        cov: That inverse (the model-based covariance).
        robust_se: Cluster-robust (CR1 sandwich) standard errors, when clusters were given.
        converged: Whether Newton's method met its tolerance.
        iterations: Newton steps taken.
        loglik: Unpenalized log-likelihood at the estimate.
        n: Observations.
        n_events: Observations with outcome 1.
        l2: Ridge weight.
        separation: True when the estimate drifts as the ridge shrinks, the sign of
            complete or quasi-complete separation (including all-zero or all-one
            outcomes). Coefficients and standard errors are then not interpretable.
        separated_terms: Terms whose contribution drifts under that test.
        rank: Numerical rank of the design.
        n_clusters: Distinct clusters, when clusters were given.
        warnings: Codes: ``all_zero_outcomes``, ``all_one_outcomes``, ``separation``,
            ``not_converged``, ``rank_deficient``, ``one_cluster``, ``specified``.
    """

    names: tuple[str, ...]
    coef: FloatArray
    se: FloatArray
    cov: FloatArray
    robust_se: FloatArray | None
    converged: bool
    iterations: int
    loglik: float
    n: int
    n_events: int
    l2: float
    separation: bool
    separated_terms: tuple[str, ...]
    rank: int
    n_clusters: int | None
    warnings: tuple[str, ...]

    def index(self, name: str) -> int:
        """Position of a term in ``names``."""
        try:
            return self.names.index(name)
        except ValueError:
            raise StatsError(f"no term {name!r}; terms are {list(self.names)}") from None

    @property
    def coefficients(self) -> dict[str, float]:
        """Coefficients by term name."""
        return {name: float(value) for name, value in zip(self.names, self.coef, strict=True)}

    def coefficient(self, name: str) -> float:
        """One coefficient (log odds)."""
        return float(self.coef[self.index(name)])

    def std_error(self, name: str, robust: bool = False) -> float:
        """Model-based or cluster-robust standard error of one term."""
        if robust:
            if self.robust_se is None:
                raise StatsError("no cluster-robust standard errors (fit without clusters)")
            return float(self.robust_se[self.index(name)])
        return float(self.se[self.index(name)])

    def wald_z(self, name: str, robust: bool = False) -> float:
        """Wald statistic ``coefficient / standard error``."""
        return self.coefficient(name) / self.std_error(name, robust)

    @property
    def usable(self) -> bool:
        """Converged, full rank, and free of separation."""
        return self.converged and not self.separation and self.rank == len(self.names)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        terms = []
        for j, name in enumerate(self.names):
            term: dict[str, Any] = {
                "name": name,
                "coef": _json_float(float(self.coef[j])),
                "se": _json_float(float(self.se[j])),
            }
            if self.robust_se is not None:
                term["robust_se"] = _json_float(float(self.robust_se[j]))
            terms.append(term)
        return {
            "model": "logistic, ridge-stabilized Newton",
            "terms": terms,
            "converged": self.converged,
            "iterations": self.iterations,
            "loglik": _json_float(self.loglik),
            "n": self.n,
            "n_events": self.n_events,
            "l2": self.l2,
            "separation": self.separation,
            "separated_terms": list(self.separated_terms),
            "rank": self.rank,
            "n_clusters": self.n_clusters,
            "warnings": list(self.warnings),
        }


def _objective(x: FloatArray, y: FloatArray, beta: FloatArray, lam: float) -> float:
    eta = x @ beta
    loglik = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
    return loglik - 0.5 * lam * float(beta @ beta)


def _newton(
    x: FloatArray, y: FloatArray, lam: float, start: FloatArray, max_iter: int, tol: float
) -> tuple[FloatArray, bool, int]:
    """Maximize the ridge-penalized log-likelihood; return (beta, converged, steps).

    Newton's method with backtracking. The objective is strictly concave (``lam > 0``),
    so every Newton direction is an ascent direction. When the Newton decrement says
    the iterate is within ``tol`` of the optimum, one last full step is taken.
    """
    beta = start.copy()
    eye = np.eye(x.shape[1])
    objective = _objective(x, y, beta, lam)
    for steps in range(max_iter):
        p = _sigmoid(x @ beta)
        grad = x.T @ (y - p) - lam * beta
        hess = (x * (p * (1.0 - p))[:, None]).T @ x + lam * eye
        direction = np.asarray(np.linalg.solve(hess, grad), dtype=np.float64)
        decrement = float(grad @ direction)
        if decrement <= 2.0 * tol:
            return beta + direction, True, steps + 1
        t = 1.0
        while True:
            candidate = beta + t * direction
            value = _objective(x, y, candidate, lam)
            if value >= objective + 1e-4 * t * decrement:
                break
            t *= 0.5
            if t < 1e-12:
                return beta, decrement < 1e-6, steps
        beta, objective = candidate, value
    return beta, False, max_iter


def fit_logistic(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    l2: float = DEFAULT_L2,
    *,
    names: Sequence[str] | None = None,
    intercept: bool = True,
    clusters: Sequence[object] | None = None,
    check_separation: bool = True,
    max_iter: int = _MAX_ITER,
    tol: float = _TOL,
) -> LogisticFit:
    """Logistic regression by Newton's method with a tiny ridge.

    Every coefficient (the intercept too) carries the penalty ``l2 / 2 * beta**2``,
    so the optimum exists even when the outcomes are all 0, all 1, or separated by
    the covariates; the fit then completes and reports the problem instead of
    diverging. Separation is detected by refitting with the ridge divided by 100: with
    a finite maximum likelihood estimate the linear predictor barely moves, while
    under separation it moves by about ``log(100)`` for the separated observations.
    A move above :data:`SEPARATION_DRIFT` sets ``separation``.

    Args:
        X: Design matrix, observations by covariates (a 1-D array is one covariate).
        y: Outcomes, 0 or 1.
        l2: Ridge weight, positive.
        names: Covariate names (default ``x1``, ``x2``, ...).
        intercept: Prepend an intercept column named ``intercept``.
        clusters: One cluster label per observation for CR1 cluster-robust errors.
        check_separation: Run the separation refit (bootstrap loops may skip it).
        max_iter: Newton step limit.
        tol: Convergence tolerance on half the squared Newton decrement.

    Raises:
        StatsError: for empty, non-finite, or mismatched input, outcomes other than
            0 and 1, a non-positive ridge or tolerance, no Newton steps, or duplicate
            names.
    """
    x = np.asarray(X, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2 or yy.ndim != 1:
        raise StatsError("X must be 1-D or 2-D and y must be 1-D")
    n = int(yy.shape[0])
    if n == 0:
        raise StatsError("fit_logistic needs at least one observation")
    if x.shape[0] != n:
        raise StatsError(f"X has {x.shape[0]} rows but y has {n} outcomes")
    if not bool(np.all(np.isfinite(x))):
        raise StatsError("X has non-finite values")
    if not bool(np.all((yy == 0.0) | (yy == 1.0))):
        raise StatsError("outcomes must be 0 or 1")
    if not (_is_real(l2) and math.isfinite(l2) and l2 > 0):
        raise StatsError(f"l2 must be a positive finite number, got {l2!r}")
    if not (_is_real(tol) and math.isfinite(tol) and tol > 0):
        raise StatsError(f"tol must be a positive finite number, got {tol!r}")
    if _count(max_iter, "max_iter") < 1:
        raise StatsError("max_iter must be at least 1")
    term_names = (
        tuple(names) if names is not None else tuple(f"x{j + 1}" for j in range(x.shape[1]))
    )
    if len(term_names) != x.shape[1]:
        raise StatsError(f"{len(term_names)} names for {x.shape[1]} columns")
    if intercept:
        x = np.column_stack([np.ones(n), x])
        term_names = ("intercept", *term_names)
    if len(set(term_names)) != len(term_names):
        raise StatsError(f"duplicate term names in {list(term_names)}")
    if x.shape[1] == 0:
        raise StatsError("there are no terms to fit")
    labels: npt.NDArray[np.intp] | None = None
    n_clusters: int | None = None
    if clusters is not None:
        if len(clusters) != n:
            raise StatsError(f"{len(clusters)} cluster labels for {n} observations")
        unique, labels = np.unique(np.array([str(c) for c in clusters]), return_inverse=True)
        n_clusters = len(unique)

    beta, converged, iterations = _newton(x, yy, float(l2), np.zeros(x.shape[1]), max_iter, tol)
    warnings: list[str] = []
    n_events = int(yy.sum())
    if n_events == 0:
        warnings.append("all_zero_outcomes")
    if n_events == n:
        warnings.append("all_one_outcomes")

    separation = False
    separated: tuple[str, ...] = ()
    if check_separation:
        shrunk, _, _ = _newton(x, yy, float(l2) / 100.0, beta, max_iter, tol)
        moved = shrunk - beta
        if float(np.max(np.abs(x @ moved))) > SEPARATION_DRIFT:
            separation = True
            contribution = np.abs(moved) * np.max(np.abs(x), axis=0)
            separated = tuple(
                name
                for name, value in zip(term_names, contribution, strict=True)
                if value > SEPARATION_DRIFT / 2.0
            )
            warnings.append("separation")
    if not converged:
        warnings.append("not_converged")
    rank = int(np.linalg.matrix_rank(x))
    if rank < x.shape[1]:
        warnings.append("rank_deficient")

    p = _sigmoid(x @ beta)
    hess = (x * (p * (1.0 - p))[:, None]).T @ x + float(l2) * np.eye(x.shape[1])
    cov = np.asarray(np.linalg.inv(hess), dtype=np.float64)
    se = np.sqrt(np.diag(cov))
    robust: FloatArray | None = None
    if labels is not None and n_clusters is not None:
        if n_clusters < 2:
            warnings.append("one_cluster")
        else:
            scores = np.zeros((n_clusters, x.shape[1]))
            np.add.at(scores, labels, x * (yy - p)[:, None])
            sandwich = cov @ (scores.T @ scores) @ cov * (n_clusters / (n_clusters - 1))
            robust = np.sqrt(np.clip(np.diag(sandwich), 0.0, None))
    loglik = float(np.sum(yy * (x @ beta) - np.logaddexp(0.0, x @ beta)))
    for array in (beta, se, cov, robust):
        if array is not None:
            array.setflags(write=False)
    return LogisticFit(
        names=term_names,
        coef=beta,
        se=se,
        cov=cov,
        robust_se=robust,
        converged=converged,
        iterations=iterations,
        loglik=loglik,
        n=n,
        n_events=n_events,
        l2=float(l2),
        separation=separation,
        separated_terms=separated,
        rank=rank,
        n_clusters=n_clusters,
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------- design matrices


@dataclass(frozen=True)
class _Design:
    matrix: FloatArray
    names: tuple[str, ...]
    levels: dict[str, tuple[str, ...]]
    counts: dict[str, dict[str, int]]
    reference: dict[str, str]


def _check_terms(covariates: Sequence[str], categorical: Sequence[str]) -> None:
    every = [*covariates, *categorical]
    if len(set(every)) != len(every):
        raise StatsError(f"covariates and categorical terms repeat a name: {every}")


def _build_design(
    rows: Sequence[Row],
    covariates: Sequence[str],
    categorical: Sequence[str],
    transform: str,
    levels: Mapping[str, Sequence[str]] | None,
) -> _Design:
    """Design columns (no intercept) for numeric covariates and treatment-coded factors.

    For each factor the reference level is the first declared level present in the
    rows (declared order is ``levels[name]`` or sorted observed values); declared
    levels absent from the rows get no column.
    """
    if not rows:
        raise StatsError("no rows to model")
    _check_transform(transform)
    _check_terms(covariates, categorical)
    columns: list[FloatArray] = []
    names: list[str] = []
    for name in covariates:
        values = np.array([_number(row, name, i) for i, row in enumerate(rows)], dtype=np.float64)
        if name == "position" and transform == "log1p":
            values = np.log1p(values)
        if len(rows) > 1 and float(np.ptp(values)) == 0.0:
            raise StatsError(
                f"covariate {name!r} takes a single value in these rows, so its effect "
                "cannot be estimated; drop it from the model"
            )
        columns.append(values)
        names.append(_position_name(transform) if name == "position" else name)
    all_levels: dict[str, tuple[str, ...]] = {}
    counts: dict[str, dict[str, int]] = {}
    reference: dict[str, str] = {}
    for name in categorical:
        observed = [str(_field(row, name, i)) for i, row in enumerate(rows)]
        if levels is not None and name in levels:
            declared = tuple(str(level) for level in levels[name])
        else:
            declared = tuple(sorted(set(observed)))
        unknown = sorted(set(observed) - set(declared))
        if unknown:
            raise StatsError(f"{name!r} has levels {unknown} outside the declared {list(declared)}")
        tally = dict.fromkeys(declared, 0)
        for value in observed:
            tally[value] += 1
        present = [level for level in declared if tally[level] > 0]
        all_levels[name] = declared
        counts[name] = tally
        reference[name] = present[0]
        for level in present[1:]:
            columns.append(np.array([1.0 if v == level else 0.0 for v in observed]))
            names.append(f"{name}[{level}]")
    matrix = (
        np.column_stack(columns) if columns else np.zeros((len(rows), 0), dtype=np.float64)
    ).astype(np.float64)
    full = np.column_stack([np.ones(len(rows)), matrix])
    rank = int(np.linalg.matrix_rank(full))
    if rank < full.shape[1]:
        raise StatsError(
            f"the design is rank deficient (rank {rank} of {full.shape[1]} columns: "
            f"intercept, {', '.join(names)}); some terms are collinear in these rows"
        )
    return _Design(matrix, tuple(names), all_levels, counts, reference)


# --------------------------------------------------------------------------- hazard models


@dataclass(frozen=True, eq=False)
class HazardFit:
    """A discrete-time hazard model: logistic regression of ``violated`` per opportunity.

    Attributes:
        fit: The underlying :class:`LogisticFit`.
        covariates: Numeric covariates; ``position`` enters through ``position_transform``.
        categorical: Treatment-coded factors.
        levels: Declared levels per factor; the first is the declared reference.
        reference: Level per factor that has no column (the first declared level
            present in the data; it differs from the declared reference only when
            that one is absent).
        position_transform: ``linear`` or ``log1p``.
        level_counts: Rows per level, or None for a model specified by coefficients.
        n_rows: Rows fitted.
        n_events: Violations among them.
        n_clusters: Distinct clusters (sessions by default).
        used_risk_set: Whether rows were cut to the risk set first.
    """

    fit: LogisticFit
    covariates: tuple[str, ...]
    categorical: tuple[str, ...]
    levels: Mapping[str, tuple[str, ...]]
    reference: Mapping[str, str]
    position_transform: PositionTransform
    level_counts: Mapping[str, Mapping[str, int]] | None = None
    n_rows: int = 0
    n_events: int = 0
    n_clusters: int = 0
    used_risk_set: bool = True

    @classmethod
    def from_coefficients(
        cls,
        coefficients: Mapping[str, float],
        covariates: Sequence[str] = DEFAULT_COVARIATES,
        categorical: Sequence[str] = (),
        position_transform: PositionTransform = "linear",
        levels: Mapping[str, Sequence[str]] | None = None,
    ) -> HazardFit:
        """A model specified by its coefficients instead of fitted to data.

        Useful for what-if numbers in a preregistration (for example the half-life
        implied by an assumed base rate and slope) and for exact tests. Names are
        ``intercept``, the position column (``position`` or ``log1p(position)``), the
        other covariates, and ``factor[level]`` for each non-reference level.
        """
        _check_transform(position_transform)
        _check_terms(covariates, categorical)
        declared: dict[str, tuple[str, ...]] = {}
        for name in categorical:
            if levels is None or name not in levels or not levels[name]:
                raise StatsError(f"levels for factor {name!r} are required")
            declared[name] = tuple(str(level) for level in levels[name])
        names = ["intercept"]
        names += [_position_name(position_transform) if c == "position" else c for c in covariates]
        for name in categorical:
            names += [f"{name}[{level}]" for level in declared[name][1:]]
        missing = [name for name in names if name not in coefficients]
        extra = sorted(set(coefficients) - set(names))
        if missing or extra:
            raise StatsError(f"coefficients missing {missing} or unexpected {extra}")
        coef = np.array([float(coefficients[name]) for name in names], dtype=np.float64)
        nan = np.full(len(names), math.nan)
        fit = LogisticFit(
            names=tuple(names),
            coef=coef,
            se=nan,
            cov=np.full((len(names), len(names)), math.nan),
            robust_se=None,
            converged=True,
            iterations=0,
            loglik=math.nan,
            n=0,
            n_events=0,
            l2=0.0,
            separation=False,
            separated_terms=(),
            rank=len(names),
            n_clusters=None,
            warnings=("specified",),
        )
        return cls(
            fit=fit,
            covariates=tuple(covariates),
            categorical=tuple(categorical),
            levels=declared,
            reference={name: declared[name][0] for name in categorical},
            position_transform=position_transform,
            level_counts=None,
            used_risk_set=False,
        )

    @cached_property
    def coefficients(self) -> dict[str, float]:
        """Coefficients (log odds) by term name."""
        return self.fit.coefficients

    def _check_level(self, name: str, level: str) -> None:
        if level not in self.levels[name]:
            raise StatsError(
                f"{name!r} has no level {level!r}; levels are {list(self.levels[name])}"
            )
        if self.level_counts is not None and self.level_counts[name][level] == 0:
            raise StatsError(f"level {level!r} of {name!r} is absent from the fitted rows")

    def linear_predictor(self, position: float, /, **covariates: float | str) -> float:
        """Log odds of a violation at ``position`` with every other covariate given.

        Raises:
            StatsError: for a missing or unknown covariate, or a level the fit cannot
                predict (unknown, or absent from the fitted rows).
        """
        needed = {c for c in self.covariates if c != "position"} | set(self.categorical)
        missing = sorted(needed - set(covariates))
        unknown = sorted(set(covariates) - needed)
        if missing or unknown:
            raise StatsError(f"prediction needs values for {missing}; unknown terms {unknown}")
        if not (_is_real(position) and math.isfinite(position) and position >= 0):
            raise StatsError(f"position must be a finite number >= 0, got {position!r}")
        coef = self.coefficients
        eta = coef["intercept"]
        for name in self.covariates:
            if name == "position":
                term = _position_name(self.position_transform)
                eta += coef[term] * _position_term(float(position), self.position_transform)
            else:
                value = covariates[name]
                if not isinstance(value, (int, float, np.integer, np.floating)) or not (
                    math.isfinite(float(value))
                ):
                    raise StatsError(f"covariate {name!r} needs a finite number, got {value!r}")
                eta += coef[name] * float(value)
        for name in self.categorical:
            level = str(covariates[name])
            self._check_level(name, level)
            if level != self.reference[name]:
                eta += coef[f"{name}[{level}]"]
        return float(eta)

    def predict(self, position: float, /, **covariates: float | str) -> float:
        """Predicted hazard (probability of a violation) at ``position``."""
        return _scalar_sigmoid(self.linear_predictor(position, **covariates))

    def curve(self, positions: Iterable[float], /, **covariates: float | str) -> list[float]:
        """Predicted hazard at each position."""
        return [self.predict(position, **covariates) for position in positions]

    def level_effect(self, name: str, level: str) -> float:
        """Log odds ratio of ``level`` against the declared reference level of ``name``."""
        if name not in self.categorical:
            raise StatsError(f"{name!r} is not a factor of this model")
        declared_ref = self.levels[name][0]
        for value in (declared_ref, level):
            self._check_level(name, value)
        coef = self.coefficients

        def column(value: str) -> float:
            return 0.0 if value == self.reference[name] else coef[f"{name}[{value}]"]

        return column(level) - column(declared_ref)

    @property
    def position_coefficient(self) -> float:
        """Coefficient of the position term (0 when position is not modeled)."""
        if "position" not in self.covariates:
            return 0.0
        return self.coefficients[_position_name(self.position_transform)]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "model": "discrete-time hazard, logistic per opportunity",
            "position_transform": self.position_transform,
            "covariates": list(self.covariates),
            "categorical": list(self.categorical),
            "levels": {name: list(values) for name, values in self.levels.items()},
            "reference_levels": dict(self.reference),
            "level_counts": (
                None
                if self.level_counts is None
                else {name: dict(tally) for name, tally in self.level_counts.items()}
            ),
            "used_risk_set": self.used_risk_set,
            "n_rows": self.n_rows,
            "n_events": self.n_events,
            "n_clusters": self.n_clusters,
            "fit": self.fit.to_dict(),
        }


def fit_hazard(
    rows: Iterable[Row],
    covariates: Sequence[str] = DEFAULT_COVARIATES,
    categorical: Sequence[str] = DEFAULT_CATEGORICAL,
    position_transform: PositionTransform = "linear",
    *,
    levels: Mapping[str, Sequence[str]] | None = None,
    use_risk_set: bool = True,
    l2: float = DEFAULT_L2,
    cluster: str = "session_id",
    check_separation: bool = True,
) -> HazardFit:
    """Fit the discrete-time hazard of a violation per opportunity.

    ``logit P(violated) = intercept + b * g(position) + sum(c_k * x_k) + factor effects``
    with ``g`` the identity or ``log1p``, fitted on the risk set (opportunities up to
    and including each session's first violation) unless ``use_risk_set`` is False.
    Standard errors are model-based and, by ``cluster``, cluster-robust.

    Args:
        rows: Opportunity rows.
        covariates: Numeric covariates; include ``position`` to model decay.
        categorical: Factors, treatment coded.
        position_transform: ``linear`` or ``log1p``.
        levels: Declared levels per factor (first is the reference); default sorted
            observed levels. Bootstrap loops pass the full-data levels.
        use_risk_set: Cut rows to the risk set first.
        l2: Ridge weight.
        cluster: Row field for cluster-robust standard errors.
        check_separation: Run the separation test.

    Raises:
        StatsError: for no rows, missing fields, a covariate with a single value, or
            a rank-deficient design.
    """
    data = risk_set(rows) if use_risk_set else [dict(row) for row in rows]
    if not data:
        raise StatsError("no opportunity rows to fit")
    design = _build_design(data, covariates, categorical, position_transform, levels)
    y = np.array([_outcome(row, "violated", i) for i, row in enumerate(data)], dtype=np.float64)
    clusters = [str(_field(row, cluster, i)) for i, row in enumerate(data)]
    fit = fit_logistic(
        design.matrix,
        y,
        l2,
        names=design.names,
        clusters=clusters,
        check_separation=check_separation,
    )
    return HazardFit(
        fit=fit,
        covariates=tuple(covariates),
        categorical=tuple(categorical),
        levels=design.levels,
        reference=design.reference,
        position_transform=position_transform,
        level_counts=design.counts,
        n_rows=len(data),
        n_events=int(y.sum()),
        n_clusters=len(set(clusters)),
        used_risk_set=use_risk_set,
    )


def half_life(fit: HazardFit, max_position: float, /, **fixed: float | str) -> float | NotReached:
    """The position at which the predicted hazard reaches 0.5.

    Every covariate other than position is held at the value given in ``fixed`` (for
    example ``compaction_count=0, constraint_type="prohibition"``). The model has one
    position term, so the crossing is solved exactly: ``g(p) = -eta(0) / b``.

    Returns:
        ``0.0`` when the hazard is already at least 0.5 at position 0, the crossing
        position (a float, in tool calls) when it lies in ``[0, max_position]``, and
        ``NotReached(max_position)`` when the hazard stays below 0.5 there (including
        a flat or falling hazard).

    Raises:
        StatsError: for a negative ``max_position``, a fit that did not converge, or
            covariates the fit cannot predict for.
    """
    if not (_is_real(max_position) and math.isfinite(max_position) and max_position >= 0):
        raise StatsError(f"max_position must be a finite number >= 0, got {max_position!r}")
    if not fit.fit.converged:
        raise StatsError("the hazard fit did not converge, so its half-life is undefined")
    eta0 = fit.linear_predictor(0.0, **fixed)
    if eta0 >= 0.0:
        return 0.0
    slope = fit.position_coefficient
    if slope <= 0.0:
        return NotReached(max_position)
    crossing = -eta0 / slope
    if crossing > _position_term(float(max_position), fit.position_transform):
        return NotReached(max_position)
    if fit.position_transform == "linear":
        return crossing
    return min(math.expm1(crossing), float(max_position))


# --------------------------------------------------------------------------- bootstrap


@dataclass(frozen=True)
class Interval:
    """A point estimate with a cluster-bootstrap percentile interval.

    ``estimate``, ``lo``, and ``hi`` are numbers or :class:`NotReached` bounds. Draws
    on which the statistic was undefined are counted in ``n_failed`` and left out of
    the percentiles.
    """

    estimate: Value
    lo: Value
    hi: Value
    level: float
    n_boot: int
    n_failed: int = 0
    n_not_reached: int = 0
    seed: int | None = None
    cluster: str = "session_id"
    n_clusters: int = 0
    strata: str | None = None
    method: str = _PERCENTILE_METHOD

    @property
    def n_used(self) -> int:
        """Draws that entered the percentiles."""
        return self.n_boot - self.n_failed

    @property
    def share_not_reached(self) -> float:
        """Share of used draws whose value was a :class:`NotReached` bound."""
        return self.n_not_reached / self.n_used if self.n_used else 0.0

    def transformed(self, function: Callable[[float], float]) -> Interval:
        """The interval under a monotone increasing function (``NotReached`` kept)."""

        def apply(value: Value) -> Value:
            return value if isinstance(value, NotReached) else function(value)

        return dataclasses.replace(
            self, estimate=apply(self.estimate), lo=apply(self.lo), hi=apply(self.hi)
        )

    def label(self, digits: int = 4) -> str:
        """Display form, for example ``37.2 [20.1, >150]``."""
        return (
            f"{_fmt_value(self.estimate, digits)} "
            f"[{_fmt_value(self.lo, digits)}, {_fmt_value(self.hi, digits)}]"
        )

    def to_dict(self, key: str = "estimate") -> dict[str, Any]:
        """JSON-ready form; the point estimate goes under ``key`` (``half_life`` in G7).

        A ``NotReached`` point estimate becomes ``key: null`` plus
        ``not_reached_within: N``; a ``NotReached`` bound becomes ``lo: null`` or
        ``hi: null`` plus ``lo_not_reached_within`` or ``hi_not_reached_within``.
        """
        out: dict[str, Any] = {}
        for name, value, extra in (
            (key, self.estimate, "not_reached_within"),
            ("lo", self.lo, "lo_not_reached_within"),
            ("hi", self.hi, "hi_not_reached_within"),
        ):
            if isinstance(value, NotReached):
                out[name] = None
                out[extra] = _json_bound(value.within)
            else:
                out[name] = _json_float(value)
        out.update(
            {
                "share_not_reached": _json_float(self.share_not_reached),
                "level": self.level,
                "n_boot": self.n_boot,
                "n_failed": self.n_failed,
                "seed": self.seed,
                "cluster": self.cluster,
                "n_clusters": self.n_clusters,
                "strata": self.strata,
                "method": self.method,
            }
        )
        return out


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default) of sorted values; +inf allowed."""
    h = q * (len(values) - 1)
    low = math.floor(h)
    high = min(low + 1, len(values) - 1)
    fraction = h - low
    a, b = values[low], values[high]
    if fraction == 0.0 or a == b:
        return a
    if math.isinf(b):
        return math.inf
    return a + fraction * (b - a)


def _as_values(result: object) -> dict[str, float | NotReached]:
    """Validate a statistic's return value into a name-to-value mapping."""
    items = result.items() if isinstance(result, Mapping) else [("value", result)]
    out: dict[str, float | NotReached] = {}
    for name, value in items:
        if isinstance(value, NotReached):
            out[str(name)] = value
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise StatsError(f"statistic {name!r} returned {value!r}, not a number or NotReached")
        number = float(value)
        if math.isinf(number):
            raise StatsError(
                f"statistic {name!r} returned {number}; return NotReached for a value beyond "
                "the range, or nan when it is undefined on a resample"
            )
        out[str(name)] = number
    if not out:
        raise StatsError("statistic returned no values")
    return out


def _resample(
    rows: Iterable[Row],
    statistic: Callable[[list[dict[str, Any]]], object],
    n_boot: int,
    seed: int | None,
    cluster: str,
    level: float,
    strata: str | None,
) -> dict[str, Interval]:
    data = [dict(row) for row in rows]
    if not data:
        raise StatsError("no rows to resample")
    level = _check_level(level)
    n_boot = _count(n_boot, "n_boot")
    if n_boot < 1:
        raise StatsError("n_boot must be at least 1")
    seed = _require_seed(seed)
    clusters: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(data):
        clusters.setdefault(str(_field(row, cluster, index)), []).append(row)
    keys = sorted(clusters)
    if strata is None:
        groups = [list(range(len(keys)))]
    else:
        stratum_of: dict[str, str] = {}
        for key in keys:
            seen = {str(_field(row, strata, i)) for i, row in enumerate(clusters[key])}
            if len(seen) != 1:
                raise StatsError(f"cluster {key!r} spans several {strata!r} values")
            stratum_of[key] = seen.pop()
        groups = [
            [i for i, key in enumerate(keys) if stratum_of[key] == stratum]
            for stratum in sorted(set(stratum_of.values()))
        ]

    point = _as_values(statistic([dict(row) for row in data]))
    for name, value in point.items():
        if not isinstance(value, NotReached) and math.isnan(value):
            raise StatsError(f"statistic {name!r} is undefined on the full data")

    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in point}
    bounds: dict[str, list[float]] = {name: [] for name in point}
    failed = dict.fromkeys(point, 0)
    relabel_session = cluster != "session_id"
    for _ in range(n_boot):
        picks: list[int] = []
        for group in groups:
            picks.extend(group[j] for j in rng.integers(0, len(group), size=len(group)))
        sample: list[dict[str, Any]] = []
        for slot, pick in enumerate(picks):
            key = keys[pick]
            tag = f"#{slot}"
            for row in clusters[key]:
                copy = dict(row)
                copy[cluster] = key + tag
                if relabel_session and "session_id" in copy:
                    copy["session_id"] = f"{copy['session_id']}{tag}"
                sample.append(copy)
        try:
            result = statistic(sample)
        except StatsError:
            for name in failed:
                failed[name] += 1
            continue
        values = _as_values(result)  # a malformed return value is a bug, not a failed draw
        if set(values) != set(point):
            raise StatsError("the statistic returned different names on a resample")
        for name, value in values.items():
            if isinstance(value, NotReached):
                draws[name].append(math.inf)
                bounds[name].append(value.within)
            elif math.isnan(value):
                failed[name] += 1
            else:
                draws[name].append(value)

    q = (1.0 - level) / 2.0
    out: dict[str, Interval] = {}
    for name, estimate in point.items():
        ordered = sorted(draws[name])
        if not ordered:
            raise StatsError(f"the statistic {name!r} was undefined on every resample")
        within = min(bounds[name]) if bounds[name] else 0.0

        def bound(value: float, within: float = within) -> Value:
            return NotReached(within) if math.isinf(value) else value

        out[name] = Interval(
            estimate=estimate,
            lo=bound(_percentile(ordered, q)),
            hi=bound(_percentile(ordered, 1.0 - q)),
            level=level,
            n_boot=n_boot,
            n_failed=failed[name],
            n_not_reached=len(bounds[name]),
            seed=seed,
            cluster=cluster,
            n_clusters=len(keys),
            strata=strata,
        )
    return out


def cluster_bootstrap(
    rows: Iterable[Row],
    statistic: Callable[[list[dict[str, Any]]], Value],
    n_boot: int = 2000,
    seed: int | None = None,
    cluster: str = "session_id",
    level: float = 0.95,
    *,
    strata: str | None = None,
) -> Interval:
    """Percentile interval for a statistic, resampling whole clusters (sessions).

    Each draw picks as many clusters as there are, with replacement (within each
    stratum when ``strata`` names a field that is constant within clusters). The rows
    of the cluster in draw slot ``k`` get ``#k`` appended to their cluster field (and
    to ``session_id`` when that is a different field), so a cluster drawn twice counts
    as two sessions, for example in :func:`risk_set`.

    The statistic receives a list of row copies and returns a number or a
    :class:`NotReached`. ``NotReached`` sorts above every number: a percentile that
    falls among those draws is reported as ``NotReached`` (``>N``), and
    ``share_not_reached`` says how many draws did not reach. A statistic may raise
    :class:`StatsError` or return ``nan`` on a resample where it is undefined; such
    draws are counted in ``n_failed``. Any other return value (a string, ``inf``) is
    an error, on the full data and on every resample alike.

    ``seed`` is required (positionally after ``n_boot`` or by keyword); the draws use
    ``numpy.random.default_rng(seed)``, so a seed fixes the interval exactly. The same
    convention holds for every bootstrap function in this module.

    Raises:
        StatsError: for no rows, no seed, a statistic undefined on the full data, or
            one that is undefined on every resample.
    """
    return _resample(rows, statistic, n_boot, seed, cluster, level, strata)["value"]


def cluster_bootstrap_many(
    rows: Iterable[Row],
    statistic: Callable[[list[dict[str, Any]]], Mapping[str, Value]],
    n_boot: int = 2000,
    seed: int | None = None,
    cluster: str = "session_id",
    level: float = 0.95,
    *,
    strata: str | None = None,
) -> dict[str, Interval]:
    """:func:`cluster_bootstrap` for a statistic that returns several named values.

    All values come from the same draws, so their intervals are consistent with each
    other. A ``nan`` value fails that name only.
    """
    return _resample(rows, statistic, n_boot, seed, cluster, level, strata)


@dataclass(frozen=True)
class HalfLifeResult:
    """A half-life with its cluster-bootstrap interval and the full-data model."""

    interval: Interval
    fit: HazardFit
    max_position: float
    fixed: Mapping[str, float | str]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form: ``half_life``, ``lo``, ``hi`` (or ``not_reached_within``), model."""
        out = self.interval.to_dict("half_life")
        out["max_position"] = _json_bound(self.max_position)
        out["fixed"] = dict(self.fixed)
        out["model"] = self.fit.to_dict()
        return out


def _hazard_statistic_fit(
    sample: Sequence[Row],
    covariates: Sequence[str],
    categorical: Sequence[str],
    position_transform: PositionTransform,
    levels: Mapping[str, Sequence[str]],
    l2: float,
    cluster: str,
) -> HazardFit:
    return fit_hazard(
        sample,
        covariates,
        categorical,
        position_transform,
        levels=levels,
        use_risk_set=False,
        l2=l2,
        cluster=cluster,
        check_separation=False,
    )


def bootstrap_half_life(
    rows: Iterable[Row],
    max_position: float,
    fixed: Mapping[str, float | str],
    n_boot: int = 2000,
    seed: int | None = None,
    *,
    level: float = 0.95,
    cluster: str = "session_id",
    covariates: Sequence[str] = DEFAULT_COVARIATES,
    categorical: Sequence[str] = DEFAULT_CATEGORICAL,
    position_transform: PositionTransform = "linear",
    l2: float = DEFAULT_L2,
) -> HalfLifeResult:
    """Half-life from :func:`fit_hazard` with a cluster-bootstrap interval.

    The risk set is formed once (resampling whole sessions keeps it valid), the
    factor levels are fixed from the full data, and each draw refits the model and
    solves :func:`half_life`. Draws where the fixed level is absent or the design
    is rank deficient count as failed.
    """
    data = risk_set(rows)
    # risk_set is idempotent, so the full fit can cut ``data`` again and say so.
    full = fit_hazard(
        data,
        covariates,
        categorical,
        position_transform,
        use_risk_set=True,
        l2=l2,
        cluster=cluster,
    )
    levels = dict(full.levels)

    def statistic(sample: list[dict[str, Any]]) -> Value:
        model = _hazard_statistic_fit(
            sample, covariates, categorical, position_transform, levels, l2, cluster
        )
        return half_life(model, max_position, **fixed)

    interval = cluster_bootstrap(data, statistic, n_boot, seed=seed, cluster=cluster, level=level)
    return HalfLifeResult(interval, full, float(max_position), dict(fixed))


@dataclass(frozen=True)
class HazardCurve:
    """Predicted hazard at positions, with cluster-bootstrap intervals."""

    positions: tuple[float, ...]
    intervals: tuple[Interval, ...]
    fit: HazardFit
    fixed: Mapping[str, float | str]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "fixed": dict(self.fixed),
            "points": [
                {"position": _json_float(p), **interval.to_dict("hazard")}
                for p, interval in zip(self.positions, self.intervals, strict=True)
            ],
            "model": self.fit.to_dict(),
        }


def bootstrap_hazard_curve(
    rows: Iterable[Row],
    positions: Sequence[float],
    fixed: Mapping[str, float | str],
    n_boot: int = 2000,
    seed: int | None = None,
    *,
    level: float = 0.95,
    cluster: str = "session_id",
    covariates: Sequence[str] = DEFAULT_COVARIATES,
    categorical: Sequence[str] = DEFAULT_CATEGORICAL,
    position_transform: PositionTransform = "linear",
    l2: float = DEFAULT_L2,
) -> HazardCurve:
    """Predicted hazard at each position with pointwise cluster-bootstrap intervals."""
    if not positions:
        raise StatsError("no positions to predict at")
    data = risk_set(rows)
    # risk_set is idempotent, so the full fit can cut ``data`` again and say so.
    full = fit_hazard(
        data,
        covariates,
        categorical,
        position_transform,
        use_risk_set=True,
        l2=l2,
        cluster=cluster,
    )
    full.curve(positions, **fixed)  # validates positions and covariates before resampling
    levels = dict(full.levels)

    def statistic(sample: list[dict[str, Any]]) -> dict[str, Value]:
        model = _hazard_statistic_fit(
            sample, covariates, categorical, position_transform, levels, l2, cluster
        )
        return {str(i): model.predict(p, **fixed) for i, p in enumerate(positions)}

    found = cluster_bootstrap_many(data, statistic, n_boot, seed=seed, cluster=cluster, level=level)
    return HazardCurve(
        tuple(float(p) for p in positions),
        tuple(found[str(i)] for i in range(len(positions))),
        full,
        dict(fixed),
    )


def bootstrap_coefficients(
    rows: Iterable[Row],
    n_boot: int = 2000,
    seed: int | None = None,
    *,
    level: float = 0.95,
    cluster: str = "session_id",
    covariates: Sequence[str] = DEFAULT_COVARIATES,
    categorical: Sequence[str] = DEFAULT_CATEGORICAL,
    position_transform: PositionTransform = "linear",
    l2: float = DEFAULT_L2,
) -> dict[str, Interval]:
    """Hazard coefficients (log odds) with cluster-bootstrap intervals.

    Factor terms are effects against the declared reference level, and
    ``intercept`` is the log odds at position 0 with numeric covariates at 0 and
    factors at their references. A term whose level or reference is absent from a
    resample fails for that draw.
    """
    data = risk_set(rows)
    # risk_set is idempotent, so the full fit can cut ``data`` again and say so.
    full = fit_hazard(
        data,
        covariates,
        categorical,
        position_transform,
        use_risk_set=True,
        l2=l2,
        cluster=cluster,
    )
    levels = dict(full.levels)
    factor_terms = [
        (name, level)
        for name in full.categorical
        for level in full.levels[name][1:]
        if full.level_counts is not None and full.level_counts[name][level] > 0
    ]

    def summarize(model: HazardFit) -> dict[str, Value]:
        out: dict[str, Value] = {}
        references_present = all(
            model.reference[name] == model.levels[name][0] for name in model.categorical
        )
        out["intercept"] = model.coefficients["intercept"] if references_present else math.nan
        for name in model.covariates:
            term = _position_name(model.position_transform) if name == "position" else name
            out[term] = model.coefficients[term]
        for name, level in factor_terms:
            try:
                out[f"{name}[{level}]"] = model.level_effect(name, level)
            except StatsError:
                out[f"{name}[{level}]"] = math.nan
        return out

    def statistic(sample: list[dict[str, Any]]) -> dict[str, Value]:
        return summarize(
            _hazard_statistic_fit(
                sample, covariates, categorical, position_transform, levels, l2, cluster
            )
        )

    return cluster_bootstrap_many(data, statistic, n_boot, seed=seed, cluster=cluster, level=level)


# --------------------------------------------------------------------------- Kaplan-Meier


@dataclass(frozen=True)
class KaplanMeier:
    """A Kaplan-Meier curve of time to first violation (descriptive only).

    One entry per distinct time (event or censoring); ``survival`` is the value
    just after that time. Bounds are Greenwood log-log intervals; they are 1 before
    the first event and None where the curve is 0.
    """

    times: tuple[float, ...]
    n_risk: tuple[int, ...]
    n_event: tuple[int, ...]
    n_censor: tuple[int, ...]
    survival: tuple[float, ...]
    lower: tuple[float | None, ...]
    upper: tuple[float | None, ...]
    level: float
    n: int
    n_events: int
    median: float | NotReached
    descriptive: bool = True
    note: str = KM_NOTE

    def survival_at(self, time: float) -> float:
        """The curve's value at ``time`` (right-continuous step function)."""
        value = 1.0
        for t, s in zip(self.times, self.survival, strict=True):
            if t > time:
                break
            value = s
        return value

    @property
    def censored_times(self) -> tuple[float, ...]:
        """Times with at least one censored session."""
        return tuple(t for t, c in zip(self.times, self.n_censor, strict=True) if c > 0)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form, labeled ``descriptive: true`` as G7 requires."""
        median: dict[str, Any]
        if isinstance(self.median, NotReached):
            median = {"median": None, "median_not_reached_within": _json_bound(self.median.within)}
        else:
            median = {"median": _json_float(self.median)}
        return {
            "descriptive": self.descriptive,
            "note": self.note,
            "method": "Kaplan-Meier, Greenwood variance, log-log interval",
            "level": self.level,
            "n": self.n,
            "n_events": self.n_events,
            **median,
            "steps": [
                {
                    "time": _json_float(t),
                    "n_risk": r,
                    "n_event": e,
                    "n_censor": c,
                    "survival": _json_float(s),
                    "lower": None if lo is None else _json_float(lo),
                    "upper": None if hi is None else _json_float(hi),
                }
                for t, r, e, c, s, lo, hi in zip(
                    self.times,
                    self.n_risk,
                    self.n_event,
                    self.n_censor,
                    self.survival,
                    self.lower,
                    self.upper,
                    strict=True,
                )
            ],
        }


def _loglog_bounds(
    survival: float, greenwood: float, z: float
) -> tuple[float | None, float | None]:
    if survival >= 1.0:
        return 1.0, 1.0
    if survival <= 0.0 or not math.isfinite(greenwood):
        return None, None
    log_s = math.log(survival)
    spread = z * math.sqrt(greenwood) / abs(log_s)
    return survival ** math.exp(spread), survival ** math.exp(-spread)


def kaplan_meier(times: Times, events: Flags, level: float = 0.95) -> KaplanMeier:
    """Kaplan-Meier estimate with Greenwood log-log intervals.

    Sessions censored at a time are counted at risk at that time (events come
    first). The log-log interval is ``S ** exp(+-z * sqrt(V) / |log S|)`` with
    Greenwood's ``V = sum(d / (n * (n - d)))``. The median is the first time at which
    the curve is at most 0.5, or ``NotReached(last time)``.

    The result is flagged descriptive: under this design the time to a first
    violation depends on how often decision points occur.

    Raises:
        StatsError: for empty or mismatched input, non-finite times, or non-boolean events.
    """
    if len(times) == 0:
        raise StatsError("kaplan_meier needs at least one time")
    if len(times) != len(events):
        raise StatsError(f"{len(times)} times but {len(events)} event flags")
    t_values: list[float] = []
    e_values: list[int] = []
    for i, (time, event) in enumerate(zip(times, events, strict=True)):
        if isinstance(time, bool) or not isinstance(time, (int, float, np.integer, np.floating)):
            raise StatsError(f"times[{i}] must be a number, got {time!r}")
        if not math.isfinite(float(time)):
            raise StatsError(f"times[{i}] is not finite ({time!r})")
        if not isinstance(event, (bool, np.bool_)) and event not in (0, 1):
            raise StatsError(f"events[{i}] must be a boolean, got {event!r}")
        t_values.append(float(time))
        e_values.append(int(bool(event)))
    z = _z(level)
    event_counts = Counter(t for t, e in zip(t_values, e_values, strict=True) if e)
    censor_counts = Counter(t for t, e in zip(t_values, e_values, strict=True) if not e)
    remaining = len(t_values)
    survival = 1.0
    greenwood = 0.0
    out_t: list[float] = []
    out_r: list[int] = []
    out_e: list[int] = []
    out_c: list[int] = []
    out_s: list[float] = []
    out_lo: list[float | None] = []
    out_hi: list[float | None] = []
    median: float | NotReached | None = None
    for time in sorted(set(t_values)):
        d = event_counts[time]
        c = censor_counts[time]
        if d:
            survival *= 1.0 - d / remaining
            greenwood = greenwood + d / (remaining * (remaining - d)) if remaining > d else math.inf
        lo, hi = _loglog_bounds(survival, greenwood, z)
        out_t.append(time)
        out_r.append(remaining)
        out_e.append(d)
        out_c.append(c)
        out_s.append(survival)
        out_lo.append(lo)
        out_hi.append(hi)
        if median is None and survival <= 0.5:
            median = time
        remaining -= d + c
    return KaplanMeier(
        times=tuple(out_t),
        n_risk=tuple(out_r),
        n_event=tuple(out_e),
        n_censor=tuple(out_c),
        survival=tuple(out_s),
        lower=tuple(out_lo),
        upper=tuple(out_hi),
        level=_check_level(level),
        n=len(t_values),
        n_events=sum(e_values),
        median=median if median is not None else NotReached(max(t_values)),
    )


def kaplan_meier_rows(rows: Iterable[Row], level: float = 0.95) -> KaplanMeier:
    """:func:`kaplan_meier` of time to first violation per session from opportunity rows.

    Times come from :func:`halflife.analysis.time_to_first_violation`: the position of
    the first violated opportunity, or of the last one reached (censored).
    """
    triples = time_to_first_violation(list(rows))
    if not triples:
        raise StatsError("no opportunity rows")
    return kaplan_meier([t for _, t, _ in triples], [e for _, _, e in triples], level)


# --------------------------------------------------------------------------- contrasts


@dataclass(frozen=True)
class ArmSummary:
    """Per-opportunity violation rate in one arm."""

    arm: str
    n_sessions: int
    n_opportunities: int
    n_violations: int
    rate: Interval
    naive: Proportion

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "arm": self.arm,
            "n_sessions": self.n_sessions,
            "n_opportunities": self.n_opportunities,
            "n_violations": self.n_violations,
            "rate": self.rate.to_dict("rate"),
            "naive_wilson": {**self.naive.to_dict(), "note": NAIVE_NOTE},
        }


@dataclass(frozen=True)
class RateContrast:
    """Arm ``b`` against reference arm ``a``: rate difference and odds ratio."""

    arm_key: str
    a: ArmSummary
    b: ArmSummary
    difference: Interval
    log_odds_ratio: Interval
    odds_ratio: Interval
    fit: LogisticFit
    term: str
    stratified: bool
    used_risk_set: bool
    covariates: tuple[str, ...]
    categorical: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "arm_key": self.arm_key,
            "reference_arm": self.a.to_dict(),
            "arm": self.b.to_dict(),
            "difference": self.difference.to_dict("difference"),
            "log_odds_ratio": self.log_odds_ratio.to_dict("log_odds_ratio"),
            "odds_ratio": self.odds_ratio.to_dict("odds_ratio"),
            "odds_ratio_term": self.term,
            "separation": self.fit.separation,
            "stratified_by_arm": self.stratified,
            "used_risk_set": self.used_risk_set,
            "covariates": list(self.covariates),
            "categorical": list(self.categorical),
            "fit": self.fit.to_dict(),
        }


def rate_contrast(
    rows: Iterable[Row],
    arm_key: str,
    arm_a: object,
    arm_b: object,
    n_boot: int = 2000,
    seed: int | None = None,
    *,
    level: float = 0.95,
    cluster: str = "session_id",
    covariates: Sequence[str] = (),
    categorical: Sequence[str] = (),
    position_transform: PositionTransform = "linear",
    use_risk_set: bool = False,
    l2: float = DEFAULT_L2,
) -> RateContrast:
    """Compare per-opportunity violation rates between two arms.

    Rows whose ``arm_key`` equals ``arm_a`` (the reference) or ``arm_b`` are used;
    other arms are ignored. Reported: each arm's rate, the difference ``b - a``, and
    the odds ratio from a logistic fit of ``violated`` on an indicator of arm ``b``
    (plus optional covariates), all with cluster-bootstrap intervals from the same
    draws. When every cluster sits in one arm, clusters are resampled within arms so
    arm sizes stay fixed. The odds-ratio interval is the exponential of the log
    odds-ratio interval. By default every opportunity counts (``use_risk_set``
    False), so the rate is the share of decision points with a violation.

    Raises:
        StatsError: for equal arms, a row without ``arm_key``, or an arm with no rows.
    """
    if arm_a == arm_b:
        raise StatsError("arm_a and arm_b must differ")
    data = [dict(row) for row in rows]
    for index, row in enumerate(data):
        _field(row, arm_key, index)
    selected = [row for row in data if row[arm_key] == arm_a or row[arm_key] == arm_b]
    if use_risk_set:
        selected = risk_set(selected)
    for arm in (arm_a, arm_b):
        if not any(row[arm_key] == arm for row in selected):
            raise StatsError(f"no opportunities in arm {arm!r} of {arm_key!r}")
    term = f"{arm_key}[{arm_b}]"
    levels: Mapping[str, Sequence[str]] | None = None
    if categorical:
        levels = _build_design(selected, covariates, categorical, position_transform, None).levels

    def model(sample: Sequence[Row], check: bool) -> tuple[float, float, LogisticFit]:
        indicator = np.array([1.0 if row[arm_key] == arm_b else 0.0 for row in sample])
        y = np.array([_outcome(row, "violated", i) for i, row in enumerate(sample)], dtype=float)
        n_b = float(indicator.sum())
        if n_b in (0.0, float(len(sample))):
            raise StatsError("a resample has no opportunities in one arm")
        x = indicator.reshape(-1, 1)
        names: tuple[str, ...] = (term,)
        if covariates or categorical:
            design = _build_design(sample, covariates, categorical, position_transform, levels)
            x = np.column_stack([indicator, design.matrix])
            names = (term, *design.names)
        fit = fit_logistic(x, y, l2, names=names, check_separation=check)
        if "rank_deficient" in fit.warnings:
            raise StatsError("the contrast design is rank deficient")
        rate_a = float(y[indicator == 0.0].mean())
        rate_b = float(y[indicator == 1.0].mean())
        return rate_a, rate_b, fit

    _, _, full_fit = model(selected, True)
    clusters: dict[str, set[bool]] = {}
    for index, row in enumerate(selected):
        clusters.setdefault(str(_field(row, cluster, index)), set()).add(row[arm_key] == arm_b)
    stratified = all(len(arms) == 1 for arms in clusters.values())

    def statistic(sample: list[dict[str, Any]]) -> dict[str, Value]:
        a, b, fit = model(sample, False)
        return {
            "rate_a": a,
            "rate_b": b,
            "difference": b - a,
            "log_odds_ratio": fit.coefficient(term),
        }

    found = cluster_bootstrap_many(
        selected,
        statistic,
        n_boot,
        seed=seed,
        cluster=cluster,
        level=level,
        strata=arm_key if stratified else None,
    )

    def summary(arm: object, rate: Interval) -> ArmSummary:
        arm_rows = [row for row in selected if row[arm_key] == arm]
        k = sum(_outcome(row, "violated", i) for i, row in enumerate(arm_rows))
        sessions = {str(row[cluster]) for row in arm_rows}
        return ArmSummary(
            str(arm),
            len(sessions),
            len(arm_rows),
            k,
            rate,
            wilson_interval(k, len(arm_rows), level),
        )

    log_or = found["log_odds_ratio"]
    return RateContrast(
        arm_key=arm_key,
        a=summary(arm_a, found["rate_a"]),
        b=summary(arm_b, found["rate_b"]),
        difference=found["difference"],
        log_odds_ratio=log_or,
        odds_ratio=log_or.transformed(_safe_exp),
        fit=full_fit,
        term=term,
        stratified=stratified,
        used_risk_set=use_risk_set,
        covariates=tuple(covariates),
        categorical=tuple(categorical),
    )


# --------------------------------------------------------------------------- recall (RQ2)


def recall_rows(rows: Iterable[Row]) -> list[dict[str, Any]]:
    """Rows with a probe-recall outcome (``recalled_before`` not None), copied."""
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if row.get("recalled_before") is None:
            continue
        _outcome(row, "recalled_before", index)
        out.append(dict(row))
    return out


@dataclass(frozen=True)
class RecallCell:
    """Probe recall among opportunities with one compaction count."""

    compaction_count: int
    n_sessions: int
    recall: Proportion

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "compaction_count": self.compaction_count,
            "n_sessions": self.n_sessions,
            "recall": {**self.recall.to_dict(), "note": NAIVE_NOTE},
        }


def recall_by_compaction(
    rows: Iterable[Row], level: float = 0.95, cluster: str = "session_id"
) -> list[RecallCell]:
    """Recall rate before each decision point, by compaction count (descriptive).

    Raises:
        StatsError: when no row has a recall outcome, or a compaction count is not
            a whole number.
    """
    data = recall_rows(rows)
    if not data:
        raise StatsError("no rows with a recall outcome (recalled_before)")
    cells: dict[int, list[dict[str, Any]]] = {}
    for index, row in enumerate(data):
        count = _number(row, "compaction_count", index)
        if not count.is_integer() or count < 0:
            raise StatsError(f"row {index}: compaction_count must be a whole number >= 0")
        cells.setdefault(int(count), []).append(row)
    out = []
    for count in sorted(cells):
        group = cells[count]
        k = sum(_outcome(row, "recalled_before", i) for i, row in enumerate(group))
        sessions = {str(_field(row, cluster, i)) for i, row in enumerate(group)}
        out.append(RecallCell(count, len(sessions), wilson_interval(k, len(group), level)))
    return out


@dataclass(frozen=True)
class RecallEffect:
    """The effect of one more compaction on the odds of recalling a constraint."""

    cells: tuple[RecallCell, ...]
    fit: LogisticFit
    term: str
    log_odds_ratio: Interval
    odds_ratio: Interval
    covariates: tuple[str, ...]
    categorical: tuple[str, ...]
    position_transform: PositionTransform
    n_rows: int
    n_clusters: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "outcome": "recalled_before",
            "term": self.term,
            "log_odds_ratio_per_compaction": self.log_odds_ratio.to_dict("log_odds_ratio"),
            "odds_ratio_per_compaction": self.odds_ratio.to_dict("odds_ratio"),
            "covariates": list(self.covariates),
            "categorical": list(self.categorical),
            "position_transform": self.position_transform,
            "n_rows": self.n_rows,
            "n_clusters": self.n_clusters,
            "by_compaction_count": [cell.to_dict() for cell in self.cells],
            "fit": self.fit.to_dict(),
        }


def recall_effect(
    rows: Iterable[Row],
    n_boot: int = 2000,
    seed: int | None = None,
    *,
    level: float = 0.95,
    cluster: str = "session_id",
    covariates: Sequence[str] = DEFAULT_COVARIATES,
    categorical: Sequence[str] = (),
    position_transform: PositionTransform = "linear",
    l2: float = DEFAULT_L2,
) -> RecallEffect:
    """Logistic model of probe recall on compaction count (RQ2).

    Every opportunity with a probe counts (recall is not absorbing, so there is no
    risk set). With randomized compaction timing and ``position`` among the
    covariates, the ``compaction_count`` coefficient compares opportunities at the
    same position that saw different numbers of compactions. The interval is a
    cluster bootstrap over sessions; the per-count table uses naive Wilson intervals.

    Raises:
        StatsError: when no row has a recall outcome or ``compaction_count`` is not a
            covariate.
    """
    if "compaction_count" not in covariates:
        raise StatsError("recall_effect needs compaction_count among the covariates")
    data = recall_rows(rows)
    if not data:
        raise StatsError("no rows with a recall outcome (recalled_before)")
    design = _build_design(data, covariates, categorical, position_transform, None)
    y = np.array([_outcome(r, "recalled_before", i) for i, r in enumerate(data)], dtype=float)
    labels = [str(_field(row, cluster, i)) for i, row in enumerate(data)]
    full = fit_logistic(design.matrix, y, l2, names=design.names, clusters=labels)
    levels = design.levels

    def statistic(sample: list[dict[str, Any]]) -> Value:
        sample_design = _build_design(sample, covariates, categorical, position_transform, levels)
        outcome = np.array(
            [_outcome(r, "recalled_before", i) for i, r in enumerate(sample)], dtype=float
        )
        fit = fit_logistic(
            sample_design.matrix, outcome, l2, names=sample_design.names, check_separation=False
        )
        return fit.coefficient("compaction_count")

    log_or = cluster_bootstrap(data, statistic, n_boot, seed=seed, cluster=cluster, level=level)
    return RecallEffect(
        cells=tuple(recall_by_compaction(data, level, cluster)),
        fit=full,
        term="compaction_count",
        log_odds_ratio=log_or,
        odds_ratio=log_or.transformed(_safe_exp),
        covariates=tuple(covariates),
        categorical=tuple(categorical),
        position_transform=position_transform,
        n_rows=len(data),
        n_clusters=len(set(labels)),
    )


# --------------------------------------------------------------------------- power


@dataclass(frozen=True)
class PowerResult:
    """Simulated power for preregistration sizing (synthetic, labeled so).

    ``compaction``: rejection rate of a two-sided Wald test (cluster-robust CR1
    errors) of the compaction coefficient in the hazard model on the risk set, with
    ``n_sessions`` sessions. ``contrast``: the same test of the arm coefficient in a
    per-opportunity logistic model with ``n_sessions`` sessions in each of two arms.
    Simulations whose fit was unusable (separation, no convergence, rank deficiency,
    one cluster) count as non-rejections and are tallied in ``*_unusable``.
    Intervals are Wilson intervals for the Monte Carlo error.
    """

    effect: float
    n_sessions: int
    opportunities_per_session: int
    base_rate: float
    n_sim: int
    seed: int
    alpha: float
    max_position: int
    compactions_per_session: int
    position_slope: float
    session_sd: float
    compaction: Proportion
    contrast: Proportion
    compaction_unusable: int
    contrast_unusable: int
    synthetic: bool = True
    note: str = POWER_NOTE

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form, labeled ``synthetic: true``."""
        return {
            "synthetic": self.synthetic,
            "note": self.note,
            "inputs": {
                "effect_log_odds": self.effect,
                "n_sessions": self.n_sessions,
                "opportunities_per_session": self.opportunities_per_session,
                "base_rate": self.base_rate,
                "n_sim": self.n_sim,
                "seed": self.seed,
                "alpha": self.alpha,
                "max_position": self.max_position,
                "compactions_per_session": self.compactions_per_session,
                "position_slope": self.position_slope,
                "session_sd": self.session_sd,
            },
            "test": "two-sided Wald test, cluster-robust (CR1) standard errors",
            "compaction": {
                **self.compaction.to_dict(),
                "unusable": self.compaction_unusable,
                "design": "risk set to first violation; terms position and compaction_count",
            },
            "contrast": {
                **self.contrast.to_dict(),
                "unusable": self.contrast_unusable,
                "design": "n_sessions sessions per arm; every opportunity; term arm",
            },
        }


def _simulate(
    rng: np.random.Generator,
    n_sessions: int,
    opportunities: int,
    max_position: int,
    compactions: int,
    intercept: float,
    position_slope: float,
    compaction_effect: float,
    session_sd: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Positions, compaction counts, outcomes, and session ids (sessions by opportunities)."""
    positions = np.sort(
        np.argsort(rng.random((n_sessions, max_position + 1)), axis=1)[:, :opportunities], axis=1
    ).astype(np.float64)
    events = np.argsort(rng.random((n_sessions, max_position)), axis=1)[:, :compactions] + 1
    counts = (events[:, None, :] <= positions[:, :, None]).sum(axis=2).astype(np.float64)
    frailty = rng.normal(0.0, session_sd, size=(n_sessions, 1)) if session_sd > 0 else 0.0
    eta = intercept + position_slope * positions + compaction_effect * counts + frailty
    outcomes = (rng.random(positions.shape) < _sigmoid(eta)).astype(np.float64)
    sessions = np.repeat(np.arange(n_sessions, dtype=np.float64)[:, None], opportunities, axis=1)
    return positions, counts, outcomes, sessions


def _rejects(fit: LogisticFit, term: str, critical: float) -> bool | None:
    """True or False for a usable fit, None when it is unusable."""
    if not fit.usable or fit.robust_se is None:
        return None
    se = fit.std_error(term, robust=True)
    if not se > 0:
        return None
    return abs(fit.coefficient(term) / se) > critical


def power_by_simulation(
    effect: float,
    n_sessions: int,
    opportunities_per_session: int,
    base_rate: float,
    n_sim: int,
    seed: int,
    alpha: float = 0.05,
    *,
    max_position: int = 150,
    compactions_per_session: int = 2,
    position_slope: float = 0.0,
    session_sd: float = 0.0,
    l2: float = DEFAULT_L2,
) -> PowerResult:
    """Simulated power for the compaction coefficient and for a two-arm contrast.

    Each simulated session has ``opportunities_per_session`` decision points at
    distinct positions drawn uniformly from ``0..max_position`` and
    ``compactions_per_session`` compaction events at distinct indices drawn from
    ``1..max_position``; an opportunity's compaction count is the number of events at
    or before its position. The hazard is
    ``logit h = logit(base_rate) + position_slope * position + effect * count + u``
    with ``u ~ Normal(0, session_sd)`` per session. ``effect`` is a log odds ratio:
    per compaction event in the first design, and for arm ``b`` against arm ``a`` in
    the second. In the second design compaction has no effect on the hazard (the
    term is 0 in both arms), only the arm shifts the log odds, and the fitted model
    has only the arm term. Each design draws from its own child of ``seed``.

    Raises:
        StatsError: for out-of-range inputs, including no compaction events.
    """
    if not (0.0 < base_rate < 1.0):
        raise StatsError("base_rate must be strictly between 0 and 1")
    if not (0.0 < alpha < 1.0):
        raise StatsError("alpha must be strictly between 0 and 1")
    for value, what in ((effect, "effect"), (position_slope, "position_slope")):
        if not math.isfinite(value):
            raise StatsError(f"{what} must be finite")
    if not (math.isfinite(session_sd) and session_sd >= 0):
        raise StatsError("session_sd must be >= 0")
    n_sessions = _count(n_sessions, "n_sessions")
    opportunities = _count(opportunities_per_session, "opportunities_per_session")
    n_sim = _count(n_sim, "n_sim")
    seed = _count(seed, "seed")
    max_position = _count(max_position, "max_position")
    compactions = _count(compactions_per_session, "compactions_per_session")
    if n_sessions < 2 or n_sim < 1 or opportunities < 1:
        raise StatsError("need n_sessions >= 2, n_sim >= 1, and opportunities_per_session >= 1")
    if opportunities > max_position + 1 or compactions > max_position:
        raise StatsError("more decision points or compactions than positions")
    if compactions == 0:
        raise StatsError("compactions_per_session must be >= 1 to estimate a compaction effect")
    critical = normal_quantile(1.0 - alpha / 2.0)
    intercept = math.log(base_rate / (1.0 - base_rate))
    rng_compaction, rng_contrast = (
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(2)
    )
    names = ("position", "compaction_count")
    hits = unusable = 0
    for _ in range(n_sim):
        pos, cnt, y, ses = _simulate(
            rng_compaction,
            n_sessions,
            opportunities,
            max_position,
            compactions,
            intercept,
            position_slope,
            effect,
            session_sd,
        )
        keep = (np.cumsum(y, axis=1) - y) == 0
        x = np.column_stack([pos[keep], cnt[keep]])
        fit = fit_logistic(x, y[keep], l2, names=names, clusters=list(ses[keep]))
        verdict = _rejects(fit, "compaction_count", critical)
        unusable += verdict is None
        hits += bool(verdict)
    arm_hits = arm_unusable = 0
    for _ in range(n_sim):
        outcomes: list[FloatArray] = []
        session_ids: list[FloatArray] = []
        for arm, shift in enumerate((0.0, effect)):
            pos, cnt, y, ses = _simulate(
                rng_contrast,
                n_sessions,
                opportunities,
                max_position,
                compactions,
                intercept + shift,
                position_slope,
                0.0,
                session_sd,
            )
            outcomes.append(y.ravel())
            session_ids.append(ses.ravel() + arm * n_sessions)
        y_all = np.concatenate(outcomes)
        indicator = np.concatenate([np.zeros(outcomes[0].size), np.ones(outcomes[1].size)])
        sessions = np.concatenate(session_ids)
        fit = fit_logistic(indicator, y_all, l2, names=("arm",), clusters=list(sessions))
        verdict = _rejects(fit, "arm", critical)
        arm_unusable += verdict is None
        arm_hits += bool(verdict)
    return PowerResult(
        effect=float(effect),
        n_sessions=n_sessions,
        opportunities_per_session=opportunities,
        base_rate=float(base_rate),
        n_sim=n_sim,
        seed=seed,
        alpha=float(alpha),
        max_position=max_position,
        compactions_per_session=compactions,
        position_slope=float(position_slope),
        session_sd=float(session_sd),
        compaction=wilson_interval(hits, n_sim),
        contrast=wilson_interval(arm_hits, n_sim),
        compaction_unusable=unusable,
        contrast_unusable=arm_unusable,
    )


def describe_interval(name: str, interval: Interval, digits: int = 3) -> str:
    """One plain line for logs: ``name: estimate [lo, hi] (95% cluster bootstrap, ...)``."""
    pct = _fmt(100.0 * interval.level)
    text = f"{name}: {interval.label(digits)} ({pct}% cluster bootstrap, {interval.n_used} draws"
    if interval.n_not_reached:
        text += f", {_fmt(100.0 * interval.share_not_reached, 3)}% not reached"
    return text + ")"


__all__ = [
    "DEFAULT_CATEGORICAL",
    "DEFAULT_COVARIATES",
    "DEFAULT_L2",
    "KM_NOTE",
    "NAIVE_NOTE",
    "POSITION_TRANSFORMS",
    "POWER_NOTE",
    "SEPARATION_DRIFT",
    "ArmSummary",
    "HalfLifeResult",
    "HazardCurve",
    "HazardFit",
    "Interval",
    "KaplanMeier",
    "LogisticFit",
    "NotReached",
    "PowerResult",
    "Proportion",
    "RateContrast",
    "RecallCell",
    "RecallEffect",
    "StatsError",
    "bootstrap_coefficients",
    "bootstrap_half_life",
    "bootstrap_hazard_curve",
    "cluster_bootstrap",
    "cluster_bootstrap_many",
    "describe_interval",
    "fit_hazard",
    "fit_logistic",
    "half_life",
    "kaplan_meier",
    "kaplan_meier_rows",
    "normal_cdf",
    "normal_quantile",
    "power_by_simulation",
    "rate_contrast",
    "recall_by_compaction",
    "recall_effect",
    "recall_rows",
    "risk_set",
    "wilson_interval",
]
