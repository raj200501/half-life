"""Survival statistics for constraint half-life (RQ1 to RQ5, gate G7).

The input is opportunity rows from :func:`halflife.analysis.opportunity_rows`: one
mapping per decision point (DP) a session reached, with at least ``family``,
``session_id``, ``position`` (the tool call index at which the DP subtask starts) and
``violated``, plus whatever covariates a model names (``compaction_count``,
``constraint_type``, ``placement``, ``mitigation``, ...).

Sessions and clusters. Every model family runs the same plans and ``session_id`` is the
plan's id, so a session is the pair :data:`SESSION_KEY` ``(family, session_id)``: risk
sets, times to first violation, and session counts group by that pair (a ``session``
argument names other fields, such as E2's ``(model, session_id)``). Bootstrap draws and
CR1 errors cluster on ``session_id`` by default, which is the plan: within one family
that is the session, and over pooled families it keeps both families' sessions of a plan
together, since the plan is the unit of randomization.

What is here:

- :func:`risk_set` and :func:`fit_hazard`: the discrete-time hazard of a first
  violation, a logistic model per opportunity on the risk set (RQ1, RQ2, RQ4).
- :func:`half_life`: the position at which the predicted hazard reaches 0.5, searched
  from the first observed decision point, or :class:`NotReached` when it does not within
  the stated range.
- :func:`cluster_bootstrap` and the ``bootstrap_*`` helpers: percentile intervals
  that resample whole clusters (sessions, or plans when families are pooled), so the
  dependence between opportunities in one session is kept. No estimate is reported at a
  value held only by the ridge (D-029): a term the full data cannot estimate (no
  violation, or separation) gets the not-estimable form of :class:`Interval`, and a draw
  that cannot estimate it counts as failed (``n_failed``).
- :func:`kaplan_meier`: time to first violation with Greenwood log-log intervals.
  It is descriptive only, because it depends on how often decision points occur.
- :func:`rate_contrast` and :func:`recall_effect`: contrasts between arms (RQ3,
  RQ5) and the compaction effect on probe recall (RQ2).
- :func:`wald_test`: a two-sided Wald test of one term, with cluster-robust (CR1)
  errors, that names the reason when no valid p-value exists instead of returning one.
- :func:`score_test`: a two-sided score test of one term at the null fit, clustered
  through an exchangeable within-cluster correlation, which stays valid under separation
  (for example an arm without violations); with ``strata`` the null model is fit within
  each stratum (pooled families, DECISIONS.md D-050), and rows a separated null model
  puts at fitted value 0 or 1 leave, as the ridge-free limit has them add nothing.
- :func:`hazard_score_test`: that test for one numeric term of the hazard model on the
  risk set (H1 and H2b), with factor levels without violations left out per stratum.
- :func:`power_by_simulation`: power for preregistration sizing, deciding by those score
  tests. Its datasets are simulated, and its result is labeled synthetic.
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
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from halflife.analysis import SESSION_KEY, time_to_first_violation

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


def _session_fields(session: Sequence[str]) -> tuple[str, ...]:
    """The fields that identify a session, checked: at least one, all distinct strings."""
    fields = (session,) if isinstance(session, str) else tuple(session)
    if not fields or not all(isinstance(name, str) and name for name in fields):
        raise StatsError(f"a session key needs at least one field name, got {session!r}")
    if len(set(fields)) != len(fields):
        raise StatsError(f"a session key repeats a field: {list(fields)}")
    return fields


def _session_of(row: Row, fields: Sequence[str], index: int) -> tuple[str, ...]:
    return tuple(str(_field(row, name, index)) for name in fields)


def risk_set(rows: Iterable[Row], by: Sequence[str] = SESSION_KEY) -> list[dict[str, Any]]:
    """The discrete-time hazard risk set for a first violation.

    Per session, opportunities in position order up to and including the first
    violated one. A session is a group of the ``by`` fields, :data:`SESSION_KEY`
    ``(family, session_id)`` by default: the families run the same plans, so their
    sessions of one plan share a ``session_id``, and one family's first violation must
    not end the other family's session. Ties in position keep their input order. Groups
    come out in sorted key order; rows are copied.

    Raises:
        StatsError: when a row lacks a ``by`` field, ``position``, or ``violated``, or
            when ``by`` names no field.
    """
    fields = _session_fields(by)
    groups: dict[tuple[str, ...], list[tuple[float, int, Row]]] = {}
    for index, row in enumerate(rows):
        key = _session_of(row, fields, index)
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


WALD_PROBLEMS: dict[str, str] = {
    "not_converged": "the fit did not converge, so the estimate is not a maximum",
    "rank_deficient": "the design is rank deficient, so the term is not identified",
    "separation": (
        "the outcomes are separated (for example an arm with no violations), so no finite "
        "maximum likelihood estimate exists; the coefficient is held only by the ridge and "
        "a Wald statistic for it is not valid"
    ),
    "one_cluster": "cluster-robust errors need at least two clusters",
    "no_cluster_robust_se": "the fit has no cluster-robust errors (it was fit without clusters)",
    "zero_se": "the standard error is zero or not finite",
}
"""Why :func:`wald_test` reports no p-value, by problem code."""


@dataclass(frozen=True)
class WaldTest:
    """A two-sided Wald test of one coefficient against zero.

    When no valid p-value exists, ``p_value`` (and ``z``) are None and ``problem`` holds
    a code from :data:`WALD_PROBLEMS` with its ``detail``; a missing p-value is never
    reported as a number such as 1.0.
    """

    term: str
    estimate: float
    std_error: float | None
    z: float | None
    p_value: float | None
    robust: bool
    n_clusters: int | None
    problem: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "test": "two-sided Wald, " + ("CR1 cluster-robust" if self.robust else "model-based"),
            "term": self.term,
            "estimate": _json_float(self.estimate),
            "std_error": None if self.std_error is None else _json_float(self.std_error),
            "z": None if self.z is None else _json_float(self.z),
            "p_value": None if self.p_value is None else _json_float(self.p_value),
            "robust": self.robust,
            "n_clusters": self.n_clusters,
            "problem": self.problem,
            "detail": self.detail,
        }


def wald_test(fit: LogisticFit, term: str, *, robust: bool = True) -> WaldTest:
    """Two-sided Wald test of ``term``, with CR1 cluster-robust errors by default.

    ``p = erfc(|z| / sqrt(2))``, which equals ``2 * (1 - Phi(|z|))`` and stays accurate in
    the far tail. No p-value is reported (``p_value`` None, ``problem`` set) when the fit
    did not converge, is rank deficient, or shows separation (D-029: flagged coefficients
    are not interpreted; under separation the CR1 error of the separated term collapses
    and its Wald statistic is spuriously large), or when the needed standard error is
    missing, zero, or not finite.

    Raises:
        StatsError: when ``term`` is not a term of the fit.
    """
    estimate = fit.coefficient(term)
    index = fit.index(term)

    def missing(code: str, se: float | None = None) -> WaldTest:
        return WaldTest(
            term, estimate, se, None, None, robust, fit.n_clusters, code, WALD_PROBLEMS[code]
        )

    if not fit.converged:
        return missing("not_converged")
    if "rank_deficient" in fit.warnings:
        return missing("rank_deficient")
    if fit.separation:
        return missing("separation")
    if robust and fit.robust_se is None:
        return missing("one_cluster" if "one_cluster" in fit.warnings else "no_cluster_robust_se")
    errors = fit.robust_se if robust and fit.robust_se is not None else fit.se
    se = float(errors[index])
    if not (math.isfinite(se) and se > 0.0):
        return missing("zero_se", se if math.isfinite(se) else None)
    z = estimate / se
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return WaldTest(term, estimate, se, z, p, robust, fit.n_clusters)


SCORE_PROBLEMS: dict[str, str] = {
    "no_events": "no outcome is a violation, so there is nothing to test",
    "all_events": "every outcome is a violation, so there is nothing to test",
    "no_stratum_with_variation": (
        "no stratum has both violations and non-violations, so there is nothing to test "
        "within strata"
    ),
    "one_cluster": "a clustered score test needs at least two clusters",
    "rank_deficient": "the term is collinear with the null model, so it cannot be tested",
    "zero_variance": "the score has zero or non-finite variance",
    "null_not_converged": "the null fit did not converge",
    "null_separation": (
        "the null model separates every row (its fitted values are all 0 or 1), so nothing "
        "is left to test"
    ),
}
"""Why :func:`score_test` reports no p-value, by problem code."""

_BOUNDARY_ROUNDS = 10
"""Refits :func:`score_test` makes in one stratum while rows of a separated null model leave."""

SCORE_RHO_MAX = 0.99
"""Upper bound of the within-cluster correlation :func:`score_test` uses (its lower bound
is 0)."""
SCORE_TEST_NAME = (
    "two-sided score test at the null fit, variance with an exchangeable within-cluster "
    "correlation estimated at the null fit"
)


@dataclass(frozen=True)
class ScoreTest:
    """A two-sided score (Rao) test of one term at the null fit, clustered (:func:`score_test`).

    ``rho`` is the within-cluster correlation its variance used. ``p_value`` (and ``z``) are
    None, with a code from :data:`SCORE_PROBLEMS`, when no valid test exists; a missing
    p-value is never reported as a number. ``n_rows`` and ``n_events`` count the rows the
    test used and their violations. A stratified test (``n_strata`` not None) fits its null
    model within each stratum; ``strata_skipped`` names the strata that add nothing (no
    violations, or only violations, left), and ``n_boundary`` counts the rows a separated
    null model puts at fitted value 0 or 1, which add nothing either; neither is counted in
    ``n_rows``.
    """

    term: str
    z: float | None
    p_value: float | None
    n_clusters: int
    problem: str | None = None
    detail: str | None = None
    rho: float | None = None
    n_rows: int | None = None
    n_events: int | None = None
    n_strata: int | None = None
    strata_skipped: tuple[str, ...] = ()
    n_boundary: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "test": SCORE_TEST_NAME,
            "term": self.term,
            "z": None if self.z is None else _json_float(self.z),
            "p_value": None if self.p_value is None else _json_float(self.p_value),
            "rho": None if self.rho is None else _json_float(self.rho),
            "n_clusters": self.n_clusters,
            "n_rows": self.n_rows,
            "n_events": self.n_events,
            "n_strata": self.n_strata,
            "strata_skipped": list(self.strata_skipped),
            "n_boundary_rows": self.n_boundary,
            "problem": self.problem,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class _NullFit:
    """One stratum's null fit inside :func:`score_test`: rows, fitted values, residualized x,
    and how many rows left because the null model put them at fitted value 0 or 1."""

    index: npt.NDArray[np.intp]
    p: FloatArray
    resid_x: FloatArray
    boundary: int


def _stratum_null_fit(
    index: npt.NDArray[np.intp],
    xx: FloatArray,
    yy: FloatArray,
    extra: FloatArray | None,
    l2: float,
) -> _NullFit | str | None:
    """The null fit of one stratum, None when the stratum adds nothing, or a problem code.

    The null design is an intercept and the columns of ``extra`` that vary within the
    stratum (a column constant there is collinear with the intercept and adds nothing).
    When the null model is separated (the separation test of :func:`fit_logistic`: the
    linear predictor drifts by more than :data:`SEPARATION_DRIFT` as the ridge shrinks), the
    rows that drift have maximum likelihood fitted values of 0 or 1, so their residuals and
    weights are 0 and they add nothing to the score or its variance; they leave and the rest
    is refit, which is the limit of the ridge-free test (the rows that stay do not move
    along the separating direction). ``x`` is then residualized by weighted least squares,
    which stays defined when the columns are collinear (the fitted values are unique even
    when the coefficients are not).
    """
    boundary = 0
    for _ in range(_BOUNDARY_ROUNDS):
        ys = yy[index]
        if float(ys.sum()) in (0.0, float(len(index))):
            return None
        columns = [np.ones(len(index))]
        if extra is not None:
            sub = extra[index]
            columns += [sub[:, j] for j in np.flatnonzero(np.ptp(sub, axis=0) > 0.0)]
        z_mat = np.column_stack(columns)
        if z_mat.shape[1] == 1:
            p = np.full(len(index), float(ys.mean()))
            break
        beta, converged, _ = _newton(z_mat, ys, l2, np.zeros(z_mat.shape[1]), _MAX_ITER, _TOL)
        if not converged:
            return "null_not_converged"
        shrunk, _, _ = _newton(z_mat, ys, l2 / 100.0, beta, _MAX_ITER, _TOL)
        moved = np.abs(z_mat @ (shrunk - beta)) > SEPARATION_DRIFT
        if not bool(moved.any()):
            p = _sigmoid(z_mat @ beta)
            break
        boundary += int(moved.sum())
        index = index[~moved]
    else:
        return "null_separation"
    root_w = np.sqrt(p * (1.0 - p))
    xs = xx[index]
    coef = np.linalg.lstsq(z_mat * root_w[:, None], root_w * xs, rcond=None)[0]
    return _NullFit(index, p, np.asarray(xs - z_mat @ coef, dtype=np.float64), boundary)


def score_test(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    clusters: Sequence[object],
    null_design: npt.ArrayLike | None = None,
    *,
    term: str = "x",
    l2: float = DEFAULT_L2,
    strata: Sequence[object] | None = None,
) -> ScoreTest:
    """Two-sided score test of ``x`` in a logistic model of ``y``, clustered by ``clusters``.

    The null model holds an intercept and the columns of ``null_design`` (none by
    default). With ``p`` its fitted probabilities and ``w = p (1 - p)``, the score of the
    tested term is ``U = sum(x~ (y - p))`` with ``x~ = x - Z (Z'WZ)^-1 Z'W x`` (the term
    residualized on the null design, so estimating the null model leaves its first-order
    variance unchanged). Its variance treats the outcomes of one cluster as exchangeably
    correlated under the null:
    ``V = sum over clusters of (1 - rho) * sum(w x~^2) + rho * (sum(sqrt(w) x~))^2``, where
    ``rho`` is the moment estimate of the within-cluster correlation of the Pearson residuals
    ``(y - p) / sqrt(w)`` at the null fit (the sum over clusters of every within-cluster
    pair's product, over the number of such pairs), kept in ``[0, SCORE_RHO_MAX]``;
    ``z = U / sqrt(V)`` and ``p = erfc(|z| / sqrt(2))``. With no within-cluster pair
    ``rho`` is 0 and the test is the Pearson chi-square test of the two-by-two table.

    With ``strata`` (one label per row, DECISIONS.md D-050 item 2) the null model is fit
    within each stratum: its own intercept and the ``null_design`` columns that vary there,
    so ``x~`` and ``p`` are per stratum, while the score sums over strata and the variance
    sums over clusters, which may hold rows of several strata (E1 plans hold both families'
    sessions), with one ``rho`` from every within-cluster pair. A stratum without
    violations, or with only violations, has its null fitted values at 0 or 1 and adds
    nothing to the score, its variance, or ``rho``, so it is left out (``strata_skipped``)
    instead of separating a stratum term. Columns of ``null_design`` that are constant
    within a stratum (for example another stratum's columns) are dropped there. The same
    limit holds for any separated null model, stratified or not: rows it puts at fitted
    value 0 or 1 leave (``n_boundary``, see :func:`_stratum_null_fit`), and a null model
    that puts every row there leaves nothing to test.

    The test is evaluated at the null fit only, so it stays valid where the model with the
    term has no finite estimate: an arm with no violations separates that model and leaves
    no Wald test, yet its score test is defined. Its variance uses the null fit's ``w``
    for every row, not each cluster's realized score: the CR1 (sandwich) variance of the
    score sums squared cluster scores, so when the rarer arm has no violations it sees no
    variability from that arm and the test rejects far too often (in a synthetic null
    calibration on the E1 plans it rejected more than a quarter of the true nulls whose
    rarer type had no violation, and about one in eight of all tests at the 0.05 level in
    the worst setting, while the exchangeable variance stayed near the nominal level).
    Without a null design the null fit is the overall rate (exact).

    Raises:
        StatsError: for empty or mismatched input, outcomes other than 0 and 1, or
            non-finite ``x`` or ``null_design`` values.
    """
    xx = np.asarray(x, dtype=np.float64).ravel()
    yy = np.asarray(y, dtype=np.float64).ravel()
    n = int(yy.shape[0])
    if (
        n == 0
        or xx.shape[0] != n
        or len(clusters) != n
        or (strata is not None and len(strata) != n)
    ):
        raise StatsError("score_test needs x, y, clusters, and strata of the same nonzero length")
    if not bool(np.all((yy == 0.0) | (yy == 1.0))):
        raise StatsError("outcomes must be 0 or 1")
    if not bool(np.all(np.isfinite(xx))):
        raise StatsError("x has non-finite values")
    extra: FloatArray | None = None
    if null_design is not None:
        extra = np.asarray(null_design, dtype=np.float64).reshape(n, -1)
        if not bool(np.all(np.isfinite(extra))):
            raise StatsError("null_design has non-finite values")
    cluster_labels = np.array([str(c) for c in clusters])
    groups = len(np.unique(cluster_labels))
    if strata is None:
        stratum_of = np.zeros(n, dtype=np.intp)
        stratum_keys: list[str] = [""]
    else:
        stratum_keys = sorted({str(s) for s in strata})
        position = {key: i for i, key in enumerate(stratum_keys)}
        stratum_of = np.array([position[str(s)] for s in strata], dtype=np.intp)

    def missing(code: str, used: int = groups, skipped: Sequence[str] = ()) -> ScoreTest:
        return ScoreTest(
            term,
            None,
            None,
            used,
            code,
            SCORE_PROBLEMS[code],
            n_strata=None if strata is None else len(stratum_keys) - len(skipped),
            strata_skipped=tuple(skipped),
        )

    if groups < 2:
        return missing("one_cluster")
    events = float(yy.sum())
    if events == 0.0:
        return missing("no_events")
    if events == float(n):
        return missing("all_events")
    fits: list[_NullFit] = []
    skipped: list[str] = []
    for i, key in enumerate(stratum_keys):
        index = np.flatnonzero(stratum_of == i)
        k = float(yy[index].sum())
        found = (
            None if k in (0.0, float(len(index))) else _stratum_null_fit(index, xx, yy, extra, l2)
        )
        if isinstance(found, str):
            return missing(found, skipped=skipped)
        if found is None:
            skipped.append(key)
            continue
        fits.append(found)
    if not fits:
        return missing(
            "null_separation" if strata is None else "no_stratum_with_variation",
            skipped=skipped if strata is not None else (),
        )
    index = np.concatenate([f.index for f in fits])
    p = np.concatenate([f.p for f in fits])
    resid_x = np.concatenate([f.resid_x for f in fits])
    used_x, used_y = xx[index], yy[index]
    unique, labels = np.unique(cluster_labels[index], return_inverse=True)
    used = len(unique)
    if used < 2:
        return missing("one_cluster", used, skipped)
    if float(np.max(np.abs(resid_x))) <= 1e-9 * max(1.0, float(np.max(np.abs(used_x)))):
        return missing("rank_deficient", used, skipped)
    residual = used_y - p
    root_w = np.sqrt(p * (1.0 - p))
    pearson = residual / root_w
    sizes = np.bincount(labels, minlength=used).astype(np.float64)
    pairs = float(sizes @ (sizes - 1.0))
    rho = 0.0
    if pairs > 0.0:
        sum_e = np.bincount(labels, weights=pearson, minlength=used)
        sum_e2 = np.bincount(labels, weights=pearson * pearson, minlength=used)
        rho = min(max(float((sum_e * sum_e - sum_e2).sum()) / pairs, 0.0), SCORE_RHO_MAX)
    scaled = root_w * resid_x
    together = np.bincount(labels, weights=scaled, minlength=used)
    variance = (1.0 - rho) * float(scaled @ scaled) + rho * float(together @ together)
    if not (math.isfinite(variance) and variance > 0.0):
        return missing("zero_variance", used, skipped)
    z = float(resid_x @ residual) / math.sqrt(variance)
    return ScoreTest(
        term,
        z,
        math.erfc(abs(z) / math.sqrt(2.0)),
        used,
        rho=rho,
        n_rows=len(index),
        n_events=int(used_y.sum()),
        n_strata=None if strata is None else len(fits),
        strata_skipped=tuple(skipped),
        n_boundary=sum(f.boundary for f in fits),
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
    session: Sequence[str] = SESSION_KEY,
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
        cluster: Row field for cluster-robust standard errors (``session_id``, the plan:
            the session within one family, both families' sessions of it when pooled).
        session: Fields that identify a session for the risk set (:data:`SESSION_KEY`).
        check_separation: Run the separation test.

    Raises:
        StatsError: for no rows, missing fields, a covariate with a single value, or
            a rank-deficient design.
    """
    data = risk_set(rows, session) if use_risk_set else [dict(row) for row in rows]
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


@dataclass(frozen=True)
class HazardScoreTest:
    """A score test of one hazard-model term at the null fit (:func:`hazard_score_test`).

    Attributes:
        test: The clustered :class:`ScoreTest`.
        term: Design name of the tested term (``log1p(position)``, ``compaction_count``).
        null_covariates: Design names of the numeric terms of the null model.
        categorical: Factors of the null model.
        position_transform: ``linear`` or ``log1p``.
        strata: Row field whose values are the strata the null model is fit within, or None.
        n_rows: Rows of the risk set (or of the rows, without ``use_risk_set``).
        n_events: Violations among them; :data:`halflife.e1_analysis.Settings.min_events`
            compares this count.
        levels_without_variation: Factor levels whose rows, within their stratum, hold no
            violation or only violations (``stratum: factor[level]`` when stratified); at
            the null fit their fitted values are 0 or 1, so they add nothing and are left
            out of the test instead of separating the null model.
        used_risk_set: Whether rows were cut to the risk set first.
    """

    test: ScoreTest
    term: str
    null_covariates: tuple[str, ...]
    categorical: tuple[str, ...]
    position_transform: PositionTransform
    strata: str | None
    n_rows: int
    n_events: int
    levels_without_variation: tuple[str, ...] = ()
    used_risk_set: bool = True

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "term": self.term,
            "score": self.test.to_dict(),
            "null_covariates": list(self.null_covariates),
            "null_factors": list(self.categorical),
            "position_transform": self.position_transform,
            "strata": self.strata,
            "n_rows": self.n_rows,
            "n_events": self.n_events,
            "levels_without_variation": list(self.levels_without_variation),
            "used_risk_set": self.used_risk_set,
        }


def _levels_without_variation(
    y: FloatArray,
    mine: npt.NDArray[np.bool_],
    factors: Mapping[str, npt.NDArray[np.str_]],
    keep: npt.NDArray[np.bool_],
) -> list[tuple[str, str]]:
    """Drop (in ``keep``) the factor levels of one stratum without variation, repeatedly.

    A level whose kept rows hold no violation or only violations is dropped while the
    stratum still has both outcomes; dropping a level of one factor can leave a level of
    another without variation, so the search repeats until nothing changes. Returns the
    dropped ``(factor, level)`` pairs in order.
    """
    dropped: list[tuple[str, str]] = []
    changed = True
    while changed:
        changed = False
        current = mine & keep
        k = float(y[current].sum())
        if k in (0.0, float(current.sum())):
            break
        for name, values in factors.items():
            for level in sorted(set(values[current].tolist())):
                rows = current & (values == level)
                k_level = float(y[rows].sum())
                if k_level in (0.0, float(rows.sum())):
                    keep &= ~rows
                    dropped.append((name, str(level)))
                    changed = True
            if changed:
                break
    return dropped


def hazard_score_test(
    rows: Iterable[Row],
    term: str,
    covariates: Sequence[str] = DEFAULT_COVARIATES,
    categorical: Sequence[str] = DEFAULT_CATEGORICAL,
    position_transform: PositionTransform = "linear",
    *,
    strata: str | None = None,
    cluster: str = "session_id",
    session: Sequence[str] = SESSION_KEY,
    use_risk_set: bool = True,
    l2: float = DEFAULT_L2,
) -> HazardScoreTest:
    """Score test of one numeric term of the hazard model of :func:`fit_hazard`.

    The rows are cut to the risk set per ``session`` (unless ``use_risk_set`` is False).
    The tested term is ``term`` (``position`` enters through ``position_transform``); the
    null model holds an intercept, the other ``covariates``, and the treatment-coded
    ``categorical`` factors, fit within each value of the ``strata`` field when one is
    named (D-050 item 2: pooled E1 tests fit their nuisance terms within each family, and a
    family without violations adds nothing). Within a stratum, a factor level without
    violations, or with only violations, has no finite null estimate; its rows have fitted
    values 0 or 1 and add nothing to the score or its variance, so they are left out
    (``levels_without_variation``) and the test is the limit of the ridge-free test, which
    an unstratified factor model could only reach through a separated fit. The test itself
    is :func:`score_test`, clustered by ``cluster`` (plans, which pooled hold both families'
    sessions) through an exchangeable within-cluster correlation. A null model separated in
    another way (for example every violation at the largest compaction count of a stratum)
    gives no p-value (``null_separation``).

    Raises:
        StatsError: when ``term`` is not among ``covariates``, for no rows, for missing
            fields, or for repeated or unknown terms.
    """
    _check_transform(position_transform)
    _check_terms(covariates, categorical)
    if term not in covariates:
        raise StatsError(f"the tested term {term!r} is not among the covariates {list(covariates)}")
    data = risk_set(rows, session) if use_risk_set else [dict(row) for row in rows]
    if not data:
        raise StatsError("no opportunity rows to test")
    y = np.array([_outcome(row, "violated", i) for i, row in enumerate(data)], dtype=np.float64)

    def column(name: str) -> FloatArray:
        values = np.array([_number(row, name, i) for i, row in enumerate(data)], dtype=np.float64)
        return np.log1p(values) if name == "position" and position_transform == "log1p" else values

    name = _position_name(position_transform) if term == "position" else term
    x = column(term)
    nuisance = [c for c in covariates if c != term]
    numeric = {c: column(c) for c in nuisance}
    factors = {
        f: np.array([str(_field(row, f, i)) for i, row in enumerate(data)]) for f in categorical
    }
    stratum = np.array(
        [str(_field(row, strata, i)) if strata else "" for i, row in enumerate(data)]
    )
    labels = np.array([str(_field(row, cluster, i)) for i, row in enumerate(data)])
    keep = np.ones(len(data), dtype=bool)
    dropped: list[str] = []
    keys = sorted(set(stratum.tolist()))
    for key in keys:
        for factor, level in _levels_without_variation(y, stratum == key, factors, keep):
            dropped.append(f"{key}: {factor}[{level}]" if strata else f"{factor}[{level}]")
    index = np.flatnonzero(keep)
    columns: list[FloatArray] = []
    for key in keys:
        mine = (stratum[index] == key).astype(np.float64)
        columns += [mine * numeric[c][index] for c in nuisance]
        for factor in categorical:
            values = factors[factor][index]
            present = sorted(set(values[mine > 0].tolist()))
            columns += [mine * (values == level) for level in present[1:]]
    result = HazardScoreTest(
        ScoreTest(name, None, None, len(set(labels.tolist()))),
        name,
        tuple(_position_name(position_transform) if c == "position" else c for c in nuisance),
        tuple(categorical),
        position_transform,
        strata,
        len(data),
        int(y.sum()),
        tuple(dropped),
        use_risk_set,
    )
    if not len(index):
        code = "no_stratum_with_variation"
        empty = ScoreTest(name, None, None, 0, code, SCORE_PROBLEMS[code], n_strata=0)
        return dataclasses.replace(result, test=empty)
    test = score_test(
        x[index],
        y[index],
        labels[index].tolist(),
        np.column_stack(columns) if columns else None,
        term=name,
        l2=l2,
        strata=stratum[index].tolist() if strata else None,
    )
    return dataclasses.replace(result, test=test)


def half_life(
    fit: HazardFit, max_position: float, /, *, first_position: float = 0.0, **fixed: float | str
) -> float | NotReached:
    """The position at which the predicted hazard reaches 0.5, searched from ``first_position``.

    Every covariate other than position is held at the value given in ``fixed`` (for
    example ``compaction_count=0, constraint_type="prohibition"``). The model has one
    position term, so the crossing is solved exactly: ``g(p) = -eta(0) / b``.

    The search runs over ``[first_position, max_position]``, the positions the data can
    speak to: ``first_position`` is the first observed decision-point position (the
    analyses pass it; 0 by default, the D-029 rule). Below it the hazard is never
    extrapolated, so a falling hazard whose extrapolation passes one half before the first
    decision point is not reached, not 0.

    Returns:
        ``float(first_position)`` when the hazard is already at least 0.5 there (the
        half-life is at or before that position, a left-censored value), the crossing
        position (a float, in tool calls) when it lies in ``(first_position,
        max_position]``, and ``NotReached(max_position)`` when the hazard stays below 0.5
        there (including a flat or falling hazard).

    Raises:
        StatsError: for a negative ``max_position``, a ``first_position`` outside
            ``[0, max_position]``, a fit that did not converge, or covariates the fit
            cannot predict for.
    """
    if not (_is_real(max_position) and math.isfinite(max_position) and max_position >= 0):
        raise StatsError(f"max_position must be a finite number >= 0, got {max_position!r}")
    if not (
        _is_real(first_position)
        and math.isfinite(first_position)
        and 0.0 <= first_position <= max_position
    ):
        raise StatsError(
            f"first_position must be a number in [0, max_position], got {first_position!r}"
        )
    if not fit.fit.converged:
        raise StatsError("the hazard fit did not converge, so its half-life is undefined")
    first = float(first_position)
    if fit.linear_predictor(first, **fixed) >= 0.0:
        return first
    slope = fit.position_coefficient
    if slope <= 0.0:
        return NotReached(max_position)
    crossing = -fit.linear_predictor(0.0, **fixed) / slope
    if crossing > _position_term(float(max_position), fit.position_transform):
        return NotReached(max_position)
    value = crossing if fit.position_transform == "linear" else math.expm1(crossing)
    return min(max(value, first), float(max_position))


# --------------------------------------------------------------------------- bootstrap


NOT_ESTIMABLE: dict[str, str] = {
    "no_violations": (
        "the rows hold no violation, so no coefficient has a finite maximum likelihood "
        "estimate; the fit is held only by the ridge"
    ),
    "all_violations": (
        "every row is a violation, so no coefficient has a finite maximum likelihood "
        "estimate; the fit is held only by the ridge"
    ),
    "separation": (
        "the outcomes are separated on this term, so it has no finite maximum likelihood "
        "estimate and its value is held only by the ridge (D-029: not interpreted)"
    ),
    "not_converged": "the fit did not converge, so the estimate is not a maximum",
}
"""Why an estimate is not reported (its point and bounds are null), by reason code."""


@dataclass(frozen=True)
class Interval:
    """A point estimate with a cluster-bootstrap percentile interval.

    ``estimate``, ``lo``, and ``hi`` are numbers or :class:`NotReached` bounds. Draws
    on which the statistic was undefined are counted in ``n_failed`` and left out of
    the percentiles. A statistic with no valid value on the full data is not estimable:
    ``not_estimable`` holds a code from :data:`NOT_ESTIMABLE`, the three values are nan
    (written as null), and no draw is made for it.
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
    not_estimable: str | None = None

    @classmethod
    def undefined(
        cls,
        reason: str,
        *,
        level: float,
        n_boot: int,
        seed: int | None,
        cluster: str = "session_id",
        n_clusters: int = 0,
        strata: str | None = None,
    ) -> Interval:
        """The not-estimable form: no point, no bounds, and the reason code."""
        if reason not in NOT_ESTIMABLE:
            raise StatsError(f"unknown not-estimable reason {reason!r}")
        nan = math.nan
        return cls(
            nan,
            nan,
            nan,
            level,
            n_boot,
            seed=seed,
            cluster=cluster,
            n_clusters=n_clusters,
            strata=strata,
            not_estimable=reason,
        )

    @property
    def n_used(self) -> int:
        """Draws that entered the percentiles."""
        return self.n_boot - self.n_failed

    @property
    def failed_share(self) -> float:
        """Share of the draws on which the statistic was undefined (0 when none ran)."""
        return self.n_failed / self.n_boot if self.n_boot else 0.0

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
        ``hi: null`` plus ``lo_not_reached_within`` or ``hi_not_reached_within``. A
        not-estimable interval has null point and bounds and its ``not_estimable`` code;
        ``failed_share`` is ``n_failed / n_boot``.
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
                "failed_share": _json_float(self.failed_share),
                "seed": self.seed,
                "cluster": self.cluster,
                "n_clusters": self.n_clusters,
                "strata": self.strata,
                "method": self.method,
                "not_estimable": self.not_estimable,
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
    allow_undefined: bool = False,
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
        if not ordered and not allow_undefined:
            raise StatsError(f"the statistic {name!r} was undefined on every resample")
        within = min(bounds[name]) if bounds[name] else 0.0

        def bound(value: float, within: float = within) -> Value:
            return NotReached(within) if math.isinf(value) else value

        out[name] = Interval(
            estimate=estimate,
            # Undefined on every draw (allowed): no bounds, and n_failed says why.
            lo=bound(_percentile(ordered, q)) if ordered else math.nan,
            hi=bound(_percentile(ordered, 1.0 - q)) if ordered else math.nan,
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
    allow_undefined: bool = False,
) -> Interval:
    """Percentile interval for a statistic, resampling whole clusters.

    The default cluster is ``session_id``, the plan: a draw then holds whole sessions
    within one family and, over pooled families, every family's session of each drawn
    plan together. Each draw picks as many clusters as there are, with replacement
    (within each stratum when ``strata`` names a field that is constant within
    clusters). The rows of the cluster in draw slot ``k`` get ``#k`` appended to their
    cluster field (and to ``session_id`` when that is a different field), so a cluster
    drawn twice counts as two sessions of each family, for example in :func:`risk_set`.

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

    With ``allow_undefined``, a statistic that is undefined on every resample gets nan
    bounds (written null) and ``n_failed == n_boot`` instead of an error.

    Raises:
        StatsError: for no rows, no seed, a statistic undefined on the full data, or
            one that is undefined on every resample (unless ``allow_undefined``).
    """
    return _resample(rows, statistic, n_boot, seed, cluster, level, strata, allow_undefined)[
        "value"
    ]


def cluster_bootstrap_many(
    rows: Iterable[Row],
    statistic: Callable[[list[dict[str, Any]]], Mapping[str, Value]],
    n_boot: int = 2000,
    seed: int | None = None,
    cluster: str = "session_id",
    level: float = 0.95,
    *,
    strata: str | None = None,
    allow_undefined: bool = False,
) -> dict[str, Interval]:
    """:func:`cluster_bootstrap` for a statistic that returns several named values.

    All values come from the same draws, so their intervals are consistent with each
    other. A ``nan`` value fails that name only.
    """
    return _resample(rows, statistic, n_boot, seed, cluster, level, strata, allow_undefined)


@dataclass(frozen=True)
class HalfLifeResult:
    """A half-life with its cluster-bootstrap interval and the full-data model.

    ``first_position`` is where the search starts (the first observed decision point). A
    value equal to it, the point or a bound, means the hazard was already at or above one
    half there: the half-life is at or before it. ``share_at_first`` is the share of used
    draws with that value.
    """

    interval: Interval
    fit: HazardFit
    max_position: float
    fixed: Mapping[str, float | str]
    first_position: float = 0.0
    share_at_first: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form: ``half_life``, ``lo``, ``hi`` (or ``not_reached_within``), model."""
        out = self.interval.to_dict("half_life")
        out["max_position"] = _json_bound(self.max_position)
        out["first_position"] = _json_bound(self.first_position)
        out["share_at_first"] = _json_float(self.share_at_first)
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
    check_separation: bool = False,
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
        check_separation=check_separation,
    )


def _flagged_terms(fit: LogisticFit) -> set[str]:
    """Terms a separated fit flags (every term when the test names none)."""
    if not fit.separation:
        return set()
    return set(fit.separated_terms) or set(fit.names)


def not_estimable_terms(model: HazardFit, names: Iterable[str]) -> dict[str, str]:
    """The terms among ``names`` that ``model`` cannot estimate, with a reason code.

    Every term fails when the rows hold no violation or only violations (no coefficient
    has a finite maximum likelihood estimate), or when the fit did not converge; under
    separation the flagged terms fail (D-029: flagged coefficients are not interpreted).
    A fit without the separation check never flags separation, so callers that need the
    rule fit with ``check_separation=True``.
    """
    wanted = list(names)
    if model.n_events == 0:
        return dict.fromkeys(wanted, "no_violations")
    if model.n_events == model.n_rows:
        return dict.fromkeys(wanted, "all_violations")
    if not model.fit.converged:
        return dict.fromkeys(wanted, "not_converged")
    flagged = _flagged_terms(model.fit)
    return {name: "separation" for name in wanted if name in flagged}


@dataclass(frozen=True, eq=False)
class HazardCoefficients(Mapping[str, Interval]):
    """Hazard coefficients with bootstrap intervals, and the full-data model.

    It maps each term to its :class:`Interval`; a term the full data cannot estimate has
    the not-estimable form (:meth:`Interval.undefined`) and no draws.
    """

    intervals: Mapping[str, Interval]
    fit: HazardFit

    def __getitem__(self, name: str) -> Interval:
        return self.intervals[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.intervals)

    def __len__(self) -> int:
        return len(self.intervals)

    @property
    def not_estimable(self) -> dict[str, str]:
        """Not-estimable terms and their reason codes."""
        return {
            name: interval.not_estimable
            for name, interval in self.intervals.items()
            if interval.not_estimable is not None
        }


def bootstrap_half_life(
    rows: Iterable[Row],
    max_position: float,
    fixed: Mapping[str, float | str],
    n_boot: int = 2000,
    seed: int | None = None,
    *,
    level: float = 0.95,
    cluster: str = "session_id",
    session: Sequence[str] = SESSION_KEY,
    covariates: Sequence[str] = DEFAULT_COVARIATES,
    categorical: Sequence[str] = DEFAULT_CATEGORICAL,
    position_transform: PositionTransform = "linear",
    l2: float = DEFAULT_L2,
    first_position: float | None = None,
) -> HalfLifeResult:
    """Half-life from :func:`fit_hazard` with a cluster-bootstrap interval.

    The risk set is formed once, per ``session`` (:data:`SESSION_KEY`); resampling whole
    clusters keeps it valid, because every session lies inside one cluster (its plan).
    The factor levels and the search start (``first_position``, by default the smallest
    position in the rows) are fixed from the full data, and each draw refits the model
    and solves :func:`half_life` from there.

    A half-life is a function of the fitted coefficients, so it follows D-029 (flagged
    coefficients are not interpreted). With no violation the hazard never reaches one
    half (``NotReached``), and with only violations it is at or above one half from the
    first position on; these boundary answers hold for the full data and for each draw.
    Otherwise, when the fit separates a term that enters the prediction at ``fixed``
    (the intercept, the position term, a covariate held at a nonzero value, or the held
    factor level), the full-data half-life is not estimable (no number and no draws), and
    such a draw counts as failed, as do draws where the held level is absent or the
    design is rank deficient.
    """
    data = risk_set(rows, session)
    level = _check_level(level)
    n_boot = _count(n_boot, "n_boot")
    if n_boot < 1:
        raise StatsError("n_boot must be at least 1")
    seed = _require_seed(seed)
    # risk_set is idempotent, so the full fit can cut ``data`` again and say so.
    full = fit_hazard(
        data,
        covariates,
        categorical,
        position_transform,
        use_risk_set=True,
        l2=l2,
        cluster=cluster,
        session=session,
    )
    levels = dict(full.levels)
    first = (
        min(_number(row, "position", i) for i, row in enumerate(data))
        if first_position is None
        else float(first_position)
    )
    full.linear_predictor(first, **fixed)  # validates the held values before resampling
    terms = _prediction_terms(full, fixed)

    def boundary(model: HazardFit) -> Value | None:
        if model.n_events == 0:
            return NotReached(max_position)
        if model.n_events == model.n_rows:
            return first
        return None

    def undefined(model: HazardFit) -> str | None:
        if not model.fit.converged:
            return "not_converged"
        return "separation" if _flagged_terms(model.fit) & terms else None

    reason = undefined(full) if boundary(full) is None else None
    if reason is not None:
        interval = Interval.undefined(
            reason,
            level=level,
            n_boot=n_boot,
            seed=seed,
            cluster=cluster,
            n_clusters=len({str(row[cluster]) for row in data}),
        )
        return HalfLifeResult(interval, full, float(max_position), dict(fixed), first, 0.0)

    values: list[Value] = []

    def statistic(sample: list[dict[str, Any]]) -> Value:
        model = _hazard_statistic_fit(
            sample, covariates, categorical, position_transform, levels, l2, cluster, True
        )
        value = boundary(model)
        if value is None:
            value = (
                math.nan
                if undefined(model)
                else half_life(model, max_position, first_position=first, **fixed)
            )
        values.append(value)
        return value

    interval = cluster_bootstrap(
        data, statistic, n_boot, seed=seed, cluster=cluster, level=level, allow_undefined=True
    )
    draws = [v for v in values[1:] if isinstance(v, NotReached) or not math.isnan(v)]
    at_first = sum(1 for v in draws if not isinstance(v, NotReached) and v == first)
    share = at_first / len(draws) if draws else 0.0
    return HalfLifeResult(interval, full, float(max_position), dict(fixed), first, share)


def _prediction_terms(model: HazardFit, fixed: Mapping[str, float | str]) -> set[str]:
    """Terms that enter the predicted hazard with ``fixed`` held (for the half-life)."""
    terms = {"intercept"}
    for name in model.covariates:
        if name == "position":
            terms.add(_position_name(model.position_transform))
        elif float(fixed[name]) != 0.0:
            terms.add(name)
    for name in model.categorical:
        held = str(fixed[name])
        if held != model.reference[name]:
            terms.add(f"{name}[{held}]")
    return terms


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
    session: Sequence[str] = SESSION_KEY,
    covariates: Sequence[str] = DEFAULT_COVARIATES,
    categorical: Sequence[str] = DEFAULT_CATEGORICAL,
    position_transform: PositionTransform = "linear",
    l2: float = DEFAULT_L2,
) -> HazardCurve:
    """Predicted hazard at each position with pointwise cluster-bootstrap intervals.

    The risk set is formed once per ``session``, as in :func:`bootstrap_half_life`, and
    the same separation rule holds: a fit with violations that separates a term entering
    the prediction has no curve (an error for the full data, a failed draw otherwise),
    while a fit without violations keeps its near-zero curve.

    Raises:
        StatsError: for no positions, held values the fit cannot predict for, or a
            separated full-data fit.
    """
    if not positions:
        raise StatsError("no positions to predict at")
    data = risk_set(rows, session)
    # risk_set is idempotent, so the full fit can cut ``data`` again and say so.
    full = fit_hazard(
        data,
        covariates,
        categorical,
        position_transform,
        use_risk_set=True,
        l2=l2,
        cluster=cluster,
        session=session,
    )
    full.curve(positions, **fixed)  # validates positions and covariates before resampling
    levels = dict(full.levels)
    terms = _prediction_terms(full, fixed)

    def separated(model: HazardFit) -> bool:
        return 0 < model.n_events < model.n_rows and bool(_flagged_terms(model.fit) & terms)

    if separated(full):
        raise StatsError("the hazard fit is separated, so its curve is not estimable (D-029)")

    def statistic(sample: list[dict[str, Any]]) -> dict[str, Value]:
        model = _hazard_statistic_fit(
            sample, covariates, categorical, position_transform, levels, l2, cluster, True
        )
        if separated(model):
            return {str(i): math.nan for i in range(len(positions))}
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
    session: Sequence[str] = SESSION_KEY,
    covariates: Sequence[str] = DEFAULT_COVARIATES,
    categorical: Sequence[str] = DEFAULT_CATEGORICAL,
    position_transform: PositionTransform = "linear",
    l2: float = DEFAULT_L2,
) -> HazardCoefficients:
    """Hazard coefficients (log odds) with cluster-bootstrap intervals.

    Factor terms are effects against the declared reference level, and
    ``intercept`` is the log odds at position 0 with numeric covariates at 0 and
    factors at their references. The risk set is formed once per ``session``, as in
    :func:`bootstrap_half_life`.

    No term is reported at a value held only by the ridge (D-029). A term that the
    full-data fit cannot estimate (:func:`not_estimable_terms`: no violation, only
    violations, no convergence, or a separated term) gets the not-estimable form and no
    draws. Every draw is refit with the separation check, and a term fails for that draw
    (it is left out of the percentiles and counted in ``n_failed``) when the draw has no
    violation or only violations, separates the term, or lacks its level or reference.
    """
    data = risk_set(rows, session)
    level = _check_level(level)
    n_boot = _count(n_boot, "n_boot")
    if n_boot < 1:
        raise StatsError("n_boot must be at least 1")
    seed = _require_seed(seed)
    # risk_set is idempotent, so the full fit can cut ``data`` again and say so.
    full = fit_hazard(
        data,
        covariates,
        categorical,
        position_transform,
        use_risk_set=True,
        l2=l2,
        cluster=cluster,
        session=session,
    )
    levels = dict(full.levels)
    factor_terms = [
        (name, level)
        for name in full.categorical
        for level in full.levels[name][1:]
        if full.level_counts is not None and full.level_counts[name][level] > 0
    ]
    names = ["intercept"]
    names += [_position_name(position_transform) if c == "position" else c for c in covariates]
    names += [f"{name}[{level}]" for name, level in factor_terms]
    undefined = not_estimable_terms(full, names)
    estimable = [name for name in names if name not in undefined]

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
        model = _hazard_statistic_fit(
            sample, covariates, categorical, position_transform, levels, l2, cluster, True
        )
        values = summarize(model)
        failed = not_estimable_terms(model, estimable)
        return {name: math.nan if name in failed else values[name] for name in estimable}

    found: dict[str, Interval] = {}
    if estimable:
        found = cluster_bootstrap_many(
            data,
            statistic,
            n_boot,
            seed=seed,
            cluster=cluster,
            level=level,
            allow_undefined=True,
        )
    n_clusters = len({str(row[cluster]) for row in data})
    for name, reason in undefined.items():
        found[name] = Interval.undefined(
            reason, level=level, n_boot=n_boot, seed=seed, cluster=cluster, n_clusters=n_clusters
        )
    return HazardCoefficients({name: found[name] for name in names}, full)


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


def kaplan_meier_rows(
    rows: Iterable[Row], level: float = 0.95, *, session: Sequence[str] = SESSION_KEY
) -> KaplanMeier:
    """:func:`kaplan_meier` of time to first violation per session from opportunity rows.

    Times come from :func:`halflife.analysis.time_to_first_violation`: the position of
    the first violated opportunity, or of the last one reached (censored), per session
    as the ``session`` fields identify it (:data:`SESSION_KEY`).
    """
    data = list(rows)
    fields = _session_fields(session)
    for index, row in enumerate(data):
        _session_of(row, fields, index)
    triples = time_to_first_violation(data, fields)
    if not triples:
        raise StatsError("no opportunity rows")
    return kaplan_meier([t for _, t, _ in triples], [e for _, _, e in triples], level)


# --------------------------------------------------------------------------- contrasts


@dataclass(frozen=True)
class ArmSummary:
    """Per-opportunity violation rate in one arm; ``n_sessions`` counts sessions by the
    contrast's session key (each family's session of a plan counts once)."""

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
    """Arm ``b`` against reference arm ``a``: rate difference and odds ratio.

    ``wald`` is the two-sided Wald test of the arm term with CR1 errors clustered like
    the bootstrap (by ``cluster``); its ``p_value`` is None, with a reason, when no
    valid test exists (for example under separation). ``score`` is the two-sided score
    test of the same term at the null fit (:func:`score_test`), clustered the same way
    through an exchangeable within-cluster correlation, which stays valid under
    separation; the analyses decide by it. The odds ratio is not estimable (no number)
    when the full-data arm term has no finite estimate.
    """

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
    wald: WaldTest
    score: ScoreTest
    session: tuple[str, ...] = SESSION_KEY
    cluster: str = "session_id"
    score_strata: str | None = None

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
            "wald": self.wald.to_dict(),
            "score": self.score.to_dict(),
            "score_strata": self.score_strata,
            "stratified_by_arm": self.stratified,
            "used_risk_set": self.used_risk_set,
            "session_key": list(self.session),
            "cluster": self.cluster,
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
    session: Sequence[str] = SESSION_KEY,
    covariates: Sequence[str] = (),
    categorical: Sequence[str] = (),
    position_transform: PositionTransform = "linear",
    use_risk_set: bool = False,
    l2: float = DEFAULT_L2,
    strata: str | None = None,
) -> RateContrast:
    """Compare per-opportunity violation rates between two arms.

    Rows whose ``arm_key`` equals ``arm_a`` (the reference) or ``arm_b`` are used;
    other arms are ignored. Reported: each arm's rate, the difference ``b - a``, and
    the odds ratio from a logistic fit of ``violated`` on an indicator of arm ``b``
    (plus optional covariates), all with cluster-bootstrap intervals from the same
    draws, and a Wald test of the arm term (:func:`wald_test`) with CR1 errors over the
    same clusters. When every cluster sits in one arm, clusters are resampled within
    arms so arm sizes stay fixed. The odds-ratio interval is the exponential of the log
    odds-ratio interval. By default every opportunity counts (``use_risk_set``
    False), so the rate is the share of decision points with a violation; with
    ``use_risk_set`` the rows are cut per ``session`` first. Sessions per arm are
    counted by ``session`` (:data:`SESSION_KEY`).

    With ``strata`` (a row field, such as ``family`` for a pooled contrast; D-050 item 2)
    the score test that decides the contrast fits its null model within each stratum, so a
    stratum without violations adds nothing (:func:`score_test`); the rates, the
    difference, and the odds ratio stay pooled descriptive estimates.

    Raises:
        StatsError: for equal arms, a row without ``arm_key`` (or the ``strata`` field), or
            an arm with no rows.
    """
    if arm_a == arm_b:
        raise StatsError("arm_a and arm_b must differ")
    fields = _session_fields(session)
    data = [dict(row) for row in rows]
    for index, row in enumerate(data):
        _field(row, arm_key, index)
    selected = [row for row in data if row[arm_key] == arm_a or row[arm_key] == arm_b]
    if use_risk_set:
        selected = risk_set(selected, fields)
    for arm in (arm_a, arm_b):
        if not any(row[arm_key] == arm for row in selected):
            raise StatsError(f"no opportunities in arm {arm!r} of {arm_key!r}")
    term = f"{arm_key}[{arm_b}]"
    levels: Mapping[str, Sequence[str]] | None = None
    if categorical:
        levels = _build_design(selected, covariates, categorical, position_transform, None).levels
    adjusted = bool(covariates or categorical)

    def arms_of(sample: Sequence[Row]) -> tuple[FloatArray, FloatArray]:
        indicator = np.array([1.0 if row[arm_key] == arm_b else 0.0 for row in sample])
        y = np.array([_outcome(row, "violated", i) for i, row in enumerate(sample)], dtype=float)
        n_b = float(indicator.sum())
        if n_b in (0.0, float(len(sample))):
            raise StatsError("a resample has no opportunities in one arm")
        return indicator, y

    def covariate_matrix(sample: Sequence[Row]) -> tuple[FloatArray, tuple[str, ...]]:
        design = _build_design(sample, covariates, categorical, position_transform, levels)
        return design.matrix, design.names

    def model(
        sample: Sequence[Row], check: bool, labels: Sequence[str] | None = None
    ) -> LogisticFit:
        indicator, y = arms_of(sample)
        x = indicator.reshape(-1, 1)
        names: tuple[str, ...] = (term,)
        if adjusted:
            matrix, extra = covariate_matrix(sample)
            x = np.column_stack([indicator, matrix])
            names = (term, *extra)
        fit = fit_logistic(x, y, l2, names=names, clusters=labels, check_separation=check)
        if "rank_deficient" in fit.warnings:
            raise StatsError("the contrast design is rank deficient")
        return fit

    def one_sided(indicator: FloatArray, y: FloatArray) -> bool:
        """An arm with no violations or only violations: no finite log odds ratio."""
        for mask in (indicator == 0.0, indicator == 1.0):
            k = float(y[mask].sum())
            if k == 0.0 or k == float(mask.sum()):
                return True
        return False

    # The full-data fit carries CR1 errors over the bootstrap's clusters, for the Wald
    # test; resample fits only need their coefficient, so they skip the sandwich.
    labels = [str(_field(row, cluster, index)) for index, row in enumerate(selected)]
    full_fit = model(selected, True, labels)
    wald = wald_test(full_fit, term, robust=True)
    indicator, outcomes = arms_of(selected)
    score = score_test(
        indicator,
        outcomes,
        labels,
        covariate_matrix(selected)[0] if adjusted else None,
        term=term,
        l2=l2,
        strata=(
            None
            if strata is None
            else [str(_field(row, strata, index)) for index, row in enumerate(selected)]
        ),
    )
    or_missing: str | None = None
    if full_fit.n_events in (0, full_fit.n):
        or_missing = "no_violations" if full_fit.n_events == 0 else "all_violations"
    elif term in _flagged_terms(full_fit):
        or_missing = "separation"
    clusters: dict[str, set[bool]] = {}
    for label, row in zip(labels, selected, strict=True):
        clusters.setdefault(label, set()).add(row[arm_key] == arm_b)
    stratified = all(len(arms) == 1 for arms in clusters.values())

    def statistic(sample: list[dict[str, Any]]) -> dict[str, Value]:
        arm, y = arms_of(sample)
        rate_a, rate_b = float(y[arm == 0.0].mean()), float(y[arm == 1.0].mean())
        out: dict[str, Value] = {"rate_a": rate_a, "rate_b": rate_b, "difference": rate_b - rate_a}
        if or_missing is None:
            # A draw whose log odds ratio has no finite estimate (an arm with no or only
            # violations, or a separated arm term) fails for the odds ratio only.
            if not adjusted:
                out["log_odds_ratio"] = (
                    math.nan if one_sided(arm, y) else model(sample, False).coefficient(term)
                )
            else:
                fit = model(sample, True)
                flagged = fit.n_events in (0, fit.n) or term in _flagged_terms(fit)
                out["log_odds_ratio"] = math.nan if flagged else fit.coefficient(term)
        return out

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
        sessions = {_session_of(row, fields, i) for i, row in enumerate(arm_rows)}
        return ArmSummary(
            str(arm),
            len(sessions),
            len(arm_rows),
            k,
            rate,
            wilson_interval(k, len(arm_rows), level),
        )

    if or_missing is None:
        log_or = found["log_odds_ratio"]
    else:
        log_or = Interval.undefined(
            or_missing,
            level=found["difference"].level,
            n_boot=found["difference"].n_boot,
            seed=found["difference"].seed,
            cluster=cluster,
            n_clusters=found["difference"].n_clusters,
            strata=found["difference"].strata,
        )
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
        wald=wald,
        score=score,
        session=fields,
        cluster=cluster,
        score_strata=strata,
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


RECALL_COUNT_FIELD = "compactions_before_probe"
"""Compactions the before-DP probe saw (:func:`halflife.analysis.opportunity_rows`)."""


def recall_count_field(rows: Sequence[Row]) -> str:
    """The compaction count recall is grouped by: the one the probe saw, when every row has it.

    A compaction at a decision point's start index runs after that point's probe, so the
    probe's recall belongs with :data:`RECALL_COUNT_FIELD`; rows banked before that field
    existed fall back to ``compaction_count``, and the outputs name the field used.
    """
    return (
        RECALL_COUNT_FIELD
        if all(RECALL_COUNT_FIELD in row for row in rows)
        else ("compaction_count")
    )


@dataclass(frozen=True)
class RecallCell:
    """Probe recall among opportunities with one compaction count (``count_field``)."""

    compaction_count: int
    n_sessions: int
    recall: Proportion
    count_field: str = "compaction_count"

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "compaction_count": self.compaction_count,
            "count_field": self.count_field,
            "n_sessions": self.n_sessions,
            "recall": {**self.recall.to_dict(), "note": NAIVE_NOTE},
        }


def recall_by_compaction(
    rows: Iterable[Row],
    level: float = 0.95,
    session: Sequence[str] = SESSION_KEY,
    count_field: str | None = None,
) -> list[RecallCell]:
    """Recall rate before each decision point, by compaction count (descriptive).

    The count is ``count_field``, by default the compactions the probe saw
    (:func:`recall_count_field`). ``n_sessions`` counts sessions by the ``session``
    fields (:data:`SESSION_KEY`).

    Raises:
        StatsError: when no row has a recall outcome, or a compaction count is not
            a whole number.
    """
    fields = _session_fields(session)
    data = recall_rows(rows)
    if not data:
        raise StatsError("no rows with a recall outcome (recalled_before)")
    name = count_field or recall_count_field(data)
    cells: dict[int, list[dict[str, Any]]] = {}
    for index, row in enumerate(data):
        count = _number(row, name, index)
        if not count.is_integer() or count < 0:
            raise StatsError(f"row {index}: {name} must be a whole number >= 0")
        cells.setdefault(int(count), []).append(row)
    out = []
    for count in sorted(cells):
        group = cells[count]
        k = sum(_outcome(row, "recalled_before", i) for i, row in enumerate(group))
        sessions = {_session_of(row, fields, i) for i, row in enumerate(group)}
        out.append(RecallCell(count, len(sessions), wilson_interval(k, len(group), level), name))
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
    session: Sequence[str] = SESSION_KEY,
    covariates: Sequence[str] = DEFAULT_COVARIATES,
    categorical: Sequence[str] = (),
    position_transform: PositionTransform = "linear",
    l2: float = DEFAULT_L2,
    count_field: str | None = None,
) -> RecallEffect:
    """Logistic model of probe recall on compaction count (RQ2).

    Every opportunity with a probe counts (recall is not absorbing, so there is no
    risk set). With randomized compaction timing and ``position`` among the
    covariates, the compaction coefficient compares opportunities at the same position
    that saw different numbers of compactions. The count is ``count_field``, by default
    the compactions the probe saw (:func:`recall_count_field`); it replaces
    ``compaction_count`` among the covariates. The interval is a cluster bootstrap over
    ``cluster`` (plans by default, so pooled families stay together); the per-count
    table counts sessions by ``session`` and uses naive Wilson intervals.

    Raises:
        StatsError: when no row has a recall outcome or ``compaction_count`` is not a
            covariate.
    """
    if "compaction_count" not in covariates:
        raise StatsError("recall_effect needs compaction_count among the covariates")
    data = recall_rows(rows)
    if not data:
        raise StatsError("no rows with a recall outcome (recalled_before)")
    count = count_field or recall_count_field(data)
    covariates = tuple(count if name == "compaction_count" else name for name in covariates)
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
        return fit.coefficient(count)

    log_or = cluster_bootstrap(data, statistic, n_boot, seed=seed, cluster=cluster, level=level)
    return RecallEffect(
        cells=tuple(recall_by_compaction(data, level, session, count)),
        fit=full,
        term=count,
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

    ``compaction``: rejection rate of the two-sided score test of the compaction term at
    the null fit of the hazard model on the risk set (:func:`score_test`, clustered by
    session), with ``n_sessions`` sessions. ``contrast``: the same test of the arm term
    among decision points after a compaction, with ``arm_sessions`` sessions in each of
    two arms. These are the tests that decide H2b and H2a (DECISIONS.md D-050 items 1 and
    6). Simulations without a p-value (no violation, a null model that separates every
    row, one cluster) count as non-rejections and are tallied in ``*_unusable``.
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
    arm_sessions: int
    position_transform: PositionTransform = "log1p"
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
                "arm_sessions": self.arm_sessions,
                "opportunities_per_session": self.opportunities_per_session,
                "base_rate": self.base_rate,
                "n_sim": self.n_sim,
                "seed": self.seed,
                "alpha": self.alpha,
                "max_position": self.max_position,
                "compactions_per_session": self.compactions_per_session,
                "position_slope": self.position_slope,
                "position_transform": self.position_transform,
                "session_sd": self.session_sd,
            },
            "test": (
                "two-sided score test at the null fit, clustered by session through an "
                "exchangeable within-session correlation (the test that decides H2a and H2b)"
            ),
            "compaction": {
                **self.compaction.to_dict(),
                "unusable": self.compaction_unusable,
                "design": (
                    "risk set to first violation; terms "
                    f"{_position_name(self.position_transform)} and compaction_count (the "
                    "E1 factors are not simulated); score test of compaction_count with "
                    f"{_position_name(self.position_transform)} in the null model, as H2b"
                ),
            },
            "contrast": {
                **self.contrast.to_dict(),
                "unusable": self.contrast_unusable,
                "design": (
                    f"{self.arm_sessions} sessions per arm; decision points after at least "
                    "one compaction, where the arms differ, as in H2a; score test of the "
                    "arm term at the overall rate, as H2a"
                ),
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
    *,
    after_shift: float = 0.0,
    transform: PositionTransform = "linear",
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Positions, compaction counts, outcomes, and session ids (sessions by opportunities).

    ``after_shift`` adds to the log odds of opportunities after at least one compaction
    only (an arm effect that, as in H2a, exists only once a compaction has happened), and
    ``position_slope`` multiplies the position term under ``transform``.
    """
    positions = np.sort(
        np.argsort(rng.random((n_sessions, max_position + 1)), axis=1)[:, :opportunities], axis=1
    ).astype(np.float64)
    events = np.argsort(rng.random((n_sessions, max_position)), axis=1)[:, :compactions] + 1
    counts = (events[:, None, :] <= positions[:, :, None]).sum(axis=2).astype(np.float64)
    frailty = rng.normal(0.0, session_sd, size=(n_sessions, 1)) if session_sd > 0 else 0.0
    term = positions if transform == "linear" else np.log1p(positions)
    eta = (
        intercept
        + position_slope * term
        + compaction_effect * counts
        + after_shift * (counts >= 1.0)
        + frailty
    )
    outcomes = (rng.random(positions.shape) < _sigmoid(eta)).astype(np.float64)
    sessions = np.repeat(np.arange(n_sessions, dtype=np.float64)[:, None], opportunities, axis=1)
    return positions, counts, outcomes, sessions


def _rejects(test: ScoreTest, alpha: float) -> bool | None:
    """True or False for a computed score test, None when it has no p-value."""
    return None if test.p_value is None else test.p_value < alpha


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
    arm_sessions: int | None = None,
    l2: float = DEFAULT_L2,
    position_transform: PositionTransform = "log1p",
) -> PowerResult:
    """Simulated power for the compaction coefficient and for a two-arm contrast.

    The hazard design has ``n_sessions`` sessions. The contrast design has
    ``arm_sessions`` sessions in each arm (``n_sessions`` when None); a contrast between
    two arms that one family's ``n`` sessions are randomized to passes ``n // 2``.

    Each simulated session has ``opportunities_per_session`` decision points at
    distinct positions drawn uniformly from ``0..max_position`` and
    ``compactions_per_session`` compaction events at distinct indices drawn from
    ``1..max_position``; an opportunity's compaction count is the number of events at
    or before its position. The hazard is
    ``logit h = logit(base_rate) + position_slope * g(position) + effect * count + u``
    with ``g`` the ``position_transform`` (``log1p`` by default, as the E1 hazard model)
    and ``u ~ Normal(0, session_sd)`` per session. ``effect`` is a log odds ratio: per
    compaction event in the first design, fitted with ``g(position)`` and the count (the
    E1 model without its factors). In the second design, as in H2a, the arms differ only
    after a compaction: arm ``b`` adds ``effect`` to the log odds of its opportunities
    with at least one compaction before them, compaction has no other effect, and the
    arm term is tested on those opportunities only. Both designs decide by the score tests
    that decide H2b and H2a (:func:`score_test` at the null fit, clustered by session
    through an exchangeable within-session correlation; DECISIONS.md D-050 items 1 and 6):
    the compaction count with ``g(position)`` in the null model, and the arm at the overall
    rate. Each design draws from its own child of ``seed``.

    Raises:
        StatsError: for out-of-range inputs, including no compaction events.
    """
    _check_transform(position_transform)
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
    per_arm = n_sessions if arm_sessions is None else _count(arm_sessions, "arm_sessions")
    if per_arm < 1:
        raise StatsError("arm_sessions must be at least 1")
    if n_sessions < 2 or n_sim < 1 or opportunities < 1:
        raise StatsError("need n_sessions >= 2, n_sim >= 1, and opportunities_per_session >= 1")
    if opportunities > max_position + 1 or compactions > max_position:
        raise StatsError("more decision points or compactions than positions")
    if compactions == 0:
        raise StatsError("compactions_per_session must be >= 1 to estimate a compaction effect")
    intercept = math.log(base_rate / (1.0 - base_rate))
    rng_compaction, rng_contrast = (
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(2)
    )
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
            transform=position_transform,
        )
        keep = (np.cumsum(y, axis=1) - y) == 0
        term = pos[keep] if position_transform == "linear" else np.log1p(pos[keep])
        test = score_test(cnt[keep], y[keep], list(ses[keep]), term, term="compaction_count", l2=l2)
        verdict = _rejects(test, alpha)
        unusable += verdict is None
        hits += bool(verdict)
    arm_hits = arm_unusable = 0
    for _ in range(n_sim):
        outcomes: list[FloatArray] = []
        session_ids: list[FloatArray] = []
        for arm, shift in enumerate((0.0, effect)):
            pos, cnt, y, ses = _simulate(
                rng_contrast,
                per_arm,
                opportunities,
                max_position,
                compactions,
                intercept,
                position_slope,
                0.0,
                session_sd,
                after_shift=shift,
                transform=position_transform,
            )
            after = cnt >= 1.0
            outcomes.append(y[after])
            session_ids.append(ses[after] + arm * per_arm)
        if not (outcomes[0].size and outcomes[1].size):
            arm_unusable += 1  # no decision point after a compaction in one arm
            continue
        y_all = np.concatenate(outcomes)
        indicator = np.concatenate([np.zeros(outcomes[0].size), np.ones(outcomes[1].size)])
        sessions = np.concatenate(session_ids)
        verdict = _rejects(score_test(indicator, y_all, list(sessions), term="arm", l2=l2), alpha)
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
        arm_sessions=per_arm,
        position_transform=position_transform,
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
    "NOT_ESTIMABLE",
    "POSITION_TRANSFORMS",
    "POWER_NOTE",
    "RECALL_COUNT_FIELD",
    "SCORE_PROBLEMS",
    "SCORE_RHO_MAX",
    "SCORE_TEST_NAME",
    "SEPARATION_DRIFT",
    "SESSION_KEY",
    "WALD_PROBLEMS",
    "ArmSummary",
    "HalfLifeResult",
    "HazardCoefficients",
    "HazardCurve",
    "HazardFit",
    "HazardScoreTest",
    "Interval",
    "KaplanMeier",
    "LogisticFit",
    "NotReached",
    "PowerResult",
    "Proportion",
    "RateContrast",
    "RecallCell",
    "RecallEffect",
    "ScoreTest",
    "StatsError",
    "WaldTest",
    "bootstrap_coefficients",
    "bootstrap_half_life",
    "bootstrap_hazard_curve",
    "cluster_bootstrap",
    "cluster_bootstrap_many",
    "describe_interval",
    "fit_hazard",
    "fit_logistic",
    "half_life",
    "hazard_score_test",
    "kaplan_meier",
    "kaplan_meier_rows",
    "normal_cdf",
    "normal_quantile",
    "not_estimable_terms",
    "power_by_simulation",
    "rate_contrast",
    "recall_by_compaction",
    "recall_count_field",
    "recall_effect",
    "recall_rows",
    "risk_set",
    "score_test",
    "wald_test",
    "wilson_interval",
]
