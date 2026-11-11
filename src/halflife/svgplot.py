"""Deterministic SVG figures for half-life reports (docs/DESIGN.md, ``svgplot``).

Four chart forms, written without a plotting library:

- :func:`line_chart`: lines with optional interval bands (for example hazard by position).
- :func:`step_chart`: step curves with bands and censoring marks (Kaplan-Meier).
- :func:`bar_chart`: grouped bars with error bars.
- :func:`forest_plot`: estimates with intervals against a reference line.

Each function returns the SVG document as text and :func:`write_svg` saves it. The
output depends only on the arguments: coordinates use fixed two-decimal formatting,
elements come out in a fixed order, ids derive from the title, and nothing reads
the clock, the locale, or a random source, so the same input gives byte-identical
files. Every figure has ``<title>``, ``<desc>``, a ``viewBox``, a legend, and ticks,
and draws its own light background so it reads the same on light and dark pages.

Colors: series take the slots of :data:`PALETTE` in fixed order (color follows the
series, never its rank). It is the reference categorical palette of the
data-visualization guidance this project follows, and its validator passes it on
:data:`SURFACE` (lightness band, chroma floor, adjacent-pair color-vision-deficiency
separation, normal-vision floor). The same check warns that slots 3 to 5 sit below
3:1 contrast against the surface, so a figure that uses them needs visible labels or
a table of the plotted values beside it; report tables serve as that table view.
Color is never the only cue: lines also differ by dash pattern (and by marker
shape in line charts), bars keep a fixed left-to-right order within each group,
and every figure has a legend. Text uses ink colors, never series colors.

Text passed in may not contain em dashes (the project style rule) or control
characters. A figure of synthetic data says so on its face (``synthetic=True``).
"""

from __future__ import annotations

import itertools
import math
import numbers
import re
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PALETTE: tuple[str, ...] = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
DASHES: tuple[str, ...] = ("", "7 4", "2 3", "10 3 2 3", "4 4", "1 3", "12 4", "6 2 2 2")
MARKERS: tuple[str, ...] = (
    "circle",
    "square",
    "diamond",
    "triangle-up",
    "triangle-down",
    "triangle-left",
    "triangle-right",
    "hexagon",
)
MAX_SERIES = len(PALETTE)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
FONT_FAMILY = "system-ui, -apple-system, 'Segoe UI', sans-serif"

TITLE_SIZE = 16
SUBTITLE_SIZE = 13
LABEL_SIZE = 13
TICK_SIZE = 12
LEGEND_SIZE = 12
NOTE_SIZE = 11
BAND_OPACITY = "0.16"
SYNTHETIC_LABEL = "Synthetic data: not model results or real-world results"

_EM_DASH = "\u2014"  # escaped so this source file holds no em dash itself
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_SQRT3_2 = 0.8660254037844386
_PAD = 16.0


class PlotError(ValueError):
    """Raised for data or labels a figure cannot show."""


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class Series:
    """One line of a line chart, with an optional interval band.

    ``y``, ``lower``, and ``upper`` may hold None (or nan) for missing values; the
    line breaks there.
    """

    label: str
    x: Sequence[float]
    y: Sequence[float | None]
    lower: Sequence[float | None] | None = None
    upper: Sequence[float | None] | None = None


@dataclass(frozen=True)
class StepSeries:
    """One right-continuous step curve, for example a Kaplan-Meier estimate.

    The curve starts at ``start`` and jumps to ``y[i]`` at ``x[i]``. ``lower`` and
    ``upper`` hold the band after each jump (None where undefined). ``end_x``
    extends the last step; ``censored_x`` places censoring ticks on the curve.
    """

    label: str
    x: Sequence[float]
    y: Sequence[float]
    lower: Sequence[float | None] | None = None
    upper: Sequence[float | None] | None = None
    start: tuple[float, float] = (0.0, 1.0)
    end_x: float | None = None
    censored_x: Sequence[float] = ()


@dataclass(frozen=True)
class BarSeries:
    """One series of a bar chart: a value per category, with optional error bars."""

    label: str
    values: Sequence[float | None]
    lower: Sequence[float | None] | None = None
    upper: Sequence[float | None] | None = None


@dataclass(frozen=True)
class ForestRow:
    """One row of a forest plot.

    ``lo_open`` or ``hi_open`` mark an interval that continues past the axis (for
    example a half-life bound ``>150``); it is drawn to the edge with an arrow.
    ``text`` is shown in a column on the right. ``group`` picks the color.
    """

    label: str
    estimate: float | None
    lo: float | None = None
    hi: float | None = None
    text: str = ""
    lo_open: bool = False
    hi_open: bool = False
    group: str | None = None


# --------------------------------------------------------------------------- formatting


def _num(value: float) -> str:
    """Coordinate text with two decimals, never ``-0.00``."""
    text = f"{value:.2f}"
    return "0.00" if text == "-0.00" else text


def _short(value: float) -> str:
    """A number for tooltips and descriptions (up to 4 significant digits)."""
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.4g}"


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _check_text(value: object, what: str) -> str:
    if not isinstance(value, str):
        raise PlotError(f"{what} must be a string, got {value!r}")
    if _EM_DASH in value:
        raise PlotError(f"{what} contains an em dash; the project style rule forbids them")
    if _CONTROL.search(value):
        raise PlotError(f"{what} contains a control character")
    return value


def _finite(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise PlotError(f"{what} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise PlotError(f"{what} must be finite, got {value!r}")
    return number


def _optional(value: object, what: str) -> float | None:
    """A finite number, or None for a missing value (None or nan)."""
    if value is None:
        return None
    if isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isnan(value):
        return None
    return _finite(value, what)


def _text_width(text: str, size: float) -> float:
    """Estimated rendered width of ``text`` in a sans-serif face (deterministic)."""
    total = 0.0
    for char in text:
        if char in " il.,:;|!'`":
            total += 0.28
        elif char in "fjrt()[]{}/-":
            total += 0.36
        elif char in "mwMW%@":
            total += 0.86
        elif char.isupper():
            total += 0.66
        elif char.isdigit():
            total += 0.56
        else:
            total += 0.53
    return total * size


def _decimals(step: float) -> int:
    for digits in range(10):
        if abs(round(step, digits) - step) <= 1e-9 * max(1.0, abs(step)):
            return digits
    return 10


def _clean(value: float) -> float:
    """``value`` rounded to 12 significant digits, which drops float noise (0.1 * 3)."""
    value = float(f"{value:.12g}")
    return 0.0 if value == 0 else value


def _tick_labels(ticks: Sequence[float]) -> list[str]:
    step = ticks[1] - ticks[0] if len(ticks) > 1 else 1.0
    digits = _decimals(step)
    labels = []
    for tick in ticks:
        text = f"{tick:,.{digits}f}" if abs(tick) >= 10000 else f"{tick:.{digits}f}"
        if float(text.replace(",", "")) == 0:
            text = text.lstrip("-")
        labels.append(text)
    return labels


def _log_label(value: float) -> str:
    return f"{value:g}"


def nice_ticks(lo: float, hi: float, max_ticks: int = 6) -> list[float]:
    """Round tick values (steps of 1, 2, 2.5, or 5 times a power of ten) covering ``[lo, hi]``.

    The first tick is at or below ``lo`` and the last at or above ``hi``. Equal
    bounds are widened so there is always a span.

    Raises:
        PlotError: for non-finite bounds, ``hi < lo``, or ``max_ticks < 2``.
    """
    lo = _finite(lo, "lo")
    hi = _finite(hi, "hi")
    if hi < lo:
        raise PlotError(f"tick range is reversed ({lo} > {hi})")
    if max_ticks < 2:
        raise PlotError("max_ticks must be at least 2")
    if hi == lo:
        pad = abs(lo) * 0.1 if lo else 1.0
        lo, hi = lo - pad, hi + pad
    raw = (hi - lo) / (max_ticks - 1)
    base = 10.0 ** math.floor(math.log10(raw))
    step = base
    for multiple in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = multiple * base
        first = math.floor(lo / step + 1e-9)
        last = math.ceil(hi / step - 1e-9)
        if last - first <= max_ticks - 1:
            break
    first = math.floor(lo / step + 1e-9)
    last = math.ceil(hi / step - 1e-9)
    return [_clean(i * step) for i in range(first, last + 1)]


def log_ticks(lo: float, hi: float) -> list[float]:
    """Ticks at 1, 2, and 5 times powers of ten covering ``[lo, hi]`` (both positive).

    Long ranges keep only the powers of ten and the two ends.

    Raises:
        PlotError: for non-positive or reversed bounds.
    """
    lo = _finite(lo, "lo")
    hi = _finite(hi, "hi")
    if lo <= 0 or hi <= 0:
        raise PlotError("a log axis needs positive values")
    if hi < lo:
        raise PlotError(f"tick range is reversed ({lo} > {hi})")
    if hi == lo:
        lo, hi = lo / 2.0, hi * 2.0
    k_lo = math.floor(math.log10(lo))
    k_hi = math.ceil(math.log10(hi))
    candidates = sorted(
        {_clean(m * 10.0**k) for k in range(k_lo - 1, k_hi + 1) for m in (1.0, 2.0, 5.0)}
    )
    below = [c for c in candidates if c <= lo * (1 + 1e-12)]
    above = [c for c in candidates if c >= hi * (1 - 1e-12)]
    start = below[-1]
    end = above[0]
    ticks = [c for c in candidates if start <= c <= end]
    if len(ticks) > 8:
        decades = [c for c in ticks if math.log10(c) == round(math.log10(c))]
        ticks = sorted({start, end, *decades})
    return ticks


# --------------------------------------------------------------------------- elements


def _attrs(attributes: Sequence[tuple[str, str | None]]) -> str:
    return "".join(f' {name}="{_esc(value)}"' for name, value in attributes if value is not None)


def _element(
    tag: str, attributes: Sequence[tuple[str, str | None]], body: str | None = None
) -> str:
    if body is None:
        return f"<{tag}{_attrs(attributes)}/>"
    return f"<{tag}{_attrs(attributes)}>{body}</{tag}>"


def _text(
    x: float,
    y: float,
    content: str,
    *,
    size: int,
    fill: str = INK,
    anchor: str | None = None,
    weight: str | None = None,
    rotate: bool = False,
    cls: str | None = None,
) -> str:
    return _element(
        "text",
        [
            ("x", _num(x)),
            ("y", _num(y)),
            ("font-size", str(size)),
            ("fill", fill),
            ("text-anchor", anchor),
            ("font-weight", weight),
            ("transform", f"rotate(-90 {_num(x)} {_num(y)})" if rotate else None),
            ("class", cls),
        ],
        _esc(content),
    )


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str,
    width: str = "1",
    dash: str | None = None,
    cls: str | None = None,
) -> str:
    return _element(
        "line",
        [
            ("x1", _num(x1)),
            ("y1", _num(y1)),
            ("x2", _num(x2)),
            ("y2", _num(y2)),
            ("stroke", stroke),
            ("stroke-width", width),
            ("stroke-dasharray", dash or None),
            ("class", cls),
        ],
    )


def _path(
    d: str,
    *,
    fill: str = "none",
    stroke: str | None = None,
    width: str | None = None,
    dash: str | None = None,
    opacity: str | None = None,
    cls: str | None = None,
    tooltip: str | None = None,
) -> str:
    attributes: list[tuple[str, str | None]] = [
        ("d", d),
        ("fill", fill),
        ("fill-opacity", opacity),
        ("stroke", stroke),
        ("stroke-width", width),
        ("stroke-dasharray", dash or None),
    ]
    if stroke is not None and fill == "none":
        attributes += [("stroke-linejoin", "round"), ("stroke-linecap", "round")]
    attributes.append(("class", cls))
    body = None if tooltip is None else f"<title>{_esc(tooltip)}</title>"
    return _element("path", attributes, body)


def _marker(
    shape: str, x: float, y: float, color: str, r: float = 4.0, tooltip: str | None = None
) -> str:
    """A filled marker with a 2px ring in the surface color."""
    body = None if tooltip is None else f"<title>{_esc(tooltip)}</title>"
    paint: list[tuple[str, str | None]] = [
        ("fill", color),
        ("stroke", SURFACE),
        ("stroke-width", "2"),
        ("class", "marker"),
    ]
    if shape == "circle":
        return _element("circle", [("cx", _num(x)), ("cy", _num(y)), ("r", _num(r)), *paint], body)
    if shape == "square":
        s = r * 0.9
        points = [(x - s, y - s), (x + s, y - s), (x + s, y + s), (x - s, y + s)]
    elif shape == "diamond":
        k = r * 1.3
        points = [(x, y - k), (x + k, y), (x, y + k), (x - k, y)]
    elif shape == "triangle-up":
        points = [(x, y - 1.3 * r), (x + 1.2 * r, y + 0.8 * r), (x - 1.2 * r, y + 0.8 * r)]
    elif shape == "triangle-down":
        points = [(x, y + 1.3 * r), (x + 1.2 * r, y - 0.8 * r), (x - 1.2 * r, y - 0.8 * r)]
    elif shape == "triangle-left":
        points = [(x - 1.3 * r, y), (x + 0.8 * r, y - 1.2 * r), (x + 0.8 * r, y + 1.2 * r)]
    elif shape == "triangle-right":
        points = [(x + 1.3 * r, y), (x - 0.8 * r, y - 1.2 * r), (x - 0.8 * r, y + 1.2 * r)]
    else:
        h = r * 1.15
        offsets = [(1.0, 0.0), (0.5, _SQRT3_2), (-0.5, _SQRT3_2), (-1.0, 0.0)]
        offsets += [(-0.5, -_SQRT3_2), (0.5, -_SQRT3_2)]
        points = [(x + h * dx, y + h * dy) for dx, dy in offsets]
    d = "M" + " L".join(f"{_num(px)} {_num(py)}" for px, py in points) + " Z"
    return _element("path", [("d", d), *paint], body)


def _polyline(points: Sequence[tuple[float, float] | None]) -> str:
    """Path data that starts a new subpath after every gap (None)."""
    parts: list[str] = []
    pen_down = False
    for point in points:
        if point is None:
            pen_down = False
            continue
        parts.append(f"{'L' if pen_down else 'M'}{_num(point[0])} {_num(point[1])}")
        pen_down = True
    return " ".join(parts)


def _band(xs: Sequence[float], lows: Sequence[float | None], highs: Sequence[float | None]) -> str:
    """Closed areas between ``lows`` and ``highs`` over runs where both exist (pixels)."""
    runs: list[list[int]] = [[]]
    for i in range(len(xs)):
        if lows[i] is None or highs[i] is None:
            runs.append([])
        else:
            runs[-1].append(i)
    shapes = []
    for run in runs:
        if len(run) < 2:
            continue
        upper = [(xs[i], highs[i]) for i in run]
        lower = [(xs[i], lows[i]) for i in reversed(run)]
        points = [(x, y) for x, y in upper + lower if y is not None]
        shapes.append("M" + " L".join(f"{_num(x)} {_num(y)}" for x, y in points) + " Z")
    return " ".join(shapes)


# --------------------------------------------------------------------------- layout


@dataclass(frozen=True)
class _Key:
    kind: str
    label: str
    color: str = INK_SECONDARY
    dash: str = ""
    marker: str | None = None


@dataclass(frozen=True)
class _Scale:
    lo: float
    hi: float
    start: float
    end: float
    log: bool = False

    def __call__(self, value: float) -> float:
        if self.log:
            a, b, v = math.log10(self.lo), math.log10(self.hi), math.log10(value)
        else:
            a, b, v = self.lo, self.hi, value
        if b == a:
            return (self.start + self.end) / 2.0
        return self.start + (v - a) / (b - a) * (self.end - self.start)


def _legend_key(key: _Key, x: float, cy: float) -> list[str]:
    if key.kind == "line":
        out = [_line(x, cy, x + 26, cy, stroke=key.color, width="2", dash=key.dash)]
        if key.marker is not None:
            out.append(_marker(key.marker, x + 13, cy, key.color))
        return out
    if key.kind == "band":
        rect = f"M{_num(x)} {_num(cy - 6)} H{_num(x + 26)} V{_num(cy + 6)} H{_num(x)} Z"
        return [_path(rect, fill=key.color, opacity=BAND_OPACITY)]
    if key.kind == "bar":
        return [_path(_bar_d(x + 6, x + 20, cy + 7, cy - 7), fill=key.color)]
    if key.kind == "errorbar":
        return [
            _line(x + 13, cy - 7, x + 13, cy + 7, stroke=INK, width="1.5"),
            _line(x + 9, cy - 7, x + 17, cy - 7, stroke=INK, width="1.5"),
            _line(x + 9, cy + 7, x + 17, cy + 7, stroke=INK, width="1.5"),
        ]
    if key.kind == "interval":
        return [
            _line(x + 2, cy, x + 24, cy, stroke=key.color, width="2"),
            _line(x + 2, cy - 4, x + 2, cy + 4, stroke=key.color, width="2"),
            _line(x + 24, cy - 4, x + 24, cy + 4, stroke=key.color, width="2"),
        ]
    if key.kind == "estimate":
        return [_marker("square", x + 13, cy, key.color, r=4.5)]
    if key.kind == "censor":
        return [
            _line(x, cy, x + 26, cy, stroke=INK_MUTED, width="2"),
            _line(x + 13, cy - 5, x + 13, cy + 5, stroke=INK_SECONDARY, width="1.5"),
        ]
    return [_line(x, cy, x + 26, cy, stroke=INK_SECONDARY, width="1", dash="4 3")]


def _legend(keys: Sequence[_Key], top: float, width: int) -> tuple[list[str], float]:
    """Legend items in rows from the left; returns the elements and the bottom edge."""
    if not keys:
        return [], top
    elements: list[str] = []
    x, row_top = _PAD, top
    for key in keys:
        item = 34.0 + _text_width(key.label, LEGEND_SIZE) + 18.0
        if x > _PAD and x + item > width - _PAD:
            x = _PAD
            row_top += 22.0
        cy = row_top + 11.0
        elements += _legend_key(key, x, cy)
        elements.append(_text(x + 34, cy + 4, key.label, size=LEGEND_SIZE, fill=INK_SECONDARY))
        x += item
    return elements, row_top + 22.0


@dataclass(frozen=True)
class _Frame:
    """Figure-level pieces around the plot area."""

    ident: str
    width: int
    height: int
    header: list[str]
    top: float
    notes: list[str]
    notes_top: float


def _ident(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40].strip("-")
    return f"hl-{slug or 'figure'}"


def _frame(
    *,
    title: str,
    subtitle: str | None,
    synthetic: bool,
    keys: Sequence[_Key],
    note: str | None,
    width: int,
    height: int,
) -> _Frame:
    header = [_text(_PAD, 32, title, size=TITLE_SIZE, weight="600", cls="title")]
    y = 32.0
    if subtitle:
        y += 20.0
        header.append(
            _text(_PAD, y, subtitle, size=SUBTITLE_SIZE, fill=INK_SECONDARY, cls="subtitle")
        )
    if synthetic:
        y += 19.0
        header.append(_text(_PAD, y, SYNTHETIC_LABEL, size=12, weight="600", cls="synthetic"))
    legend, bottom = _legend(keys, y + 10.0, width)
    header.append('<g class="legend">')
    header += legend
    header.append("</g>")
    lines = textwrap.wrap(note or "", width=max(20, int((width - 2 * _PAD) / (NOTE_SIZE * 0.55))))
    notes = []
    baseline = height - 14.0 - 15.0 * (len(lines) - 1)
    notes_top = baseline - NOTE_SIZE - 4.0 if lines else float(height)
    for line in lines:
        notes.append(_text(_PAD, baseline, line, size=NOTE_SIZE, fill=INK_SECONDARY, cls="note"))
        baseline += 15.0
    return _Frame(_ident(title), width, height, header, bottom + 10.0, notes, notes_top)


def _document(
    frame: _Frame,
    title: str,
    desc: str,
    plot: tuple[float, float, float, float],
    body: Sequence[str],
) -> str:
    """Assemble the SVG: root, title, desc, background, clip path, header, body, notes."""
    left, top, right, bottom = plot
    ident = frame.ident
    clip = (
        f'<defs><clipPath id="{ident}-plot"><rect x="{_num(left - 6)}" y="{_num(top - 6)}" '
        f'width="{_num(right - left + 12)}" height="{_num(bottom - top + 12)}"/></clipPath></defs>'
    )
    root = _attrs(
        [
            ("xmlns", "http://www.w3.org/2000/svg"),
            ("width", str(frame.width)),
            ("height", str(frame.height)),
            ("viewBox", f"0 0 {frame.width} {frame.height}"),
            ("role", "img"),
            ("aria-labelledby", f"{ident}-title {ident}-desc"),
            ("font-family", FONT_FAMILY),
        ]
    )
    lines = [
        f"<svg{root}>",
        f'<title id="{ident}-title">{_esc(title)}</title>',
        f'<desc id="{ident}-desc">{_esc(desc)}</desc>',
        f'<rect width="{frame.width}" height="{frame.height}" fill="{SURFACE}"/>',
        clip,
        *frame.header,
        *body,
        *frame.notes,
        "</svg>",
    ]
    return "\n".join(lines) + "\n"


def _describe(kind: str, synthetic: bool, parts: Sequence[str]) -> str:
    prefix = "Synthetic data. " if synthetic else ""
    return prefix + f"{kind}. " + " ".join(parts)


def _common(
    title: str,
    x_label: str,
    y_label: str,
    subtitle: str | None,
    note: str | None,
    desc: str | None,
    width: int,
    height: int,
) -> None:
    _check_text(title, "title")
    _check_text(x_label, "x_label")
    _check_text(y_label, "y_label")
    for value, what in ((subtitle, "subtitle"), (note, "note"), (desc, "desc")):
        if value is not None:
            _check_text(value, what)
    if not title.strip():
        raise PlotError("a figure needs a title")
    if not (isinstance(width, int) and isinstance(height, int)) or width < 320 or height < 240:
        raise PlotError("width and height must be integers of at least 320 and 240")


def _axis_values(
    values: Sequence[float], fixed: tuple[float, float] | None, max_ticks: int = 6
) -> tuple[float, float, list[float]]:
    if fixed is not None:
        lo, hi = _finite(fixed[0], "range start"), _finite(fixed[1], "range end")
        if not lo < hi:
            raise PlotError(f"axis range must increase, got {fixed}")
        span = hi - lo
        ticks = [
            t for t in nice_ticks(lo, hi, max_ticks) if lo - 1e-9 * span <= t <= hi + 1e-9 * span
        ]
        return lo, hi, ticks
    if not values:
        raise PlotError("no finite values to plot")
    ticks = nice_ticks(min(values), max(values), max_ticks)
    return ticks[0], ticks[-1], ticks


def _plot_box(
    frame: _Frame, y_labels: Sequence[str], x_band: float, min_left: float = 0.0
) -> tuple[float, float, float, float]:
    """Plot rectangle (left, top, right, bottom) for charts with a numeric y axis."""
    label_width = max((_text_width(label, TICK_SIZE) for label in y_labels), default=0.0)
    left = max(34.0 + label_width + 10.0, min_left)
    right = frame.width - 24.0
    bottom = frame.notes_top - 8.0 - LABEL_SIZE - 8.0 - x_band
    if right - left < 120 or bottom - frame.top < 80:
        raise PlotError("the figure is too small for its labels; raise width or height")
    return left, frame.top, right, bottom


def _y_axis(
    plot: tuple[float, float, float, float],
    y_scale: _Scale,
    ticks: Sequence[float],
    labels: Sequence[str],
    y_label: str,
) -> tuple[list[str], list[str]]:
    """Horizontal gridlines and y tick labels, plus the rotated y axis label."""
    left, top, right, bottom = plot
    grid = [_line(left, y_scale(t), right, y_scale(t), stroke=GRID, cls="grid") for t in ticks]
    text = [
        _text(left - 8, y_scale(t) + 4, label, size=TICK_SIZE, fill=INK_SECONDARY, anchor="end")
        for t, label in zip(ticks, labels, strict=True)
    ]
    text.append(
        _text(
            22,
            (top + bottom) / 2,
            y_label,
            size=LABEL_SIZE,
            fill=INK_SECONDARY,
            anchor="middle",
            rotate=True,
            cls="axis-label",
        )
    )
    return grid, text


def _x_axis(
    plot: tuple[float, float, float, float],
    x_scale: _Scale,
    ticks: Sequence[float],
    labels: Sequence[str],
    x_label: str,
    gridlines: bool = True,
) -> tuple[list[str], list[str]]:
    """Vertical gridlines, the baseline, tick marks, tick labels, and the x axis label."""
    left, top, right, bottom = plot
    grid = (
        [_line(x_scale(t), top, x_scale(t), bottom, stroke=GRID, cls="grid") for t in ticks]
        if gridlines
        else []
    )
    grid.append(_line(left, bottom, right, bottom, stroke=AXIS, cls="axis"))
    grid += [_line(x_scale(t), bottom, x_scale(t), bottom + 4, stroke=AXIS) for t in ticks]
    text = [
        _text(x_scale(t), bottom + 18, label, size=TICK_SIZE, fill=INK_SECONDARY, anchor="middle")
        for t, label in zip(ticks, labels, strict=True)
    ]
    text.append(
        _text(
            (left + right) / 2,
            bottom + 18 + 8 + LABEL_SIZE,
            x_label,
            size=LABEL_SIZE,
            fill=INK_SECONDARY,
            anchor="middle",
            cls="axis-label",
        )
    )
    return grid, text


def _check_series_count(count: int) -> None:
    if count < 1:
        raise PlotError("a figure needs at least one series")
    if count > MAX_SERIES:
        raise PlotError(
            f"{count} series exceed the {MAX_SERIES} palette slots; fold the tail into "
            "'Other' or split the figure"
        )


def _values(seq: Sequence[object] | None, n: int, what: str) -> list[float | None] | None:
    if seq is None:
        return None
    if len(seq) != n:
        raise PlotError(f"{what} has {len(seq)} values, expected {n}")
    return [_optional(v, f"{what}[{i}]") for i, v in enumerate(seq)]


def _gather(*groups: Sequence[float | None] | None) -> list[float]:
    return [v for group in groups if group is not None for v in group if v is not None]


# --------------------------------------------------------------------------- line chart


def line_chart(
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[Series],
    *,
    desc: str | None = None,
    subtitle: str | None = None,
    note: str | None = None,
    band_label: str = "95% interval",
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    markers: bool | None = None,
    reference_y: float | None = None,
    reference_label: str | None = None,
    width: int = 720,
    height: int = 440,
    synthetic: bool = False,
) -> str:
    """Line chart with optional interval bands, for example the hazard by position.

    Args:
        title, x_label, y_label: Figure title and axis labels.
        series: Up to 8 series; each has its own color, dash pattern, and marker.
        desc: Text for ``<desc>``; a description is generated when omitted.
        subtitle: Second header line.
        note: Footnote (wrapped).
        band_label: Legend text for the shaded bands.
        x_range, y_range: Fixed axis ranges (default: rounded data ranges).
        markers: Draw point markers; default only when every series has at most 40 points.
        reference_y: Draw a dashed horizontal reference line (for example 0.5).
        reference_label: Legend text for the reference line.
        width, height: Pixel size (also the ``viewBox``).
        synthetic: Label the figure as synthetic data.

    Raises:
        PlotError: for bad labels, mismatched lengths, non-finite values, or more
            than 8 series.
    """
    _common(title, x_label, y_label, subtitle, note, desc, width, height)
    _check_series_count(len(series))
    data = []
    for s_index, item in enumerate(series):
        label = _check_text(item.label, f"series[{s_index}].label")
        xs = [_finite(v, f"{label} x[{i}]") for i, v in enumerate(item.x)]
        if not xs:
            raise PlotError(f"series {label!r} has no points")
        ys = _values(item.y, len(xs), f"{label} y")
        if ys is None:
            raise PlotError(f"series {label!r} has no y values")
        lows = _values(item.lower, len(xs), f"{label} lower")
        highs = _values(item.upper, len(xs), f"{label} upper")
        data.append((label, xs, ys, lows, highs))
    show_markers = markers if markers is not None else max(len(d[1]) for d in data) <= 40
    has_band = any(d[3] is not None and d[4] is not None for d in data)
    keys = [
        _Key("line", label, PALETTE[i], DASHES[i], MARKERS[i] if show_markers else None)
        for i, (label, *_) in enumerate(data)
    ]
    if has_band:
        band_color = PALETTE[0] if len(data) == 1 else INK_MUTED
        keys.append(_Key("band", _check_text(band_label, "band_label"), band_color))
    if reference_y is not None:
        reference_y = _finite(reference_y, "reference_y")
        if reference_label is not None:
            keys.append(_Key("reference", _check_text(reference_label, "reference_label")))
    frame = _frame(
        title=title,
        subtitle=subtitle,
        synthetic=synthetic,
        keys=keys,
        note=note,
        width=width,
        height=height,
    )
    x_lo, x_hi, x_ticks = _axis_values(_gather(*(d[1] for d in data)), x_range)
    y_values = _gather(*(g for d in data for g in (d[2], d[3], d[4])))
    if reference_y is not None:
        y_values.append(reference_y)
    y_lo, y_hi, y_ticks = _axis_values(y_values, y_range)
    y_labels = _tick_labels(y_ticks)
    plot = _plot_box(frame, y_labels, 18.0)
    left, top, right, bottom = plot
    sx = _Scale(x_lo, x_hi, left, right)
    sy = _Scale(y_lo, y_hi, bottom, top)
    y_grid, y_text = _y_axis(plot, sy, y_ticks, y_labels, y_label)
    x_grid, x_text = _x_axis(plot, sx, x_ticks, _tick_labels(x_ticks), x_label)
    body = ['<g class="grid">', *y_grid, *x_grid, "</g>"]
    body.append(f'<g class="data" clip-path="url(#{frame.ident}-plot)">')
    if reference_y is not None:
        ref = sy(reference_y)
        body.append(_line(left, ref, right, ref, stroke=INK_SECONDARY, dash="4 3", cls="reference"))
    for i, (_, xs, _, lows, highs) in enumerate(data):
        if lows is not None and highs is not None:
            d = _band(
                [sx(x) for x in xs],
                [None if v is None else sy(v) for v in lows],
                [None if v is None else sy(v) for v in highs],
            )
            if d:
                body.append(_path(d, fill=PALETTE[i], opacity=BAND_OPACITY, cls="band"))
    for i, (_, xs, ys, _, _) in enumerate(data):
        points = [None if y is None else (sx(x), sy(y)) for x, y in zip(xs, ys, strict=True)]
        d = _polyline(points)
        if " L" in d:
            body.append(_path(d, stroke=PALETTE[i], width="2", dash=DASHES[i], cls="series"))
    if show_markers:
        for i, (label, xs, ys, lows, highs) in enumerate(data):
            for j, (x, y) in enumerate(zip(xs, ys, strict=True)):
                if y is None:
                    continue
                tip = f"{label}: {x_label} {_short(x)}, {y_label} {_short(y)}"
                if lows is not None and highs is not None:
                    lo, hi = lows[j], highs[j]
                    if lo is not None and hi is not None:
                        tip += f" ({band_label} {_short(lo)} to {_short(hi)})"
                body.append(_marker(MARKERS[i], sx(x), sy(y), PALETTE[i], tooltip=tip))
    body.append("</g>")
    body += ['<g class="ticks">', *y_text, *x_text, "</g>"]
    summary = [
        f"{y_label} against {x_label}.",
        "Series: " + "; ".join(f"{d[0]} ({len(d[1])} points)" for d in data) + ".",
    ]
    if has_band:
        summary.append(f"Shaded bands: {band_label}.")
    text = desc if desc is not None else _describe("Line chart", synthetic, summary)
    return _document(frame, title, text, plot, body)


# --------------------------------------------------------------------------- step chart


def step_chart(
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[StepSeries],
    *,
    desc: str | None = None,
    subtitle: str | None = None,
    note: str | None = None,
    band_label: str = "95% interval",
    censor_label: str = "Censored",
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = (0.0, 1.0),
    width: int = 720,
    height: int = 440,
    synthetic: bool = False,
) -> str:
    """Step curves with interval bands and censoring ticks (Kaplan-Meier curves).

    Arguments are as for :func:`line_chart`; ``y_range`` defaults to ``(0, 1)``.
    ``censor_label`` is the legend text for censoring ticks, shown when any series
    has them.

    Raises:
        PlotError: for bad labels, unsorted or mismatched steps, or more than 8 series.
    """
    _common(title, x_label, y_label, subtitle, note, desc, width, height)
    _check_series_count(len(series))
    data = []
    for s_index, item in enumerate(series):
        label = _check_text(item.label, f"series[{s_index}].label")
        xs = [_finite(v, f"{label} x[{i}]") for i, v in enumerate(item.x)]
        ys = [_finite(v, f"{label} y[{i}]") for i, v in enumerate(item.y)]
        if len(xs) != len(ys):
            raise PlotError(f"series {label!r} has {len(xs)} x values and {len(ys)} y values")
        if any(b < a for a, b in itertools.pairwise(xs)):
            raise PlotError(f"series {label!r} has x values out of order")
        start = (_finite(item.start[0], "start x"), _finite(item.start[1], "start y"))
        if xs and xs[0] < start[0]:
            raise PlotError(f"series {label!r} has a step before its start")
        lows = _values(item.lower, len(xs), f"{label} lower")
        highs = _values(item.upper, len(xs), f"{label} upper")
        end_x = None if item.end_x is None else _finite(item.end_x, "end_x")
        censored = [_finite(v, f"{label} censored_x[{i}]") for i, v in enumerate(item.censored_x)]
        data.append((label, xs, ys, lows, highs, start, end_x, censored))
    has_band = any(d[3] is not None and d[4] is not None for d in data)
    has_censor = any(d[7] for d in data)
    keys = [_Key("line", d[0], PALETTE[i], DASHES[i]) for i, d in enumerate(data)]
    if has_band:
        band_color = PALETTE[0] if len(data) == 1 else INK_MUTED
        keys.append(_Key("band", _check_text(band_label, "band_label"), band_color))
    if has_censor:
        keys.append(_Key("censor", _check_text(censor_label, "censor_label")))
    frame = _frame(
        title=title,
        subtitle=subtitle,
        synthetic=synthetic,
        keys=keys,
        note=note,
        width=width,
        height=height,
    )
    x_values = [v for d in data for v in (d[5][0], *d[1], *d[7])]
    x_values += [d[6] for d in data if d[6] is not None]
    y_values = [v for d in data for v in (d[5][1], *d[2])] + _gather(
        *(g for d in data for g in (d[3], d[4]))
    )
    x_lo, x_hi, x_ticks = _axis_values(x_values, x_range)
    y_lo, y_hi, y_ticks = _axis_values(y_values, y_range)
    y_labels = _tick_labels(y_ticks)
    plot = _plot_box(frame, y_labels, 18.0)
    left, top, right, bottom = plot
    sx = _Scale(x_lo, x_hi, left, right)
    sy = _Scale(y_lo, y_hi, bottom, top)
    y_grid, y_text = _y_axis(plot, sy, y_ticks, y_labels, y_label)
    x_grid, x_text = _x_axis(plot, sx, x_ticks, _tick_labels(x_ticks), x_label)
    body = ['<g class="grid">', *y_grid, *x_grid, "</g>"]
    body.append(f'<g class="data" clip-path="url(#{frame.ident}-plot)">')
    for i, (_, xs, _, lows, highs, _, end_x, _) in enumerate(data):
        if lows is None or highs is None:
            continue
        rects = []
        for j, x in enumerate(xs):
            nxt = xs[j + 1] if j + 1 < len(xs) else end_x
            lo, hi = lows[j], highs[j]
            if nxt is None or nxt <= x or lo is None or hi is None:
                continue
            rects.append(
                f"M{_num(sx(x))} {_num(sy(lo))} H{_num(sx(nxt))} V{_num(sy(hi))} H{_num(sx(x))} Z"
            )
        if rects:
            body.append(_path(" ".join(rects), fill=PALETTE[i], opacity=BAND_OPACITY, cls="band"))
    for i, (_, xs, ys, _, _, start, end_x, censored) in enumerate(data):
        d = f"M{_num(sx(start[0]))} {_num(sy(start[1]))}"
        for x, y in zip(xs, ys, strict=True):
            d += f" H{_num(sx(x))} V{_num(sy(y))}"
        last_x = xs[-1] if xs else start[0]
        if end_x is not None and end_x > last_x:
            d += f" H{_num(sx(end_x))}"
        body.append(_path(d, stroke=PALETTE[i], width="2", dash=DASHES[i], cls="series"))
        for c in censored:
            value = start[1]
            for x, y in zip(xs, ys, strict=True):
                if x > c:
                    break
                value = y
            body.append(
                _line(
                    sx(c),
                    sy(value) - 5,
                    sx(c),
                    sy(value) + 5,
                    stroke=PALETTE[i],
                    width="1.5",
                    cls="censor",
                )
            )
    body.append("</g>")
    body += ['<g class="ticks">', *y_text, *x_text, "</g>"]
    summary = [
        f"{y_label} against {x_label}.",
        "Series: " + "; ".join(f"{d[0]} ({len(d[1])} steps)" for d in data) + ".",
    ]
    if has_band:
        summary.append(f"Shaded bands: {band_label}.")
    if has_censor:
        summary.append("Ticks mark censored observations.")
    text = desc if desc is not None else _describe("Step chart", synthetic, summary)
    return _document(frame, title, text, plot, body)


# --------------------------------------------------------------------------- bar chart


def _bar_d(x0: float, x1: float, base: float, end: float, radius: float = 4.0) -> str:
    """A bar from ``base`` to ``end`` (pixels) with a rounded data end, square at the base."""
    r = min(radius, (x1 - x0) / 2.0, abs(end - base))
    if end <= base:  # grows upward
        return (
            f"M{_num(x0)} {_num(base)} V{_num(end + r)} Q{_num(x0)} {_num(end)} "
            f"{_num(x0 + r)} {_num(end)} H{_num(x1 - r)} Q{_num(x1)} {_num(end)} "
            f"{_num(x1)} {_num(end + r)} V{_num(base)} Z"
        )
    return (
        f"M{_num(x0)} {_num(base)} V{_num(end - r)} Q{_num(x0)} {_num(end)} "
        f"{_num(x0 + r)} {_num(end)} H{_num(x1 - r)} Q{_num(x1)} {_num(end)} "
        f"{_num(x1)} {_num(end - r)} V{_num(base)} Z"
    )


def bar_chart(
    title: str,
    x_label: str,
    y_label: str,
    categories: Sequence[str],
    series: Sequence[BarSeries],
    *,
    desc: str | None = None,
    subtitle: str | None = None,
    note: str | None = None,
    interval_label: str = "95% interval",
    y_range: tuple[float, float] | None = None,
    width: int = 720,
    height: int = 440,
    synthetic: bool = False,
) -> str:
    """Grouped bars (one group per category) with optional error bars.

    Bars are at most 24 px wide, grow from a baseline at 0 with a rounded data end,
    and sit 2 px apart within a group. Category labels turn 35 degrees when they do
    not fit side by side. A missing value (None) draws no bar.

    Raises:
        PlotError: for bad labels, mismatched lengths, a ``y_range`` without 0, too
            many bars for the width, or more than 8 series.
    """
    _common(title, x_label, y_label, subtitle, note, desc, width, height)
    _check_series_count(len(series))
    cats = [_check_text(c, f"categories[{i}]") for i, c in enumerate(categories)]
    if not cats:
        raise PlotError("a bar chart needs at least one category")
    data = []
    for s_index, item in enumerate(series):
        label = _check_text(item.label, f"series[{s_index}].label")
        values = _values(item.values, len(cats), f"{label} values")
        if values is None:
            raise PlotError(f"series {label!r} has no values")
        lows = _values(item.lower, len(cats), f"{label} lower")
        highs = _values(item.upper, len(cats), f"{label} upper")
        data.append((label, values, lows, highs))
    has_errors = any(d[2] is not None and d[3] is not None for d in data)
    keys = [_Key("bar", d[0], PALETTE[i]) for i, d in enumerate(data)]
    if has_errors:
        keys.append(_Key("errorbar", _check_text(interval_label, "interval_label")))
    frame = _frame(
        title=title,
        subtitle=subtitle,
        synthetic=synthetic,
        keys=keys,
        note=note,
        width=width,
        height=height,
    )
    y_values = [0.0, *_gather(*(g for d in data for g in (d[1], d[2], d[3])))]
    y_lo, y_hi, y_ticks = _axis_values(y_values, y_range)
    if not y_lo <= 0.0 <= y_hi:
        raise PlotError(f"a bar chart's y range must include 0 (bars start at zero), got {y_range}")
    y_labels = _tick_labels(y_ticks)
    label_width = max(_text_width(c, TICK_SIZE) for c in cats)
    provisional = _plot_box(frame, y_labels, 18.0)
    band = (provisional[2] - provisional[0]) / len(cats)
    rotate = label_width > band - 6.0
    x_band = 18.0 if not rotate else label_width * 0.574 + 16.0
    # A turned first label hangs to the left of its bar group; keep it on the canvas.
    min_left = _text_width(cats[0], TICK_SIZE) * 0.819 - band / 2.0 + 8.0 if rotate else 0.0
    plot = _plot_box(frame, y_labels, x_band, min_left)
    left, top, right, bottom = plot
    band = (right - left) / len(cats)
    gap = 2.0
    bar_width = min(24.0, (band * 0.8 - gap * (len(data) - 1)) / len(data))
    if bar_width < 3.0:
        raise PlotError("too many bars for the width; raise width or split the figure")
    group = len(data) * bar_width + (len(data) - 1) * gap
    sy = _Scale(y_lo, y_hi, bottom, top)
    y_grid, y_text = _y_axis(plot, sy, y_ticks, y_labels, y_label)
    base = sy(0.0)
    body = ['<g class="grid">', *y_grid, "</g>"]
    body.append(f'<g class="data" clip-path="url(#{frame.ident}-plot)">')
    errors: list[str] = []
    for c, cat in enumerate(cats):
        x_start = left + c * band + (band - group) / 2.0
        for i, (label, values, lows, highs) in enumerate(data):
            x0 = x_start + i * (bar_width + gap)
            centre = x0 + bar_width / 2.0
            value = values[c]
            if value is not None and value != 0.0:
                tip = f"{label}, {cat}: {_short(value)}"
                if lows is not None and highs is not None:
                    lo, hi = lows[c], highs[c]
                    if lo is not None and hi is not None:
                        tip += f" ({interval_label} {_short(lo)} to {_short(hi)})"
                body.append(
                    _path(
                        _bar_d(x0, x0 + bar_width, base, sy(value)),
                        fill=PALETTE[i],
                        cls="bar",
                        tooltip=tip,
                    )
                )
            if lows is not None and highs is not None:
                lo, hi = lows[c], highs[c]
                if lo is not None and hi is not None:
                    y0, y1 = sy(lo), sy(hi)
                    errors += [
                        _line(centre, y0, centre, y1, stroke=INK, width="1.5", cls="errorbar"),
                        _line(centre - 4, y0, centre + 4, y0, stroke=INK, width="1.5"),
                        _line(centre - 4, y1, centre + 4, y1, stroke=INK, width="1.5"),
                    ]
    body += errors
    body.append("</g>")
    body.append(_line(left, base, right, base, stroke=AXIS, cls="axis"))
    x_text = []
    for c, cat in enumerate(cats):
        cx = left + (c + 0.5) * band
        if rotate:
            x_text.append(
                _element(
                    "text",
                    [
                        ("x", _num(cx)),
                        ("y", _num(bottom + 14)),
                        ("font-size", str(TICK_SIZE)),
                        ("fill", INK_SECONDARY),
                        ("text-anchor", "end"),
                        ("transform", f"rotate(-35 {_num(cx)} {_num(bottom + 14)})"),
                    ],
                    _esc(cat),
                )
            )
        else:
            x_text.append(
                _text(cx, bottom + 18, cat, size=TICK_SIZE, fill=INK_SECONDARY, anchor="middle")
            )
    x_text.append(
        _text(
            (left + right) / 2,
            bottom + x_band + 8 + LABEL_SIZE,
            x_label,
            size=LABEL_SIZE,
            fill=INK_SECONDARY,
            anchor="middle",
            cls="axis-label",
        )
    )
    body += ['<g class="ticks">', *y_text, *x_text, "</g>"]
    shown = ", ".join(cats[:6]) + (", ..." if len(cats) > 6 else "")
    summary = [
        f"{y_label} for {len(cats)} categories of {x_label} ({shown}).",
        "Series: " + "; ".join(d[0] for d in data) + ".",
    ]
    if has_errors:
        summary.append(f"Error bars: {interval_label}.")
    text = desc if desc is not None else _describe("Bar chart", synthetic, summary)
    return _document(frame, title, text, plot, body)


# --------------------------------------------------------------------------- forest plot


def forest_plot(
    title: str,
    x_label: str,
    y_label: str,
    rows: Sequence[ForestRow],
    *,
    desc: str | None = None,
    subtitle: str | None = None,
    note: str | None = None,
    reference: float | Literal["auto"] | None = "auto",
    reference_label: str | None = None,
    estimate_label: str = "Estimate",
    interval_label: str = "95% interval",
    text_header: str | None = None,
    log_x: bool = False,
    x_range: tuple[float, float] | None = None,
    width: int = 720,
    height: int | None = None,
    synthetic: bool = False,
) -> str:
    """Estimates with intervals, one row each, against a dashed reference line.

    The row labels sit in a left column headed by ``y_label``; each row's ``text``
    sits in a right column headed by ``text_header``. Intervals that are open
    (``lo_open``/``hi_open``) or that pass a fixed ``x_range`` run to the edge and end
    in an arrow. With ``log_x`` the axis is logarithmic (for odds ratios) and every
    value must be positive. ``reference`` is ``"auto"`` by default: 0 on a linear
    axis and 1 (no effect on a ratio) on a log axis; None draws no reference line.
    The height grows with the number of rows unless given.

    Raises:
        PlotError: for bad labels, reversed intervals, non-positive values on a log
            axis, or more than 8 groups.
    """
    _common(title, x_label, y_label, subtitle, note, desc, width, height or 480)
    if not rows:
        raise PlotError("a forest plot needs at least one row")
    groups: list[str] = []
    parsed = []
    for r_index, row in enumerate(rows):
        label = _check_text(row.label, f"rows[{r_index}].label")
        text = _check_text(row.text, f"rows[{r_index}].text")
        estimate = _optional(row.estimate, f"{label} estimate")
        lo = _optional(row.lo, f"{label} lo")
        hi = _optional(row.hi, f"{label} hi")
        if lo is not None and hi is not None and lo > hi:
            raise PlotError(f"row {label!r} has lo > hi")
        group = None if row.group is None else _check_text(row.group, f"rows[{r_index}].group")
        if group is not None and group not in groups:
            groups.append(group)
        parsed.append((label, estimate, lo, hi, text, row.lo_open, row.hi_open, group))
    _check_series_count(max(1, len(groups)))
    row_height = 26.0
    keys: list[_Key] = []
    if groups:
        keys += [_Key("interval", g, PALETTE[i]) for i, g in enumerate(groups)]
    else:
        keys.append(_Key("estimate", _check_text(estimate_label, "estimate_label"), PALETTE[0]))
        keys.append(_Key("interval", _check_text(interval_label, "interval_label"), PALETTE[0]))
    if reference == "auto":
        reference = 1.0 if log_x else 0.0
    if reference is not None:
        reference = _finite(reference, "reference")
        if reference_label is not None:
            keys.append(_Key("reference", _check_text(reference_label, "reference_label")))
    has_text = any(p[4] for p in parsed)
    header_text = _check_text(text_header, "text_header") if text_header is not None else None
    # The header and the note block do not depend on the height, so a provisional
    # frame gives the height that fits every row.
    probe = _frame(
        title=title,
        subtitle=subtitle,
        synthetic=synthetic,
        keys=keys,
        note=note,
        width=width,
        height=480,
    )
    note_block = probe.height - probe.notes_top
    auto_height = math.ceil(probe.top + 22.0 + len(parsed) * row_height + 47.0 + note_block)
    final_height = height if height is not None else max(240, auto_height)
    frame = _frame(
        title=title,
        subtitle=subtitle,
        synthetic=synthetic,
        keys=keys,
        note=note,
        width=width,
        height=final_height,
    )
    values = [v for p in parsed for v in (p[1], p[2], p[3]) if v is not None]
    if reference is not None:
        values.append(reference)
    if log_x:
        if any(v <= 0 for v in values):
            raise PlotError("a log axis needs positive estimates, bounds, and reference")
        if x_range is not None:
            x_lo, x_hi = _finite(x_range[0], "range start"), _finite(x_range[1], "range end")
            if not 0 < x_lo < x_hi:
                raise PlotError(f"log axis range must be positive and increasing, got {x_range}")
            ticks = [t for t in log_ticks(x_lo, x_hi) if x_lo <= t <= x_hi]
        else:
            if not values:
                raise PlotError("no finite values to plot")
            ticks = log_ticks(min(values), max(values))
            x_lo, x_hi = ticks[0], ticks[-1]
        tick_labels = [_log_label(t) for t in ticks]
    else:
        x_lo, x_hi, ticks = _axis_values(values, x_range)
        tick_labels = _tick_labels(ticks)
    label_column = max(
        [_text_width(p[0], TICK_SIZE) for p in parsed] + [_text_width(y_label, TICK_SIZE)]
    )
    text_column = 0.0
    if has_text or header_text:
        text_column = (
            max(
                [_text_width(p[4], TICK_SIZE) for p in parsed]
                + [_text_width(header_text or "", TICK_SIZE)]
            )
            + 16.0
        )
    left = _PAD + label_column + 20.0
    right = width - _PAD - text_column - 12.0
    top = frame.top + 22.0
    bottom = top + len(parsed) * row_height
    if right - left < 120:
        raise PlotError("labels leave too little room for the plot; shorten them or widen")
    if bottom + 47.0 > frame.notes_top:
        raise PlotError("the figure is too short for its rows; raise height")
    plot = (left, top, right, bottom)
    sx = _Scale(x_lo, x_hi, left, right, log=log_x)
    x_grid, x_text = _x_axis(plot, sx, ticks, tick_labels, x_label)
    body = ['<g class="grid">', *x_grid, "</g>"]
    columns = [_text(_PAD, top - 8, y_label, size=TICK_SIZE, weight="600", cls="column-header")]
    if header_text:
        columns.append(
            _text(
                right + 16, top - 8, header_text, size=TICK_SIZE, weight="600", cls="column-header"
            )
        )
    body += columns
    body.append(f'<g class="data" clip-path="url(#{frame.ident}-plot)">')
    if reference is not None and x_lo <= reference <= x_hi:
        rx = sx(reference)
        body.append(_line(rx, top, rx, bottom, stroke=INK_SECONDARY, dash="4 3", cls="reference"))
    labels: list[str] = []
    for r_index, (label, estimate, lo, hi, text, lo_open, hi_open, group) in enumerate(parsed):
        cy = top + (r_index + 0.5) * row_height
        color = PALETTE[groups.index(group)] if group is not None else PALETTE[0]
        start = lo if lo is not None else estimate
        end = hi if hi is not None else estimate
        left_arrow = lo_open or (start is not None and start < x_lo)
        right_arrow = hi_open or (end is not None and end > x_hi)
        if lo_open:
            start = x_lo
        if hi_open:
            end = x_hi
        has_interval = lo is not None or hi is not None or lo_open or hi_open
        if has_interval and start is not None and end is not None:
            x0 = sx(min(max(start, x_lo), x_hi))
            x1 = sx(min(max(end, x_lo), x_hi))
            body.append(_line(x0, cy, x1, cy, stroke=color, width="2", cls="interval"))
            if left_arrow:
                body.append(_path(_arrow(x0, cy, -1.0), fill=color, cls="arrow"))
            elif lo is not None:
                body.append(_line(x0, cy - 4, x0, cy + 4, stroke=color, width="2"))
            if right_arrow:
                body.append(_path(_arrow(x1, cy, 1.0), fill=color, cls="arrow"))
            elif hi is not None:
                body.append(_line(x1, cy - 4, x1, cy + 4, stroke=color, width="2"))
        if estimate is not None and x_lo <= estimate <= x_hi:
            tip = f"{label}: {text}" if text else f"{label}: {_short(estimate)}"
            body.append(_marker("square", sx(estimate), cy, color, r=4.5, tooltip=tip))
        labels.append(_text(_PAD, cy + 4, label, size=TICK_SIZE, fill=INK, cls="row-label"))
        if text:
            labels.append(
                _text(right + 16, cy + 4, text, size=TICK_SIZE, fill=INK_SECONDARY, cls="row-text")
            )
    body.append("</g>")
    body += ['<g class="ticks">', *labels, *x_text, "</g>"]
    shown = ", ".join(p[0] for p in parsed[:6]) + (", ..." if len(parsed) > 6 else "")
    summary = [f"{x_label} for {len(parsed)} rows of {y_label} ({shown})."]
    if reference is not None:
        summary.append(f"Dashed reference line at {_short(reference)}.")
    if log_x:
        summary.append("Logarithmic x axis.")
    text_desc = desc if desc is not None else _describe("Forest plot", synthetic, summary)
    return _document(frame, title, text_desc, plot, body)


def _arrow(x: float, y: float, direction: float) -> str:
    """A small arrowhead at ``x`` pointing left (-1) or right (+1)."""
    tip = x + direction * 2.0
    back = x - direction * 6.0
    return f"M{_num(tip)} {_num(y)} L{_num(back)} {_num(y - 4.5)} L{_num(back)} {_num(y + 4.5)} Z"


# --------------------------------------------------------------------------- output


def write_svg(path: str | Path, svg: str) -> Path:
    """Write SVG text to ``path`` (UTF-8, ``\\n`` line ends), creating parent folders.

    Raises:
        PlotError: when ``svg`` is not an SVG document.
    """
    if (
        not isinstance(svg, str)
        or not svg.startswith("<svg")
        or not svg.rstrip().endswith("</svg>")
    ):
        raise PlotError("not an SVG document")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(svg, encoding="utf-8", newline="\n")
    return target


__all__ = [
    "AXIS",
    "BAND_OPACITY",
    "DASHES",
    "FONT_FAMILY",
    "GRID",
    "INK",
    "INK_MUTED",
    "INK_SECONDARY",
    "MARKERS",
    "MAX_SERIES",
    "PALETTE",
    "SURFACE",
    "SYNTHETIC_LABEL",
    "BarSeries",
    "ForestRow",
    "PlotError",
    "Series",
    "StepSeries",
    "bar_chart",
    "forest_plot",
    "line_chart",
    "log_ticks",
    "nice_ticks",
    "step_chart",
    "write_svg",
]
