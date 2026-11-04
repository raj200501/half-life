"""Tests for :mod:`halflife.svgplot`: deterministic, accessible SVG figures.

The figures here are drawn from small synthetic inputs made up for the tests (and
labeled synthetic on their face); they are not results.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from halflife import stats, svgplot
from halflife.svgplot import (
    DASHES,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    MAX_SERIES,
    PALETTE,
    SYNTHETIC_LABEL,
    BarSeries,
    ForestRow,
    PlotError,
    Series,
    StepSeries,
    bar_chart,
    forest_plot,
    line_chart,
    log_ticks,
    nice_ticks,
    step_chart,
    write_svg,
)

NS = "{http://www.w3.org/2000/svg}"
COORD = re.compile(r"^-?\d+\.\d{2}$")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
COORD_ATTRS = ("x", "y", "x1", "y1", "x2", "y2", "cx", "cy", "r")
TEXT_COLORS = {INK, INK_SECONDARY, INK_MUTED}


# --------------------------------------------------------------------------- example figures


def _line_svg() -> str:
    xs = [0, 50, 100, 150]
    return line_chart(
        "Hazard by position",
        "Tool call index",
        "Predicted hazard",
        [
            Series(
                "family A",
                xs,
                [0.1, 0.2, 0.35, 0.5],
                [0.05, 0.15, 0.3, 0.4],
                [0.15, 0.25, 0.4, 0.6],
            ),
            Series("family B", xs, [0.05, None, 0.1, 0.15]),
        ],
        reference_y=0.5,
        reference_label="Hazard 0.5",
        note="Synthetic test figure. Bands are 95% intervals.",
        synthetic=True,
    )


def _step_svg() -> str:
    km = stats.kaplan_meier([1, 2, 2, 3, 4, 4], [True, True, False, True, False, True])
    return step_chart(
        "Time to first violation",
        "Tool call index",
        "Share without a violation",
        [
            StepSeries(
                "family A",
                km.times,
                km.survival,
                km.lower,
                km.upper,
                end_x=6.0,
                censored_x=km.censored_times,
            ),
            StepSeries("family B", [2.0, 5.0], [0.75, 0.5], end_x=6.0),
        ],
        subtitle="Kaplan-Meier, descriptive only",
        synthetic=True,
    )


def _bar_svg() -> str:
    return bar_chart(
        "Violation rate by constraint type",
        "Constraint type",
        "Violations per opportunity",
        ["prohibition", "invariant", "procedure"],
        [
            BarSeries("family A", [0.2, 0.35, 0.1], [0.1, 0.25, 0.05], [0.3, 0.45, 0.2]),
            BarSeries("family B", [0.15, None, 0.0]),
        ],
        y_range=(0.0, 1.0),
        synthetic=True,
    )


def _forest_svg() -> str:
    return forest_plot(
        "Half-life by family",
        "Half-life (tool calls)",
        "Family",
        [
            ForestRow("family A", 37.2, 20.1, None, "37.2 [20.1, >150]", hi_open=True),
            ForestRow("family B", 90.0, 60.5, 140.2, "90 [60.5, 140.2]"),
            ForestRow("family C", None, 80.0, None, ">150 [80, >150]", hi_open=True),
        ],
        x_range=(0.0, 150.0),
        reference=None,
        text_header="Half-life [95% interval]",
        synthetic=True,
    )


FIGURES = {"line": _line_svg, "step": _step_svg, "bar": _bar_svg, "forest": _forest_svg}

# SHA-256 of each example figure. These pin the byte-exact output: every coordinate
# comes from IEEE arithmetic and fixed two-decimal formatting, so the bytes are the
# same on every platform. A deliberate change to the drawing code updates them.
GOLDEN = {
    "line": "73b027c462b72a7504898ee257034279e1b212ab8dd27fd75c753e1e7af02f59",
    "step": "272d73983b64ebc429d70ba6eb49aad3389ae98d6da35ddf035d730e8865dcf9",
    "bar": "f3a625ab295b66ccb00922e426d8fc57073168643a0d4f0b97b99b8fcce4c9f3",
    "forest": "ff53f31e69ad8ff6461bfa0673426865ed884681eddfabdb8ad0061ea5a63be9",
}


def _parse(svg: str) -> ET.Element:
    return ET.fromstring(svg)


def _all(root: ET.Element, tag: str) -> list[ET.Element]:
    return list(root.iter(f"{NS}{tag}"))


def _group(root: ET.Element, cls: str) -> ET.Element:
    for element in root.iter(f"{NS}g"):
        if element.get("class") == cls:
            return element
    raise AssertionError(f"no <g class={cls!r}>")


def _texts(element: ET.Element) -> list[str]:
    return [t.text or "" for t in element.iter(f"{NS}text")]


# --------------------------------------------------------------------------- shared structure


@pytest.mark.parametrize("name", sorted(FIGURES))
def test_figures_have_title_desc_viewbox_and_aria(name: str) -> None:
    root = _parse(FIGURES[name]())
    assert root.tag == f"{NS}svg"
    width, height = root.get("width"), root.get("height")
    assert root.get("viewBox") == f"0 0 {width} {height}"
    assert root.get("role") == "img"
    children = list(root)
    assert children[0].tag == f"{NS}title" and children[1].tag == f"{NS}desc"
    ids = (root.get("aria-labelledby") or "").split()
    assert ids == [children[0].get("id"), children[1].get("id")]
    assert children[1].text and children[1].text.startswith("Synthetic data.")
    assert children[2].tag == f"{NS}rect" and children[2].get("fill") == svgplot.SURFACE


@pytest.mark.parametrize("name", sorted(FIGURES))
def test_figures_use_fixed_formatting_readable_text_and_ink_colors(name: str) -> None:
    svg = FIGURES[name]()
    root = _parse(svg)
    for element in root.iter():
        for attr in COORD_ATTRS:
            value = element.get(attr)
            if value is not None:
                assert COORD.match(value), (element.tag, attr, value)
        d = element.get("d")
        if d is not None:
            assert all(re.match(r"^-?\d+\.\d{2}$", n) for n in NUMBER.findall(d)), d
    for text in _all(root, "text"):
        assert int(text.get("font-size", "0")) >= 11
        assert text.get("fill") in TEXT_COLORS  # text wears ink, never a series color
    assert SYNTHETIC_LABEL in _texts(root)
    assert "\u2014" not in svg and "-0.00" not in svg
    assert svg.endswith("</svg>\n") and "\r" not in svg


@pytest.mark.parametrize("name", sorted(FIGURES))
def test_figures_are_deterministic_and_match_golden_bytes(name: str) -> None:
    first = FIGURES[name]()
    assert FIGURES[name]() == first
    assert hashlib.sha256(first.encode("utf-8")).hexdigest() == GOLDEN[name]


def test_figures_are_identical_across_processes_and_hash_seeds(tmp_path: Path) -> None:
    script = (
        "import hashlib, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import test_svgplot as t\n"
        "for name in sorted(t.FIGURES):\n"
        "    print(name, hashlib.sha256(t.FIGURES[name]().encode()).hexdigest())\n"
    )
    expected = "".join(
        f"{name} {hashlib.sha256(FIGURES[name]().encode()).hexdigest()}\n"
        for name in sorted(FIGURES)
    )
    tests_dir = str(Path(__file__).resolve().parent)
    for seed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, "-c", script, tests_dir],
            capture_output=True,
            text=True,
            env=env,
            check=True,
            timeout=120,
        )
        assert out.stdout == expected


# --------------------------------------------------------------------------- line chart


def test_line_chart_series_order_legend_ticks_and_reference() -> None:
    root = _parse(_line_svg())
    series = [p for p in _all(root, "path") if p.get("class") == "series"]
    assert [p.get("stroke") for p in series] == [PALETTE[0], PALETTE[1]]
    assert series[0].get("stroke-dasharray") is None
    assert series[1].get("stroke-dasharray") == DASHES[1]
    # family B has a gap at x = 50, so its path starts a new subpath there.
    assert (series[1].get("d") or "").count("M") == 2
    bands = [p for p in _all(root, "path") if p.get("class") == "band"]
    assert len(bands) == 1 and bands[0].get("fill") == PALETTE[0]
    assert bands[0].get("fill-opacity") == svgplot.BAND_OPACITY
    legend = _texts(_group(root, "legend"))
    assert legend == ["family A", "family B", "95% interval", "Hazard 0.5"]
    ticks = _texts(_group(root, "ticks"))
    for label in ("0", "50", "100", "150", "0.0", "0.2", "0.4", "0.6"):
        assert label in ticks
    assert "Tool call index" in ticks and "Predicted hazard" in ticks
    reference = [line for line in _all(root, "line") if line.get("class") == "reference"]
    assert len(reference) == 1 and reference[0].get("stroke-dasharray") == "4 3"
    markers = [m for m in _group(root, "data").iter() if m.get("class") == "marker"]
    assert len(markers) == 7  # 4 points of A and 3 of B (one is missing)
    legend_markers = [m for m in _group(root, "legend").iter() if m.get("class") == "marker"]
    assert [m.tag for m in legend_markers] == [f"{NS}circle", f"{NS}path"]  # circle, square
    tooltip = markers[0].find(f"{NS}title")
    assert tooltip is not None and tooltip.text == (
        "family A: Tool call index 0, Predicted hazard 0.1 (95% interval 0.05 to 0.15)"
    )


def test_line_chart_options() -> None:
    xs = list(range(60))
    svg = line_chart(
        "Many points",
        "x",
        "y",
        [Series("only", xs, [float(x) for x in xs])],
        desc="A described chart.",
        subtitle="Sub",
        x_range=(0, 100),
        y_range=(0, 80),
    )
    root = _parse(svg)
    assert not [m for m in root.iter() if m.get("class") == "marker"]  # over 40 points
    assert list(root)[1].text == "A described chart."
    assert "Sub" in _texts(root) and SYNTHETIC_LABEL not in _texts(root)
    assert "100" in _texts(_group(root, "ticks"))
    with_markers = line_chart("Few", "x", "y", [Series("s", [0, 1], [0, 1])], markers=False)
    assert not [m for m in _parse(with_markers).iter() if m.get("class") == "marker"]
    nan_point = line_chart("Gap", "x", "y", [Series("s", [0, 1, 2], [0.0, float("nan"), 1.0])])
    assert "Line chart." in (list(_parse(nan_point))[1].text or "")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"title": "  "},
        {"title": "Dash \u2014 here"},
        {"x_label": "bad\x07label"},
        {"y_label": 3},
        {"note": "a \u2014 b"},
        {"width": 200},
        {"height": 100},
        {"width": 720.0},
        {"series": []},
        {"series": [Series(f"s{i}", [0, 1], [0, 1]) for i in range(MAX_SERIES + 1)]},
        {"series": [Series("s", [], [])]},
        {"series": [Series("s", [0, 1], [0])]},
        {"series": [Series("s", [0, float("inf")], [0, 1])]},
        {"series": [Series("s", [0, 1], [0, "1"])]},  # type: ignore[list-item]  # invalid
        {"series": [Series("s", [0, 1], [0, 1], [0], [1, 2])]},
        {"series": [Series("bad \u2014 label", [0, 1], [0, 1])]},
        {"x_range": (5, 5)},
        {"band_label": "a \u2014 b", "series": [Series("s", [0, 1], [0, 1], [0, 0], [1, 1])]},
        {"reference_y": float("nan")},
    ],
)
def test_line_chart_rejects_bad_input(kwargs: dict[str, object]) -> None:
    args: dict[str, object] = {
        "title": "T",
        "x_label": "x",
        "y_label": "y",
        "series": [Series("s", [0, 1], [0, 1])],
    }
    args.update(kwargs)
    with pytest.raises(PlotError):
        line_chart(**args)  # type: ignore[arg-type]  # mixed argument types on purpose


def test_line_chart_bands_with_gaps_and_missing_values() -> None:
    svg = line_chart(
        "Gaps",
        "x",
        "y",
        [
            Series(
                "s",
                [0, 1, 2, 3, 4],
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.0, None, 0.2, 0.3, 0.4],
                [0.2, 0.3, None, 0.5, 0.6],
            )
        ],
        reference_y=0.5,
    )
    root = _parse(svg)
    bands = [p for p in _all(root, "path") if p.get("class") == "band"]
    # Bounds exist together only at x = 0 (a lone point, no area) and x = 3..4.
    assert len(bands) == 1 and (bands[0].get("d") or "").count("Z") == 1
    assert _texts(_group(root, "legend")) == ["s", "95% interval"]  # reference has no label
    markers = [m for m in _group(root, "data").iter() if m.get("class") == "marker"]
    tip = markers[1].find(f"{NS}title")
    assert tip is not None and tip.text == "s: x 1, y 0.2"
    with pytest.raises(PlotError, match="no finite values"):
        line_chart("T", "x", "y", [Series("s", [0, 1], [None, None])])
    with pytest.raises(PlotError, match="no y values"):
        line_chart("T", "x", "y", [Series("s", [0, 1], None)])  # type: ignore[arg-type]  # invalid


def test_line_chart_too_small_for_its_labels() -> None:
    long = "a very long legend label that wraps " * 6
    with pytest.raises(PlotError, match="too small"):
        line_chart(
            "T",
            "x",
            "y",
            [Series(f"{long} {i}", [0, 1], [0, 1]) for i in range(8)],
            width=320,
            height=240,
        )


# --------------------------------------------------------------------------- step chart


def test_step_chart_draws_steps_bands_and_censoring() -> None:
    root = _parse(_step_svg())
    series = [p for p in _all(root, "path") if p.get("class") == "series"]
    assert len(series) == 2
    d = series[0].get("d") or ""
    assert d.startswith("M") and d.count(" H") == 5 and d.count(" V") == 4  # 4 steps + end
    bands = [p for p in _all(root, "path") if p.get("class") == "band"]
    assert len(bands) == 1 and (bands[0].get("d") or "").count("Z") == 4
    censor = [line for line in _all(root, "line") if line.get("class") == "censor"]
    assert len(censor) == 2 and {c.get("stroke") for c in censor} == {PALETTE[0]}
    assert "Censored" in _texts(_group(root, "legend"))
    assert "Ticks mark censored observations." in (list(root)[1].text or "")


def test_step_chart_censor_tick_sits_on_the_curve() -> None:
    svg = step_chart(
        "Censor",
        "t",
        "S",
        [StepSeries("s", [1.0, 3.0], [0.5, 0.25], censored_x=[2.0])],
        x_range=(0, 4),
    )
    root = _parse(svg)
    tick = next(line for line in _all(root, "line") if line.get("class") == "censor")
    series = next(p for p in _all(root, "path") if p.get("class") == "series")
    # The path holds the y pixel of each step; the tick at t = 2 is centered on S = 0.5.
    y_half = float(re.findall(r"V(-?\d+\.\d{2})", series.get("d") or "")[0])
    assert float(tick.get("y1", "0")) == pytest.approx(y_half - 5, abs=0.011)
    assert float(tick.get("y2", "0")) == pytest.approx(y_half + 5, abs=0.011)


def test_step_chart_band_gaps_without_censoring() -> None:
    svg = step_chart(
        "Plain",
        "t",
        "S",
        [StepSeries("s", [1.0, 2.0, 3.0], [0.8, 0.6, 0.4], [0.6, None, 0.2], [0.9, 0.8, 0.6])],
    )
    root = _parse(svg)
    bands = [p for p in _all(root, "path") if p.get("class") == "band"]
    # Step 1 has a band to t = 2; step 2 lacks a bound; step 3 has no next step or end_x.
    assert len(bands) == 1 and (bands[0].get("d") or "").count("Z") == 1
    assert "Censored" not in _texts(_group(root, "legend"))
    assert "censored" not in (list(root)[1].text or "")
    empty = step_chart("Flat", "t", "S", [StepSeries("s", [], [], end_x=5.0)])
    series = next(p for p in _all(_parse(empty), "path") if p.get("class") == "series")
    assert (series.get("d") or "").count("H") == 1  # the start value carried to end_x


@pytest.mark.parametrize(
    "item",
    [
        StepSeries("s", [2.0, 1.0], [0.5, 0.2]),
        StepSeries("s", [1.0], [0.5, 0.2]),
        StepSeries("s", [-1.0], [0.5]),
        StepSeries("s", [1.0], [0.5], lower=[0.1, 0.2]),
        StepSeries("s", [1.0], [float("nan")]),
    ],
)
def test_step_chart_rejects_bad_steps(item: StepSeries) -> None:
    with pytest.raises(PlotError):
        step_chart("T", "x", "y", [item])


# --------------------------------------------------------------------------- bar chart


def test_bar_chart_bars_error_bars_and_missing_values() -> None:
    root = _parse(_bar_svg())
    bars = [p for p in _all(root, "path") if p.get("class") == "bar"]
    # family A has 3 bars; family B has one (None draws nothing, 0 draws nothing).
    assert [b.get("fill") for b in bars] == [PALETTE[0], PALETTE[1], PALETTE[0], PALETTE[0]]
    errors = [line for line in _all(root, "line") if line.get("class") == "errorbar"]
    assert len(errors) == 3
    tooltip = bars[0].find(f"{NS}title")
    assert (
        tooltip is not None
        and tooltip.text == "family A, prohibition: 0.2 (95% interval 0.1 to 0.3)"
    )
    assert _texts(_group(root, "legend")) == ["family A", "family B", "95% interval"]
    assert not any((t.get("transform") or "").startswith("rotate(-35") for t in _all(root, "text"))


def test_bar_chart_rotates_long_labels_and_handles_negative_values() -> None:
    svg = bar_chart(
        "Rotated",
        "Arm",
        "Change",
        ["constraint_pinned_summary", "reinjection_after_compaction", "policy_guard", "none"],
        [BarSeries("change", [0.1, -0.2, 0.05, 0.4])],
    )
    root = _parse(svg)
    rotated = [t for t in _all(root, "text") if (t.get("transform") or "").startswith("rotate(-35")]
    assert len(rotated) == 4
    assert all(float(t.get("x", "0")) > 0 for t in rotated)
    bars = [p for p in _all(root, "path") if p.get("class") == "bar"]
    assert len(bars) == 4
    assert "-0.2" in _texts(_group(root, "ticks"))
    summary = list(root)[1].text or ""
    assert summary.startswith("Bar chart.") and "4 categories" in summary


def test_bar_chart_partial_error_bars() -> None:
    svg = bar_chart(
        "Partial",
        "x",
        "y",
        ["a", "b"],
        [BarSeries("s", [0.4, 0.6], [0.3, None], [0.5, 0.7])],
    )
    root = _parse(svg)
    assert len([line for line in _all(root, "line") if line.get("class") == "errorbar"]) == 1
    bars = [p for p in _all(root, "path") if p.get("class") == "bar"]
    tip = bars[1].find(f"{NS}title")
    assert tip is not None and tip.text == "s, b: 0.6"
    with pytest.raises(PlotError, match="no values"):
        bar_chart("T", "x", "y", ["a"], [BarSeries("s", None)])  # type: ignore[arg-type]  # invalid


def test_bar_chart_many_categories_are_summarized() -> None:
    cats = [f"c{i}" for i in range(8)]
    svg = bar_chart("Many", "x", "y", cats, [BarSeries("s", [1.0] * 8)], width=900)
    assert "c0, c1, c2, c3, c4, c5, ..." in (list(_parse(svg))[1].text or "")


@pytest.mark.parametrize(
    ("categories", "series", "kwargs"),
    [
        ([], [BarSeries("s", [])], {}),
        (["a"], [BarSeries("s", [1.0, 2.0])], {}),
        (["a"], [BarSeries("s", [1.0], [0.0, 1.0])], {}),
        (["a", "b \u2014 c"], [BarSeries("s", [1.0, 2.0])], {}),
        (["a"], [BarSeries("s", [1.0])], {"y_range": (0.5, 2.0)}),
        ([f"c{i}" for i in range(60)], [BarSeries(f"s{i}", [1.0] * 60) for i in range(8)], {}),
    ],
)
def test_bar_chart_rejects_bad_input(
    categories: list[str], series: list[BarSeries], kwargs: dict[str, object]
) -> None:
    with pytest.raises(PlotError):
        bar_chart("T", "x", "y", categories, series, **kwargs)  # type: ignore[arg-type]  # kwargs


# --------------------------------------------------------------------------- forest plot


def test_forest_plot_open_intervals_text_column_and_missing_estimate() -> None:
    root = _parse(_forest_svg())
    arrows = [p for p in _all(root, "path") if p.get("class") == "arrow"]
    assert len(arrows) == 2  # the two open upper bounds
    markers = [m for m in _group(root, "data").iter() if m.get("class") == "marker"]
    assert len(markers) == 2  # family C has no point estimate
    ticks = _group(root, "ticks")
    assert [t.text for t in ticks.iter(f"{NS}text") if t.get("class") == "row-text"] == [
        "37.2 [20.1, >150]",
        "90 [60.5, 140.2]",
        ">150 [80, >150]",
    ]
    headers = [t.text for t in _all(root, "text") if t.get("class") == "column-header"]
    assert headers == ["Family", "Half-life [95% interval]"]
    assert not [line for line in _all(root, "line") if line.get("class") == "reference"]
    assert _texts(_group(root, "legend")) == ["Estimate", "95% interval"]


def test_forest_plot_groups_log_axis_and_reference() -> None:
    rows = [
        ForestRow("pin vs none", 0.5, 0.3, 0.9, group="E4"),
        ForestRow("guard vs none", 0.1, 0.02, 0.6, group="E4"),
        ForestRow("system vs first_user", 1.2, 0.7, 2.1, group="E3"),
    ]
    svg = forest_plot(
        "Odds ratios",
        "Odds ratio (log scale)",
        "Contrast",
        rows,
        reference=1.0,
        reference_label="No effect",
        log_x=True,
    )
    root = _parse(svg)
    assert _texts(_group(root, "legend")) == ["E4", "E3", "No effect"]
    intervals = [line for line in _all(root, "line") if line.get("class") == "interval"]
    assert [i.get("stroke") for i in intervals] == [PALETTE[0], PALETTE[0], PALETTE[1]]
    ticks = _texts(_group(root, "ticks"))
    for label in ("0.02", "0.1", "1", "2", "5"):
        assert label in ticks
    reference = [line for line in _all(root, "line") if line.get("class") == "reference"]
    assert len(reference) == 1
    assert "Logarithmic x axis." in (list(root)[1].text or "")


def test_forest_plot_fixed_range_clips_with_arrows_and_grows_with_rows() -> None:
    rows = [
        ForestRow(f"row {i}", 0.5 * i, 0.5 * i - 3, 0.5 * i + 3, lo_open=i == 0) for i in range(6)
    ]
    svg = forest_plot("Clipped", "x", "Rows", rows, x_range=(0.0, 2.0), reference=5.0)
    root = _parse(svg)
    arrows = [p for p in _all(root, "path") if p.get("class") == "arrow"]
    assert len(arrows) == 12  # every interval passes both ends of the range
    assert not [line for line in _all(root, "line") if line.get("class") == "reference"]
    short = forest_plot("Short", "x", "Rows", rows[:1])
    tall = forest_plot("Tall", "x", "Rows", rows)
    assert int(_parse(tall).get("height", "0")) > int(_parse(short).get("height", "0"))
    log_fixed = forest_plot(
        "Log fixed", "OR", "Rows", [ForestRow("a", 1.0, 0.5, 2.0)], log_x=True, x_range=(0.1, 10.0)
    )
    assert "10" in _texts(_group(_parse(log_fixed), "ticks"))
    # The default reference is 1 (no effect) on a log axis and 0 on a linear one.
    assert "Dashed reference line at 1." in (list(_parse(log_fixed))[1].text or "")
    assert "Dashed reference line at 0." in (list(_parse(short))[1].text or "")


def test_forest_plot_rows_without_intervals_or_values() -> None:
    rows = [
        ForestRow("point only", 2.0),
        ForestRow("not estimable", None, text="no events"),
        ForestRow("upper only", 1.0, None, 3.0),
    ]
    root = _parse(forest_plot("Sparse", "x", "Rows", rows, x_range=(0.0, 4.0)))
    data = _group(root, "data")
    markers = [m for m in data.iter() if m.get("class") == "marker"]
    assert len(markers) == 2
    intervals = [line for line in data.iter(f"{NS}line") if line.get("class") == "interval"]
    assert len(intervals) == 1  # a point estimate alone draws no interval line
    caps = [line for line in data.iter(f"{NS}line") if line.get("class") is None]
    assert len(caps) == 1  # only the upper bound of the last row has a cap
    assert "no events" in _texts(_group(root, "ticks"))


@pytest.mark.parametrize(
    ("rows", "kwargs"),
    [
        ([], {}),
        ([ForestRow("a", 1.0, 2.0, 0.5)], {}),
        ([ForestRow("a", 1.0, 0.0, 2.0)], {"log_x": True, "reference": 1.0}),
        ([ForestRow("a", 1.0, 0.5, 2.0)], {"log_x": True, "reference": 1.0, "x_range": (0, 2)}),
        ([ForestRow("a", None)], {"log_x": True, "reference": None}),
        ([ForestRow("a", 1.0, text="x \u2014 y")], {}),
        ([ForestRow("a", 1.0, group=f"g{i}") for i in range(MAX_SERIES + 1)], {}),
        ([ForestRow("a", 1.0)] * 30, {"height": 240}),
        ([ForestRow("a" * 120, 1.0)], {}),
        ([ForestRow("a", float("inf"))], {}),
    ],
)
def test_forest_plot_rejects_bad_input(rows: list[ForestRow], kwargs: dict[str, object]) -> None:
    with pytest.raises(PlotError):
        forest_plot("T", "x", "y", rows, **kwargs)  # type: ignore[arg-type]  # kwargs on purpose


# --------------------------------------------------------------------------- ticks and output


def test_nice_ticks_known_values() -> None:
    # Worked by hand: the raw step is span / (max_ticks - 1), rounded up to 1, 2, 2.5,
    # or 5 times a power of ten until the ticks covering the span fit.
    assert nice_ticks(0, 1) == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert nice_ticks(0, 150) == [0.0, 50.0, 100.0, 150.0]
    assert nice_ticks(-1, 1) == [-1.0, -0.5, 0.0, 0.5, 1.0]
    # Equal bounds widen by 10%: 0.27 to 0.33, raw step 0.012, step 0.02.
    assert nice_ticks(0.3, 0.3) == [0.26, 0.28, 0.3, 0.32, 0.34]
    assert nice_ticks(0, 0) == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert nice_ticks(0.1, 0.7, 4) == [0.0, 0.25, 0.5, 0.75]
    # max_ticks is a target: when no multiple fits, the widest step (10 times) is used.
    assert nice_ticks(0.95, 1.05, 2) == [0.0, 1.0, 2.0]
    with pytest.raises(PlotError, match="max_ticks"):
        nice_ticks(0, 1, 1)
    with pytest.raises(PlotError, match="reversed"):
        nice_ticks(1, 0)
    with pytest.raises(PlotError, match="finite"):
        nice_ticks(0, float("nan"))


def test_log_ticks_known_values() -> None:
    assert log_ticks(0.5, 4) == [0.5, 1.0, 2.0, 5.0]
    assert log_ticks(0.02, 0.6) == [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    assert log_ticks(1.0, 1.0) == [0.5, 1.0, 2.0]
    assert log_ticks(0.01, 1000) == [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    assert log_ticks(3e-14, 1.0)[0] == pytest.approx(2e-14)  # tiny odds ratios keep their scale
    for lo, hi in ((0.0, 1.0), (-1.0, 1.0), (2.0, 1.0)):
        with pytest.raises(PlotError):
            log_ticks(lo, hi)


def test_write_svg_writes_utf8_with_parent_folders(tmp_path: Path) -> None:
    svg = _line_svg()
    target = write_svg(tmp_path / "figures" / "nested" / "hazard.svg", svg)
    assert target.read_bytes() == svg.encode("utf-8")
    again = write_svg(str(tmp_path / "figures" / "nested" / "hazard.svg"), svg)
    assert again == target and target.read_bytes() == svg.encode("utf-8")
    for bad in ("<html></html>", "<svg>", 42):
        with pytest.raises(PlotError):
            write_svg(tmp_path / "bad.svg", bad)  # type: ignore[arg-type]  # invalid on purpose


def test_ids_derive_from_the_title() -> None:
    root = _parse(line_chart("Hazard: by position!", "x", "y", [Series("s", [0, 1], [0, 1])]))
    assert root[0].get("id") == "hl-hazard-by-position-title"
    symbols = _parse(line_chart("%%%", "x", "y", [Series("s", [0, 1], [0, 1])]))
    assert symbols[0].get("id") == "hl-figure-title"


def test_public_names_exist() -> None:
    for name in svgplot.__all__:
        assert hasattr(svgplot, name), name
    assert len(PALETTE) == len(DASHES) == len(svgplot.MARKERS) == MAX_SERIES == 8
