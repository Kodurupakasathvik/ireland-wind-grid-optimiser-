"""Generate the five release figures from committed CSV result tables.

This module uses only the Python standard library so it can be run immediately
after cloning the repository. The figures are SVG files, making their labels
and values inspectable and version-control friendly.

Run from the repository root:
    python scripts/release/generate_release_figures.py
"""

from __future__ import annotations

import csv
import math
from html import escape
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"

WIDTH = 1120
HEIGHT = 640
MARGIN_LEFT = 105
MARGIN_RIGHT = 42
MARGIN_TOP = 82
MARGIN_BOTTOM = 115
PLOT_WIDTH = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
PLOT_HEIGHT = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

INK = "#17212b"
MUTED = "#556372"
GRID = "#cbd5df"
BLUE = "#2c6eaf"
TEAL = "#138a72"
ORANGE = "#d97706"
RED = "#c43d4f"
PURPLE = "#7057a8"


def read_csv(name: str) -> list[dict[str, str]]:
    with (PROCESSED / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], column: str) -> float | None:
    value = row.get(column, "").strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def svg_document(title: str, content: Iterable[str]) -> str:
    body = "\n".join(content)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title>
<desc id="desc">Research-release figure generated from the project's committed CSV results.</desc>
<rect width="100%" height="100%" fill="#ffffff"/>
<style>
text {{ font-family: Arial, Helvetica, sans-serif; fill: {INK}; }}
.title {{ font-size: 25px; font-weight: 700; }}
.subtitle {{ font-size: 13px; fill: {MUTED}; }}
.axis {{ font-size: 12px; fill: {MUTED}; }}
.label {{ font-size: 12px; }}
.small {{ font-size: 11px; fill: {MUTED}; }}
.value {{ font-size: 11px; font-weight: 700; }}
</style>
{body}
</svg>'''


def save(name: str, title: str, content: Iterable[str]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    path.write_text(svg_document(title, content), encoding="utf-8")
    print(f"Wrote {path.relative_to(ROOT)}")


def scale(value: float, minimum: float, maximum: float, start: float, end: float) -> float:
    if maximum <= minimum:
        return (start + end) / 2
    return start + (value - minimum) / (maximum - minimum) * (end - start)


def chart_base(title: str, subtitle: str, y_label: str, maximum: float, *, top: int = MARGIN_TOP, height: int = PLOT_HEIGHT) -> list[str]:
    bottom = top + height
    result = [
        f'<text x="{MARGIN_LEFT}" y="36" class="title">{escape(title)}</text>',
        f'<text x="{MARGIN_LEFT}" y="58" class="subtitle">{escape(subtitle)}</text>',
        f'<line x1="{MARGIN_LEFT}" y1="{bottom}" x2="{WIDTH - MARGIN_RIGHT}" y2="{bottom}" stroke="{INK}" stroke-width="1"/>',
        f'<line x1="{MARGIN_LEFT}" y1="{top}" x2="{MARGIN_LEFT}" y2="{bottom}" stroke="{INK}" stroke-width="1"/>',
        f'<text x="24" y="{top + height / 2}" class="axis" transform="rotate(-90 24 {top + height / 2})">{escape(y_label)}</text>',
    ]
    for tick in range(6):
        value = maximum * tick / 5
        y = scale(value, 0, maximum, bottom, top)
        result.extend([
            f'<line x1="{MARGIN_LEFT}" y1="{y:.1f}" x2="{WIDTH - MARGIN_RIGHT}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>',
            f'<text x="{MARGIN_LEFT - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{value:.0f}</text>',
        ])
    return result


def figure_forecast_nmae() -> None:
    rows = read_csv("forecast_model_selection.csv")
    values = [number(row, "best_mean_nmae_percent") or 0 for row in rows]
    maximum = max(15.0, math.ceil(max(values) / 5.0) * 5.0)
    content = chart_base(
        "Forecast validation error by horizon",
        "Mean NMAE from five expanding chronological validation windows; lower is better.",
        "Mean NMAE (%)",
        maximum,
    )
    slot = PLOT_WIDTH / len(rows)
    for index, (row, value) in enumerate(zip(rows, values)):
        center = MARGIN_LEFT + slot * (index + 0.5)
        bar_width = min(98, slot * 0.58)
        top = scale(value, 0, maximum, MARGIN_TOP + PLOT_HEIGHT, MARGIN_TOP)
        bottom = MARGIN_TOP + PLOT_HEIGHT
        content.extend([
            f'<rect x="{center - bar_width / 2:.1f}" y="{top:.1f}" width="{bar_width:.1f}" height="{bottom - top:.1f}" fill="{BLUE}"/>',
            f'<text x="{center:.1f}" y="{top - 8:.1f}" text-anchor="middle" class="value">{value:.2f}%</text>',
            f'<text x="{center:.1f}" y="{bottom + 24}" text-anchor="middle" class="label">{escape(row["horizon"])}</text>',
            f'<text x="{center:.1f}" y="{bottom + 43}" text-anchor="middle" class="small">{escape(row["best_model"])}</text>',
        ])
    save("forecast-validation-nmae.svg", "Forecast validation error by horizon", content)


def figure_counterfactual_security() -> None:
    all_rows = read_csv("counterfactual_analysis.csv")
    rows = [row for row in all_rows if row.get("sensitivity_case") == "base"]
    lp = [number(row, "lp_potential_energy_mwh") or 0 for row in rows]
    screened = [number(row, "security_screened_energy_mwh") or 0 for row in rows]
    maximum = max(100.0, math.ceil(max(lp) / 100.0) * 100.0)
    content = chart_base(
        "Counterfactual energy: LP potential versus AC-security screen",
        "Each group is one independent 15-minute snapshot. Only green bars may receive value or CO2 credit.",
        "Energy in interval (MWh)",
        maximum,
    )
    content.extend([
        f'<rect x="{WIDTH - 405}" y="27" width="14" height="14" fill="{BLUE}"/><text x="{WIDTH - 384}" y="39" class="small">LP potential before AC screening</text>',
        f'<rect x="{WIDTH - 200}" y="27" width="14" height="14" fill="{TEAL}"/><text x="{WIDTH - 179}" y="39" class="small">AC-security-screened</text>',
    ])
    slot = PLOT_WIDTH / len(rows)
    bottom = MARGIN_TOP + PLOT_HEIGHT
    for index, row in enumerate(rows):
        center = MARGIN_LEFT + slot * (index + 0.5)
        for offset, value, color in [(-24, lp[index], BLUE), (24, screened[index], TEAL)]:
            top = scale(value, 0, maximum, bottom, MARGIN_TOP)
            content.append(f'<rect x="{center + offset - 18:.1f}" y="{top:.1f}" width="36" height="{bottom - top:.1f}" fill="{color}"/>')
        content.extend([
            f'<text x="{center:.1f}" y="{bottom + 22}" text-anchor="middle" class="label">{escape(row["scenario"].replace("_", " "))}</text>',
            f'<text x="{center:.1f}" y="{bottom + 43}" text-anchor="middle" class="small">{escape(row["security_assessment"].replace("_", " "))}</text>',
        ])
    save("counterfactual-security-screen.svg", "Counterfactual AC security screen", content)


def figure_production_security() -> None:
    rows = read_csv("production_multi_horizon_security.csv")
    content = [
        '<text x="105" y="36" class="title">Production forecast AC security screen</text>',
        '<text x="105" y="58" class="subtitle">Every forecast horizon fails at least one configured AC criterion in the saved production run.</text>',
    ]
    panels = [
        (92, 185, "Minimum voltage (pu)", "minimum_voltage_pu", 0.85, 1.0, 0.95, BLUE),
        (350, 185, "Maximum line loading (%)", "maximum_line_loading_percent", 0.0, 260.0, 100.0, RED),
    ]
    for top, height, label, field, low, high, limit, color in panels:
        bottom = top + height
        content.extend([
            f'<rect x="{MARGIN_LEFT}" y="{top}" width="{PLOT_WIDTH}" height="{height}" fill="none" stroke="{GRID}"/>',
            f'<text x="{MARGIN_LEFT}" y="{top - 9}" class="label">{label}</text>',
        ])
        for fraction in range(5):
            value = low + (high - low) * fraction / 4
            y = scale(value, low, high, bottom, top)
            content.extend([
                f'<line x1="{MARGIN_LEFT}" y1="{y:.1f}" x2="{WIDTH - MARGIN_RIGHT}" y2="{y:.1f}" stroke="{GRID}"/>',
                f'<text x="{MARGIN_LEFT - 9}" y="{y + 4:.1f}" text-anchor="end" class="axis">{value:.2f}</text>',
            ])
        limit_y = scale(limit, low, high, bottom, top)
        content.append(f'<line x1="{MARGIN_LEFT}" y1="{limit_y:.1f}" x2="{WIDTH - MARGIN_RIGHT}" y2="{limit_y:.1f}" stroke="{ORANGE}" stroke-width="2" stroke-dasharray="6 4"/>')
        points = []
        for index, row in enumerate(rows):
            value = number(row, field)
            if value is None:
                continue
            x = MARGIN_LEFT + PLOT_WIDTH * index / (len(rows) - 1)
            y = scale(value, low, high, bottom, top)
            points.append((x, y, value))
        if len(points) > 1:
            coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
            content.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3"/>')
        for index, (x, y, value) in enumerate(points):
            content.extend([
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>',
                f'<text x="{x:.1f}" y="{y - 9:.1f}" text-anchor="middle" class="value">{value:.2f}</text>',
            ])
            if top == 350:
                content.append(f'<text x="{x:.1f}" y="{bottom + 23}" text-anchor="middle" class="label">{escape(rows[index]["horizon"])}</text>')
        content.append(f'<text x="{WIDTH - MARGIN_RIGHT}" y="{limit_y - 7:.1f}" text-anchor="end" class="small">configured limit {limit:g}</text>')
    save("production-ac-security.svg", "Production forecast AC security screen", content)


def figure_s2_continuation() -> None:
    raw_rows = read_csv("s4_5j_q_continuation.csv")
    by_q: dict[float, dict[str, str]] = {}
    for row in raw_rows:
        q = number(row, "q_percentage")
        voltage = number(row, "min_voltage_pu")
        loading = number(row, "max_line_loading_pct")
        if q is None or voltage is None or loading is None:
            continue
        if row.get("valid_ac_solution") != "True" or q > 7.1:
            continue
        by_q[q] = row
    rows = [by_q[key] for key in sorted(by_q)]
    content = [
        '<text x="105" y="36" class="title">S2 controlled reactive-load continuation</text>',
        '<text x="105" y="58" class="subtitle">Conditional converged variants only; all remain outside the configured security limits.</text>',
    ]
    panels = [
        (92, 185, "Minimum voltage (pu)", "min_voltage_pu", 0.60, 1.0, 0.95, PURPLE),
        (350, 185, "Maximum line loading (%)", "max_line_loading_pct", 0.0, 200.0, 100.0, RED),
    ]
    max_q = max(number(row, "q_percentage") or 0 for row in rows)
    for top, height, label, field, low, high, limit, color in panels:
        bottom = top + height
        content.extend([
            f'<rect x="{MARGIN_LEFT}" y="{top}" width="{PLOT_WIDTH}" height="{height}" fill="none" stroke="{GRID}"/>',
            f'<text x="{MARGIN_LEFT}" y="{top - 9}" class="label">{label}</text>',
        ])
        for fraction in range(5):
            value = low + (high - low) * fraction / 4
            y = scale(value, low, high, bottom, top)
            content.extend([
                f'<line x1="{MARGIN_LEFT}" y1="{y:.1f}" x2="{WIDTH - MARGIN_RIGHT}" y2="{y:.1f}" stroke="{GRID}"/>',
                f'<text x="{MARGIN_LEFT - 9}" y="{y + 4:.1f}" text-anchor="end" class="axis">{value:.2f}</text>',
            ])
        limit_y = scale(limit, low, high, bottom, top)
        content.append(f'<line x1="{MARGIN_LEFT}" y1="{limit_y:.1f}" x2="{WIDTH - MARGIN_RIGHT}" y2="{limit_y:.1f}" stroke="{ORANGE}" stroke-width="2" stroke-dasharray="6 4"/>')
        points = []
        for row in rows:
            q = number(row, "q_percentage") or 0
            value = number(row, field) or 0
            x = scale(q, 0, max_q, MARGIN_LEFT, WIDTH - MARGIN_RIGHT)
            y = scale(value, low, high, bottom, top)
            points.append((x, y, q, value))
        coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in points)
        content.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y, q, value in points:
            content.extend([
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>',
                f'<text x="{x:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="small">{value:.2f}</text>',
            ])
        if top == 350:
            for tick in range(0, 8):
                x = scale(float(tick), 0, max_q, MARGIN_LEFT, WIDTH - MARGIN_RIGHT)
                content.append(f'<text x="{x:.1f}" y="{bottom + 24}" text-anchor="middle" class="axis">{tick}%</text>')
            content.append(f'<text x="{MARGIN_LEFT + PLOT_WIDTH / 2}" y="{bottom + 52}" text-anchor="middle" class="axis">Assumed reactive load as % of full reference value</text>')
        content.append(f'<text x="{WIDTH - MARGIN_RIGHT}" y="{limit_y - 7:.1f}" text-anchor="end" class="small">configured limit {limit:g}</text>')
    save("s2-controlled-q-continuation.svg", "S2 controlled reactive-load continuation", content)


def figure_interventions() -> None:
    rows = read_csv("s5_5_final_stage5_consolidation.csv")
    content = [
        '<text x="105" y="36" class="title">Stage-5 intervention comparison</text>',
        '<text x="105" y="58" class="subtitle">Stylised sensitivity results; reported Stage-5 pass labels are not final security conclusions.</text>',
    ]
    panels = [
        (94, 190, "Minimum voltage (pu)", "min_voltage_pu", 0.60, 1.0, 0.95, BLUE),
        (356, 190, "Maximum line loading (%)", "max_line_loading_pct", 0.0, 200.0, 100.0, TEAL),
    ]
    slot = PLOT_WIDTH / len(rows)
    for top, height, label, field, low, high, limit, color in panels:
        bottom = top + height
        content.extend([
            f'<rect x="{MARGIN_LEFT}" y="{top}" width="{PLOT_WIDTH}" height="{height}" fill="none" stroke="{GRID}"/>',
            f'<text x="{MARGIN_LEFT}" y="{top - 9}" class="label">{label}</text>',
        ])
        for fraction in range(5):
            value = low + (high - low) * fraction / 4
            y = scale(value, low, high, bottom, top)
            content.extend([
                f'<line x1="{MARGIN_LEFT}" y1="{y:.1f}" x2="{WIDTH - MARGIN_RIGHT}" y2="{y:.1f}" stroke="{GRID}"/>',
                f'<text x="{MARGIN_LEFT - 9}" y="{y + 4:.1f}" text-anchor="end" class="axis">{value:.2f}</text>',
            ])
        limit_y = scale(limit, low, high, bottom, top)
        content.append(f'<line x1="{MARGIN_LEFT}" y1="{limit_y:.1f}" x2="{WIDTH - MARGIN_RIGHT}" y2="{limit_y:.1f}" stroke="{ORANGE}" stroke-width="2" stroke-dasharray="6 4"/>')
        for index, row in enumerate(rows):
            value = number(row, field)
            center = MARGIN_LEFT + slot * (index + 0.5)
            if value is None:
                content.append(f'<text x="{center:.1f}" y="{top + height / 2}" text-anchor="middle" class="small">not reported</text>')
            else:
                y = scale(value, low, high, bottom, top)
                content.extend([
                    f'<rect x="{center - 36:.1f}" y="{y:.1f}" width="72" height="{bottom - y:.1f}" fill="{color}"/>',
                    f'<text x="{center:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="value">{value:.2f}</text>',
                ])
            if top == 356:
                content.append(f'<text x="{center:.1f}" y="{bottom + 24}" text-anchor="middle" class="label">{escape(row["stage"])}</text>')
                status = "reported pass*" if row.get("overall_security") == "True" else "not secure"
                content.append(f'<text x="{center:.1f}" y="{bottom + 43}" text-anchor="middle" class="small">{status}</text>')
        content.append(f'<text x="{WIDTH - MARGIN_RIGHT}" y="{limit_y - 7:.1f}" text-anchor="end" class="small">configured limit {limit:g}</text>')
    content.append(
        '<text x="105" y="625" class="small">* Quarantined pending clarification of the virtual-bus voltage metric and screening treatment.</text>'
    )
    save("intervention-comparison.svg", "Stage-5 intervention comparison", content)


def main() -> None:
    figure_forecast_nmae()
    figure_counterfactual_security()
    figure_production_security()
    figure_s2_continuation()
    figure_interventions()


if __name__ == "__main__":
    main()
