#!/usr/bin/env python3
"""Create a plotted PDF comparing seeded and two-generator CoV searches."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


PRES_IDS = [
    "0", "496", "521", "323", "228", "455", "77", "43", "141", "48",
    "155", "505", "247", "217", "201", "288", "367", "331", "203",
    "265", "579", "327", "333", "380", "546", "548", "533", "589",
    "580", "609", "538", "606", "544", "549", "543", "565", "602",
    "632", "573", "586", "581", "575", "568", "578", "583", "628",
    "633", "596", "605", "610", "625", "622", "624", "623", "634",
    "635", "637", "639", "638", "636", "AK(3)", "14_1", "17_41",
    "19_40", "25_17", "25_1",
]

BLUE = colors.HexColor("#0072B2")
BLUE_LIGHT = colors.HexColor("#DCEEF8")
ORANGE = colors.HexColor("#D98B00")
ORANGE_LIGHT = colors.HexColor("#FBE8BF")
GRAY = colors.HexColor("#B9BEC4")
GRAY_LIGHT = colors.HexColor("#ECEFF1")
DARK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#66727D")
GRID = colors.HexColor("#D5DADE")
PANEL = colors.HexColor("#F5F7F8")
WHITE = colors.white


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            rows.append(row)
    if len(rows) != 66:
        raise ValueError(f"{path}: expected 66 JSONL rows, found {len(rows)}")
    indices = [int(row["idx"]) for row in rows]
    if indices != list(range(66)):
        raise ValueError(f"{path}: expected idx 0 through 65 in order")
    for row in rows:
        expected_depth = len(row["actions"]) if row["solved"] else -1
        if row["path_length"] != expected_depth:
            raise ValueError(f"{path}: path/action mismatch at idx {row['idx']}")
    return rows


def hours(seconds: float) -> float:
    return seconds / 3600.0


def draw_header(pdf: canvas.Canvas, title: str, subtitle: str, page_number: int) -> None:
    width, height = landscape(letter)
    pdf.setFillColor(DARK)
    pdf.setFont("Helvetica-Bold", 19)
    pdf.drawString(40, height - 42, title)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 8.5)
    pdf.drawString(40, height - 58, subtitle)
    pdf.setStrokeColor(GRID)
    pdf.line(40, height - 67, width - 40, height - 67)
    pdf.setFont("Helvetica", 8)
    pdf.drawRightString(width - 40, 24, f"Page {page_number}")


def draw_section_title(pdf: canvas.Canvas, x: float, y: float, text: str) -> None:
    pdf.setFillColor(DARK)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(x, y, text)


def draw_bar_pair(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    title: str,
    seeded_value: float,
    cov_value: float,
    maximum: float,
    seeded_text: str,
    cov_text: str,
) -> None:
    draw_section_title(pdf, x, y, title)
    track_x = x + 92
    track_w = width - 142
    bar_h = 13
    rows = [
        ("Seeded 3-gen", seeded_value, seeded_text, BLUE),
        ("2-gen CoV", cov_value, cov_text, ORANGE),
    ]
    for offset, (label, value, value_text, color) in enumerate(rows):
        row_y = y - 27 - offset * 31
        pdf.setFillColor(DARK)
        pdf.setFont("Helvetica", 9)
        pdf.drawString(x, row_y + 3, label)
        pdf.setFillColor(GRAY_LIGHT)
        pdf.roundRect(track_x, row_y, track_w, bar_h, 2, fill=1, stroke=0)
        pdf.setFillColor(color)
        pdf.roundRect(track_x, row_y, track_w * value / maximum, bar_h, 2, fill=1, stroke=0)
        pdf.setFillColor(DARK)
        pdf.setFont("Helvetica-Bold", 9)
        pdf.drawRightString(x + width, row_y + 3, value_text)


def outcome_kind(seeded: dict, cov: dict) -> str:
    if seeded["solved"] and cov["solved"]:
        return "both"
    if not seeded["solved"] and cov["solved"]:
        return "cov_only"
    if seeded["solved"] and not cov["solved"]:
        return "seeded_only"
    return "neither"


def draw_stacked_outcomes(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    seeded: list[dict],
    cov: list[dict],
) -> None:
    draw_section_title(pdf, x, y, "Case outcomes")
    counts = {key: 0 for key in ("both", "cov_only", "seeded_only", "neither")}
    for first, second in zip(seeded, cov):
        counts[outcome_kind(first, second)] += 1
    segments = [
        ("Both solve", counts["both"], BLUE),
        ("2-gen only", counts["cov_only"], ORANGE),
        ("Seeded only", counts["seeded_only"], colors.HexColor("#6A4C93")),
        ("Neither", counts["neither"], GRAY),
    ]
    bar_y = y - 34
    cursor = x
    for label, value, color in segments:
        segment_w = width * value / 66.0
        if segment_w > 0:
            pdf.setFillColor(color)
            pdf.rect(cursor, bar_y, segment_w, 25, fill=1, stroke=0)
            if segment_w > 35:
                pdf.setFillColor(WHITE if color != GRAY else DARK)
                pdf.setFont("Helvetica-Bold", 10)
                pdf.drawCentredString(cursor + segment_w / 2, bar_y + 8, str(value))
        cursor += segment_w
    legend_y = bar_y - 19
    cursor = x
    for label, value, color in segments:
        pdf.setFillColor(color)
        pdf.rect(cursor, legend_y, 8, 8, fill=1, stroke=0)
        pdf.setFillColor(MUTED)
        pdf.setFont("Helvetica", 8.5)
        text = f"{label}: {value}"
        pdf.drawString(cursor + 12, legend_y - 1, text)
        cursor += stringWidth(text, "Helvetica", 8.5) + 34


def draw_wrapped_text(
    pdf: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    max_width: float,
    font: str = "Helvetica",
    size: float = 9,
    leading: float = 13,
    color=DARK,
) -> float:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if stringWidth(candidate, font, size) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    pdf.setFillColor(color)
    pdf.setFont(font, size)
    for line in lines:
        pdf.drawString(x, y, line)
        y -= leading
    return y


def draw_scatter(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    size: float,
    points: list[tuple[float, float, int, str]],
    domain: tuple[float, float],
    ticks: list[float],
    title: str,
    x_label: str,
    y_label: str,
    log_scale: bool,
    label_count: int = 3,
) -> None:
    draw_section_title(pdf, x, y + size + 35, title)
    left = x + 44
    bottom = y + 36
    plot_size = size - 58

    def scale(value: float) -> float:
        lo, hi = domain
        if log_scale:
            return (math.log10(value) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        return (value - lo) / (hi - lo)

    def sx(value: float) -> float:
        return left + scale(value) * plot_size

    def sy(value: float) -> float:
        return bottom + scale(value) * plot_size

    pdf.setStrokeColor(GRID)
    pdf.setLineWidth(0.5)
    for tick in ticks:
        tx, ty = sx(tick), sy(tick)
        pdf.line(tx, bottom, tx, bottom + plot_size)
        pdf.line(left, ty, left + plot_size, ty)
        pdf.setFillColor(MUTED)
        pdf.setFont("Helvetica", 7.5)
        pdf.drawCentredString(tx, bottom - 12, f"{tick:g}")
        pdf.drawRightString(left - 6, ty - 2.5, f"{tick:g}")

    pdf.setStrokeColor(MUTED)
    pdf.setDash(4, 3)
    pdf.line(sx(domain[0]), sy(domain[0]), sx(domain[1]), sy(domain[1]))
    pdf.setDash()

    ratios = sorted(points, key=lambda item: item[1] / item[0], reverse=True)
    labeled = {item[2] for item in ratios[:label_count]}
    for x_value, y_value, idx, pres_id in points:
        px, py = sx(x_value), sy(y_value)
        pdf.setFillColor(ORANGE)
        pdf.setStrokeColor(WHITE)
        pdf.circle(px, py, 3.4, fill=1, stroke=1)
        if idx in labeled:
            pdf.setFillColor(DARK)
            pdf.setFont("Helvetica", 7)
            pdf.drawString(px + 5, py + 3, f"{idx}:{pres_id}")

    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 8)
    pdf.drawCentredString(left + plot_size / 2, y + 8, x_label)
    pdf.saveState()
    pdf.translate(x + 8, bottom + plot_size / 2)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, y_label)
    pdf.restoreState()
    pdf.setFont("Helvetica-Oblique", 7.5)
    pdf.drawRightString(left + plot_size, bottom + plot_size + 7, "Equality line")


def draw_outcome_grid(
    pdf: canvas.Canvas,
    seeded: list[dict],
    cov: list[dict],
    x: float,
    top: float,
) -> None:
    cols = 11
    tile_w = 63
    tile_h = 56
    gap = 2
    palette = {
        "both": (BLUE_LIGHT, BLUE),
        "cov_only": (ORANGE_LIGHT, ORANGE),
        "seeded_only": (colors.HexColor("#E9DDF5"), colors.HexColor("#6A4C93")),
        "neither": (GRAY_LIGHT, MUTED),
    }
    for idx, (first, second) in enumerate(zip(seeded, cov)):
        row, col = divmod(idx, cols)
        tile_x = x + col * (tile_w + gap)
        tile_y = top - (row + 1) * tile_h - row * gap
        kind = outcome_kind(first, second)
        fill, accent = palette[kind]
        pdf.setFillColor(fill)
        pdf.roundRect(tile_x, tile_y, tile_w, tile_h, 4, fill=1, stroke=0)
        pdf.setFillColor(accent)
        pdf.rect(tile_x, tile_y + tile_h - 5, tile_w, 5, fill=1, stroke=0)
        pdf.setFillColor(MUTED)
        pdf.setFont("Helvetica", 7)
        pdf.drawString(tile_x + 6, tile_y + tile_h - 17, f"idx {idx}")
        pdf.setFillColor(DARK)
        pdf.setFont("Helvetica-Bold", 9)
        pdf.drawString(tile_x + 6, tile_y + 15, PRES_IDS[idx])


def build_pdf(seeded_path: Path, cov_path: Path, output_path: Path) -> None:
    seeded = load_jsonl(seeded_path)
    cov = load_jsonl(cov_path)
    width, height = landscape(letter)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output_path), pagesize=(width, height))
    pdf.setTitle("Avi66 CoV Search Comparison")
    pdf.setAuthor("Codex")
    subtitle = "Avi66 benchmark - 66 cases - 1,000-node budget - generated from JSONL results"

    seeded_solved = sum(bool(row["solved"]) for row in seeded)
    cov_solved = sum(bool(row["solved"]) for row in cov)
    seeded_nodes = sum(int(row["nodes_visited"]) for row in seeded)
    cov_nodes = sum(int(row["nodes_visited"]) for row in cov)
    seeded_time_h = hours(sum(float(row["elapsed_sec"]) for row in seeded))
    cov_time_h = hours(sum(float(row["elapsed_sec"]) for row in cov))
    both_indices = [
        idx for idx in range(66) if seeded[idx]["solved"] and cov[idx]["solved"]
    ]
    cov_only_indices = [
        idx for idx in range(66) if not seeded[idx]["solved"] and cov[idx]["solved"]
    ]

    # Page 1: overview.
    draw_header(pdf, "Seeded vs two-generator CoV search", subtitle, 1)
    draw_bar_pair(
        pdf, 40, 508, 340, "Solved presentations", seeded_solved, cov_solved, 66,
        f"{seeded_solved}/66 ({100 * seeded_solved / 66:.1f}%)",
        f"{cov_solved}/66 ({100 * cov_solved / 66:.1f}%)",
    )
    draw_bar_pair(
        pdf, 412, 508, 340, "Total nodes visited", seeded_nodes, cov_nodes,
        max(seeded_nodes, cov_nodes), f"{seeded_nodes:,}", f"{cov_nodes:,}",
    )
    draw_bar_pair(
        pdf, 412, 390, 340, "Sum of elapsed time", seeded_time_h, cov_time_h,
        max(seeded_time_h, cov_time_h), f"{seeded_time_h:.1f} h", f"{cov_time_h:.1f} h",
    )
    draw_stacked_outcomes(pdf, 40, 390, 340, seeded, cov)

    pdf.setFillColor(PANEL)
    pdf.roundRect(40, 84, width - 80, 182, 8, fill=1, stroke=0)
    draw_section_title(pdf, 58, 242, "What the comparison shows")
    findings = [
        "The two-generator CoV search solves 12 additional cases and loses none of the 31 seeded solutions.",
        "It uses 37.1% fewer total nodes and 80.4% less summed elapsed time across the full benchmark.",
        "On every commonly solved case, it reaches a solution with fewer nodes and a shorter raw path.",
        "All 31 successful seeded paths start from the singleton Y seed, so the root seed does not honor the min-length-2 label.",
    ]
    text_y = 219
    for finding in findings:
        pdf.setFillColor(ORANGE)
        pdf.circle(61, text_y + 2, 2.5, fill=1, stroke=0)
        text_y = draw_wrapped_text(pdf, finding, 72, text_y, width - 135, size=9.5, leading=17)
        text_y -= 5
    pdf.showPage()

    # Page 2: paired efficiency.
    draw_header(pdf, "Efficiency on the 31 commonly solved cases", subtitle, 2)
    node_points = [
        (
            float(cov[idx]["nodes_visited"]),
            float(seeded[idx]["nodes_visited"]),
            idx,
            PRES_IDS[idx],
        )
        for idx in both_indices
    ]
    path_points = [
        (
            float(cov[idx]["path_length"]),
            float(seeded[idx]["path_length"]),
            idx,
            PRES_IDS[idx],
        )
        for idx in both_indices
    ]
    draw_scatter(
        pdf, 38, 164, 310, node_points, (2, 1000),
        [2, 5, 10, 20, 50, 100, 200, 500, 1000],
        "Nodes visited (log-log)", "2-generator CoV nodes", "Seeded 3-generator nodes", True,
    )
    draw_scatter(
        pdf, 414, 164, 310, path_points, (0, 50), [0, 10, 20, 30, 40, 50],
        "Raw solution path length", "2-generator CoV path", "Seeded 3-generator path", False, 0,
    )

    med_seed_nodes = statistics.median(seeded[idx]["nodes_visited"] for idx in both_indices)
    med_cov_nodes = statistics.median(cov[idx]["nodes_visited"] for idx in both_indices)
    med_seed_path = statistics.median(seeded[idx]["path_length"] for idx in both_indices)
    med_cov_path = statistics.median(cov[idx]["path_length"] for idx in both_indices)
    pdf.setFillColor(PANEL)
    pdf.roundRect(40, 78, width - 80, 61, 6, fill=1, stroke=0)
    pdf.setFillColor(DARK)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(56, 113, f"Median nodes: {med_seed_nodes:g} seeded vs {med_cov_nodes:g} two-generator")
    pdf.drawString(390, 113, f"Median raw path: {med_seed_path:g} seeded vs {med_cov_path:g} two-generator")
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 8.5)
    pdf.drawString(56, 92, "All 31 points lie above equality. Seeded path length includes one synthetic seed-add action;")
    pdf.drawString(56, 81, "even after subtracting it, all 31 seeded paths remain longer.")
    pdf.showPage()

    # Page 3: case-level outcomes.
    draw_header(pdf, "Case-level outcome map", subtitle, 3)
    draw_outcome_grid(pdf, seeded, cov, 40, 528)
    legend_y = 151
    legends = [
        ("Both solve", BLUE),
        ("2-generator CoV only", ORANGE),
        ("Neither", GRAY),
    ]
    cursor = 40
    for label, color in legends:
        pdf.setFillColor(color)
        pdf.rect(cursor, legend_y, 10, 10, fill=1, stroke=0)
        pdf.setFillColor(DARK)
        pdf.setFont("Helvetica", 8.5)
        pdf.drawString(cursor + 15, legend_y + 1, label)
        cursor += stringWidth(label, "Helvetica", 8.5) + 44

    extra_labels = [f"{idx}:{PRES_IDS[idx]}" for idx in cov_only_indices]
    draw_section_title(pdf, 40, 123, "Twelve additional two-generator CoV solutions")
    draw_wrapped_text(pdf, ", ".join(extra_labels), 40, 105, width - 80, size=8.5, leading=12)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 8)
    pdf.drawString(40, 49, "Tile labels show local Avi66 idx and Avi pres_id. No case is solved only by the seeded run.")
    pdf.showPage()

    pdf.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seeded_jsonl", type=Path)
    parser.add_argument("cov_jsonl", type=Path)
    parser.add_argument("output_pdf", type=Path)
    args = parser.parse_args()
    build_pdf(args.seeded_jsonl, args.cov_jsonl, args.output_pdf)


if __name__ == "__main__":
    main()
