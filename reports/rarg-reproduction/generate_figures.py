"""Generate the report's five dependency-free SVG figures from measured results."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = json.loads((ROOT / "results" / "summary.json").read_text())
OUT = Path(__file__).resolve().parent / "images"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "DCI": "#8B95A5",
    "RARG": "#2563EB",
    "RARG+": "#7C3AED",
    "RARG++": "#E11D48",
    "ink": "#172033",
    "muted": "#667085",
    "grid": "#D9E0EA",
    "paper": "#FFFFFF",
    "accent": "#0F766E",
    "amber": "#D97706",
}


def start(title: str, subtitle: str, width: int = 900, height: int = 500) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        "<style>"
        "text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;fill:#172033}"
        ".title{font-size:25px;font-weight:700}.sub{font-size:14px;fill:#667085}"
        ".axis{font-size:13px;fill:#667085}.label{font-size:14px;font-weight:650}"
        ".value{font-size:13px;font-weight:700}.note{font-size:12px;fill:#667085}"
        "</style>",
        '<rect width="100%" height="100%" fill="#FFFFFF" rx="12"/>',
        f'<text class="title" x="55" y="45">{title}</text>',
        f'<text class="sub" x="55" y="70">{subtitle}</text>',
    ]


def finish(parts: list[str], name: str) -> None:
    parts.append("</svg>")
    (OUT / name).write_text("\n".join(parts))


def primary() -> None:
    rows = DATA["reproduction"]["conditions"]
    p = start(
        "Accuracy–efficiency frontier on 32 public questions",
        "Higher and farther left is better · Qwen3-8B open judge · mean agent tool steps",
    )
    x0, x1, y0, y1 = 120, 835, 400, 105
    xmin, xmax, ymax = 1.3, 3.4, 18
    for tick in [0, 5, 10, 15]:
        y = y0 - (tick / ymax) * (y0 - y1)
        p += [
            f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{COLORS["grid"]}"/>',
            f'<text class="axis" x="{x0-16}" y="{y+5:.1f}" text-anchor="end">{tick}%</text>',
        ]
    for tick in [1.5, 2.0, 2.5, 3.0]:
        x = x0 + (tick - xmin) / (xmax - xmin) * (x1 - x0)
        p += [
            f'<line x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y0}" stroke="{COLORS["grid"]}"/>',
            f'<text class="axis" x="{x:.1f}" y="{y0+27}" text-anchor="middle">{tick:.1f}</text>',
        ]
    pts = []
    for name in ["DCI", "RARG", "RARG+", "RARG++"]:
        x = x0 + (rows[name]["mean_tool_steps"] - xmin) / (xmax - xmin) * (x1 - x0)
        acc = rows[name]["judge_accuracy"] * 100
        y = y0 - acc / ymax * (y0 - y1)
        pts.append((x, y, name, acc, rows[name]["mean_tool_steps"]))
    p.append(
        '<polyline points="'
        + " ".join(f"{x:.1f},{y:.1f}" for x, y, *_ in pts)
        + '" fill="none" stroke="#AAB4C3" stroke-width="3" stroke-dasharray="6 7"/>'
    )
    offsets = {"DCI": (-12, -20, "end"), "RARG": (12, 28, "start"), "RARG+": (12, -21, "start"), "RARG++": (12, -20, "start")}
    for x, y, name, acc, steps in pts:
        dx, dy, anchor = offsets[name]
        p += [
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="{COLORS[name]}" stroke="white" stroke-width="3"/>',
            f'<text class="label" x="{x+dx:.1f}" y="{y+dy:.1f}" text-anchor="{anchor}">{name}</text>',
            f'<text class="note" x="{x+dx:.1f}" y="{y+dy+16:.1f}" text-anchor="{anchor}">{acc:.1f}% · {steps:.2f} steps</text>',
        ]
    p += [
        f'<text class="axis" x="{(x0+x1)/2}" y="470" text-anchor="middle">Mean tool steps → fewer is better</text>',
        '<text class="axis" transform="translate(28 260) rotate(-90)" text-anchor="middle">Judged answer accuracy</text>',
        '<text class="note" x="835" y="485" text-anchor="end">n = 32 paired questions</text>',
    ]
    finish(p, "accuracy_efficiency.svg")


def slice_accuracy() -> None:
    slices = DATA["reproduction"]["slices"]
    p = start(
        "The answer gain was concentrated in the second fixed slice",
        "Open-judge accuracy; every bar contains the same 16 questions across conditions",
    )
    x0, x1, y0, y1 = 105, 845, 410, 105
    ymax = 35
    for tick in [0, 10, 20, 30]:
        y = y0 - tick / ymax * (y0 - y1)
        p += [
            f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{COLORS["grid"]}"/>',
            f'<text class="axis" x="{x0-15}" y="{y+5:.1f}" text-anchor="end">{tick}%</text>',
        ]
    groups = [("Rows 1–16", "rows_1_16", 265), ("Rows 17–32", "rows_17_32", 650)]
    bar_w, gap = 48, 15
    for label, key, center in groups:
        start_x = center - (4 * bar_w + 3 * gap) / 2
        for i, name in enumerate(["DCI", "RARG", "RARG+", "RARG++"]):
            val = slices[key][name]["judge_accuracy"] * 100
            x = start_x + i * (bar_w + gap)
            h = val / ymax * (y0 - y1)
            if val:
                p.append(f'<rect x="{x:.1f}" y="{y0-h:.1f}" width="{bar_w}" height="{h:.1f}" rx="5" fill="{COLORS[name]}"/>')
            else:
                p.append(f'<line x1="{x:.1f}" y1="{y0-2}" x2="{x+bar_w:.1f}" y2="{y0-2}" stroke="{COLORS[name]}" stroke-width="4"/>')
            p += [
                f'<text class="value" x="{x+bar_w/2:.1f}" y="{y0-h-9:.1f}" text-anchor="middle">{val:.1f}%</text>',
                f'<text class="axis" x="{x+bar_w/2:.1f}" y="{y0+23}" text-anchor="middle">{name}</text>',
            ]
        p.append(f'<text class="label" x="{center}" y="472" text-anchor="middle">{label}</text>')
    finish(p, "accuracy_by_slice.svg")


def cost() -> None:
    rows = DATA["reproduction"]["conditions"]
    p = start(
        "Relevance reduced the text the agent had to inspect",
        "Mean grep matches shown across 32 questions; lower is better",
    )
    x0, x1, top, row_h = 190, 820, 125, 72
    for i, name in enumerate(["DCI", "RARG", "RARG+", "RARG++"]):
        val = rows[name]["mean_observed_matches"]
        y = top + i * row_h
        w = val / 100 * (x1 - x0)
        p += [
            f'<text class="label" x="{x0-18}" y="{y+21}" text-anchor="end">{name}</text>',
            f'<rect x="{x0}" y="{y}" width="{w:.1f}" height="30" rx="6" fill="{COLORS[name]}"/>',
            f'<text class="value" x="{x0+w+12:.1f}" y="{y+21}">{val:.1f}</text>',
        ]
    reduction = 100 * (1 - rows["RARG++"]["mean_observed_matches"] / rows["DCI"]["mean_observed_matches"])
    p += [
        f'<text x="{x0}" y="455" class="label" fill="{COLORS["accent"]}">RARG++ exposed {reduction:.1f}% fewer matches than DCI.</text>',
        '<text class="note" x="820" y="480" text-anchor="end">Matched subset; same 30-match observation budget per step</text>',
    ]
    finish(p, "interaction_cost.svg")


def ranks() -> None:
    values = {
        "Rows 1–16": (34665.5, 920.5),
        "Rows 17–32": (19401.5, 81.0),
        "Pooled": (25912.5, 154.5),
    }
    p = start(
        "Relevance ordering exposed gold documents much earlier",
        "Median corpus rank on a logarithmic scale; lower is better",
    )
    x0, x1, y0 = 205, 820, 145
    for power in range(0, 6):
        x = x0 + power / 5 * (x1 - x0)
        p += [
            f'<line x1="{x:.1f}" y1="105" x2="{x:.1f}" y2="405" stroke="{COLORS["grid"]}"/>',
            f'<text class="axis" x="{x:.1f}" y="430" text-anchor="middle">10^{power}</text>',
        ]
    for i, (label, (lex, rel)) in enumerate(values.items()):
        y = y0 + i * 92
        p.append(f'<text class="label" x="{x0-22}" y="{y+17}" text-anchor="end">{label}</text>')
        for j, (name, val, color) in enumerate([("Lexicographic", lex, COLORS["muted"]), ("Relevance", rel, COLORS["accent"])]):
            yy = y + j * 31
            x = x0 + math.log10(max(1, val)) / 5 * (x1 - x0)
            p += [
                f'<line x1="{x0}" y1="{yy}" x2="{x:.1f}" y2="{yy}" stroke="{color}" stroke-width="7" stroke-linecap="round"/>',
                f'<circle cx="{x:.1f}" cy="{yy}" r="7" fill="{color}"/>',
                f'<text class="value" x="{x+12:.1f}" y="{yy+5}">{val:,.1f}</text>',
            ]
    p += [
        f'<rect x="585" y="456" width="12" height="12" rx="2" fill="{COLORS["muted"]}"/><text class="note" x="604" y="467">Lexicographic</text>',
        f'<rect x="705" y="456" width="12" height="12" rx="2" fill="{COLORS["accent"]}"/><text class="note" x="724" y="467">Relevance</text>',
    ]
    finish(p, "document_rank.svg")


def evidence() -> None:
    values = {
        "Rows 1–16": (0, 6.25, 12.5),
        "Rows 17–32": (6.25, 25, 31.25),
        "Pooled": (3.125, 15.625, 21.875),
    }
    colors = [COLORS["amber"], COLORS["accent"], COLORS["RARG++"]]
    p = start(
        "Local reranking increased early gold-evidence visibility",
        "Share of questions whose gold document appeared at the entry point or in the first 30 matches",
    )
    x0, x1, y0, y1 = 95, 845, 405, 105
    ymax = 35
    for tick in [0, 10, 20, 30]:
        y = y0 - tick / ymax * (y0 - y1)
        p += [
            f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{COLORS["grid"]}"/>',
            f'<text class="axis" x="{x0-12}" y="{y+5:.1f}" text-anchor="end">{tick}%</text>',
        ]
    centers = [220, 470, 720]
    for center, (group, vals) in zip(centers, values.items()):
        for i, val in enumerate(vals):
            x = center - 60 + i * 60
            h = val / ymax * (y0 - y1)
            p.append(f'<rect x="{x}" y="{y0-h:.1f}" width="42" height="{max(h,2):.1f}" rx="4" fill="{colors[i]}"/>')
            p.append(f'<text class="value" x="{x+21}" y="{y0-h-8:.1f}" text-anchor="middle">{val:g}%</text>')
        p.append(f'<text class="label" x="{center}" y="447" text-anchor="middle">{group}</text>')
    labels = [("Paragraph entry", colors[0]), ("Ordered top 30", colors[1]), ("Reranked top 30", colors[2])]
    for i, (label, color) in enumerate(labels):
        x = 205 + i * 205
        p += [
            f'<rect x="{x}" y="473" width="12" height="12" rx="2" fill="{color}"/>',
            f'<text class="note" x="{x+19}" y="484">{label}</text>',
        ]
    finish(p, "evidence_visibility.svg")


if __name__ == "__main__":
    primary()
    slice_accuracy()
    cost()
    ranks()
    evidence()
    print(f"generated 5 SVG figures in {OUT}")
