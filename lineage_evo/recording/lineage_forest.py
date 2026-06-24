"""Lineage forest visualization for completed runs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from lineage_evo.lineage import FactorNode, LineageDAG, OperatorType


def write_lineage_forest_artifacts(dag: LineageDAG, log_dir: str | Path) -> dict[str, Path]:
    """Write SVG/PNG lineage forest and the display-label mapping CSV."""

    log_path = Path(log_dir)
    svg_path = log_path / "lineage_forest.svg"
    png_path = log_path / "lineage_forest.png"
    mapping_path = log_path / "lineage_forest_factor_mapping.csv"

    ordered_nodes = _ordered_nodes(dag)
    display_labels = {node.factor_id: f"F{idx:03d}" for idx, node in enumerate(ordered_nodes, start=1)}
    operator_by_node = _operator_by_node(dag)
    lineage_ids = _ordered_lineages(ordered_nodes)
    lineage_labels = {lineage_id: f"L{idx:03d}" for idx, lineage_id in enumerate(lineage_ids, start=1)}

    _write_mapping(mapping_path, ordered_nodes, display_labels, lineage_labels, operator_by_node)
    _draw_forest(dag, svg_path, png_path, ordered_nodes, display_labels, lineage_ids, lineage_labels, operator_by_node)
    return {"svg": svg_path, "png": png_path, "mapping": mapping_path}


def _ordered_nodes(dag: LineageDAG) -> list[FactorNode]:
    return list(dag.nodes.values())


def _operator_by_node(dag: LineageDAG) -> dict[str, str]:
    operators = {
        node.factor_id: "seed" if node.generation == 0 else "unknown"
        for node in dag.nodes.values()
    }
    for edge in dag.edges:
        if edge.role == "primary":
            operators[edge.child_id] = edge.operator.value
    return operators


def _ordered_lineages(nodes: list[FactorNode]) -> list[str]:
    lineages: list[str] = []
    for node in nodes:
        if node.generation == 0 and node.lineage_id not in lineages:
            lineages.append(node.lineage_id)
    for node in nodes:
        if node.lineage_id not in lineages:
            lineages.append(node.lineage_id)
    return lineages


def _write_mapping(
    path: Path,
    nodes: list[FactorNode],
    display_labels: dict[str, str],
    lineage_labels: dict[str, str],
    operator_by_node: dict[str, str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["display_id", "factor_id", "lineage_label", "lineage_id", "generation", "operator", "expression"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for node in nodes:
            writer.writerow(
                {
                    "display_id": display_labels[node.factor_id],
                    "factor_id": node.factor_id,
                    "lineage_label": lineage_labels.get(node.lineage_id, ""),
                    "lineage_id": node.lineage_id,
                    "generation": node.generation,
                    "operator": operator_by_node.get(node.factor_id, "unknown"),
                    "expression": node.expression.raw,
                }
            )


def _draw_forest(
    dag: LineageDAG,
    svg_path: Path,
    png_path: Path,
    ordered_nodes: list[FactorNode],
    display_labels: dict[str, str],
    lineage_ids: list[str],
    lineage_labels: dict[str, str],
    operator_by_node: dict[str, str],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    by_lineage_gen: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for node in ordered_nodes:
        by_lineage_gen[node.lineage_id][node.generation].append(node.factor_id)
    order_index = {node.factor_id: idx for idx, node in enumerate(ordered_nodes)}
    for lineage_id in by_lineage_gen:
        for generation in by_lineage_gen[lineage_id]:
            by_lineage_gen[lineage_id][generation].sort(key=lambda factor_id: order_index[factor_id])

    band_heights = _lineage_band_heights(by_lineage_gen)
    positions = _node_positions(lineage_ids, by_lineage_gen, band_heights)
    max_generation = max((node.generation for node in ordered_nodes), default=0)
    bottom_y = -sum(band_heights.get(lineage_id, 1.0) for lineage_id in lineage_ids)

    fig_width = max(16, min(36, 6 + max_generation * 0.55))
    fig_height = max(9, min(32, abs(bottom_y) + 1.5))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)

    y_cursor = 0.0
    for lineage_id in lineage_ids:
        height = band_heights.get(lineage_id, 1.0)
        lineage_label = lineage_labels[lineage_id]
        root_label = _lineage_root_label(ordered_nodes, lineage_id, display_labels)
        ax.axhspan(y_cursor - height + 0.10, y_cursor + 0.28, color="#f7f7f7", zorder=0)
        ax.text(-1.15, y_cursor, f"{lineage_label}\n{root_label}", ha="right", va="center", fontsize=7.5, color="#333333")
        y_cursor -= height

    _draw_edges(ax, dag, positions)
    _draw_nodes(ax, ordered_nodes, positions, display_labels, operator_by_node)
    _style_axes(ax, max_generation, y_cursor)
    _add_legend(ax)
    fig.tight_layout()
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)


def _lineage_band_heights(by_lineage_gen: dict[str, dict[int, list[str]]]) -> dict[str, float]:
    heights = {}
    for lineage_id, gen_map in by_lineage_gen.items():
        max_stack = max((len(items) for items in gen_map.values()), default=1)
        heights[lineage_id] = max(1.0, 0.36 * max_stack + 0.65)
    return heights


def _node_positions(
    lineage_ids: list[str],
    by_lineage_gen: dict[str, dict[int, list[str]]],
    band_heights: dict[str, float],
) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    y_cursor = 0.0
    for lineage_id in lineage_ids:
        height = band_heights.get(lineage_id, 1.0)
        base_y = y_cursor
        for generation, factor_ids in by_lineage_gen.get(lineage_id, {}).items():
            for stack_idx, factor_id in enumerate(factor_ids):
                positions[factor_id] = (generation, base_y - stack_idx * 0.34)
        y_cursor -= height
    return positions


def _lineage_root_label(nodes: list[FactorNode], lineage_id: str, display_labels: dict[str, str]) -> str:
    for node in nodes:
        if node.lineage_id == lineage_id and node.generation == 0:
            return display_labels[node.factor_id]
    return ""


def _draw_edges(ax: Any, dag: LineageDAG, positions: dict[str, tuple[float, float]]) -> None:
    edge_styles = {
        (OperatorType.MUTATION, "primary"): {"color": "#6b7280", "lw": 0.85, "ls": "-", "alpha": 0.60},
        (OperatorType.CROSSOVER, "primary"): {"color": "#d97706", "lw": 0.95, "ls": "-", "alpha": 0.70},
        (OperatorType.CROSSOVER, "secondary"): {"color": "#d97706", "lw": 0.75, "ls": (0, (3, 3)), "alpha": 0.38},
    }
    for edge in dag.edges:
        if edge.parent_id not in positions or edge.child_id not in positions:
            continue
        x1, y1 = positions[edge.parent_id]
        x2, y2 = positions[edge.child_id]
        style = edge_styles.get((edge.operator, edge.role), edge_styles[(OperatorType.MUTATION, "primary")])
        ax.plot([x1, x2], [y1, y2], zorder=1, solid_capstyle="round", **style)


def _draw_nodes(
    ax: Any,
    ordered_nodes: list[FactorNode],
    positions: dict[str, tuple[float, float]],
    display_labels: dict[str, str],
    operator_by_node: dict[str, str],
) -> None:
    node_styles = {
        "seed": {"color": "#e5e7eb", "edgecolor": "#374151", "marker": "s", "size": 95},
        "mutation": {"color": "#dbeafe", "edgecolor": "#2563eb", "marker": "o", "size": 72},
        "crossover": {"color": "#ffedd5", "edgecolor": "#ea580c", "marker": "o", "size": 72},
        "unknown": {"color": "#f3f4f6", "edgecolor": "#6b7280", "marker": "o", "size": 72},
    }
    for node in ordered_nodes:
        if node.factor_id not in positions:
            continue
        x, y = positions[node.factor_id]
        style = node_styles.get(operator_by_node.get(node.factor_id, "unknown"), node_styles["unknown"])
        ax.scatter(
            [x],
            [y],
            s=style["size"],
            c=style["color"],
            edgecolors=style["edgecolor"],
            linewidths=0.8,
            marker=style["marker"],
            zorder=3,
        )
        ax.text(x, y, display_labels[node.factor_id], ha="center", va="center", fontsize=4.6, color="#111827", zorder=4)


def _style_axes(ax: Any, max_generation: int, y_cursor: float) -> None:
    ax.set_title("LineageEvo Factor Evolution Forest (valid factors only)", fontsize=13, pad=26)
    ax.set_xlabel("Generation", fontsize=10)
    ax.set_ylabel("Lineage roots", fontsize=10)
    ax.set_xlim(-1.6, max_generation + 1.0)
    ax.set_ylim(y_cursor - 0.25, 0.65)
    tick_step = max(1, max_generation // 12 or 1)
    ax.set_xticks(range(0, max_generation + 1, tick_step))
    ax.set_yticks([])
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.6, zorder=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#9ca3af")


def _add_legend(ax: Any) -> None:
    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], marker="s", color="w", label="Seed factor", markerfacecolor="#e5e7eb", markeredgecolor="#374151", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="Mutation child", markerfacecolor="#dbeafe", markeredgecolor="#2563eb", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="Crossover child", markerfacecolor="#ffedd5", markeredgecolor="#ea580c", markersize=7),
        Line2D([0], [0], color="#6b7280", lw=1.2, label="Mutation / primary edge"),
        Line2D([0], [0], color="#d97706", lw=1.2, linestyle=(0, (3, 3)), label="Secondary crossover edge"),
    ]
    ax.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=5, frameon=False, fontsize=8)
