"""Vẽ biểu đồ so sánh cho báo cáo (matplotlib → PNG).

Hai biểu đồ:
  1. comparison_chart  — 4 chỉ số RAGAS × 3 phương pháp (bức tranh tổng thể).
  2. by_type_chart     — một chỉ số, tách theo loại câu hỏi × 3 phương pháp
                         (đây là biểu đồ CHỨNG MINH giả thuyết: GraphRAG vượt ở multi-hop/global).

UI dùng plotly cho tương tác; các hàm này dành cho ảnh tĩnh chèn vào báo cáo.
"""
from __future__ import annotations

from pathlib import Path

from src import config


def _require_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")            # không cần màn hình
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError("Cần matplotlib để vẽ biểu đồ: pip install matplotlib") from exc


def comparison_chart(results: dict[str, dict], out_path=None):
    """results = {method: payload}. Vẽ 4 chỉ số × các phương pháp, lưu PNG, trả đường dẫn."""
    plt = _require_mpl()
    out_path = Path(out_path or config.RESULTS_DIR / "comparison.png")
    methods = [m for m in config.METHODS if m in results]
    metrics = config.RAGAS_METRICS

    import numpy as np
    x = np.arange(len(metrics))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, method in enumerate(methods):
        overall = results[method].get("overall", {})
        vals = [overall.get(m) if isinstance(overall.get(m), (int, float)) else 0.0 for m in metrics]
        offset = (i - (len(methods) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=method,
                      color=config.METHOD_COLORS.get(method, "#a3a3a3"))
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Điểm số (0–1)")
    ax.set_title("So sánh chỉ số RAGAS giữa 3 phương pháp")
    ax.legend(title="Phương pháp")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def by_type_chart(results: dict[str, dict], metric: str = "context_recall", out_path=None):
    """Vẽ một chỉ số, tách theo loại câu hỏi (single/multi/global) × phương pháp."""
    plt = _require_mpl()
    out_path = Path(out_path or config.RESULTS_DIR / f"by_type_{metric}.png")
    methods = [m for m in config.METHODS if m in results]
    types = config.QUESTION_TYPES

    import numpy as np
    x = np.arange(len(types))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, method in enumerate(methods):
        by_type = results[method].get("by_type", {})
        vals = [by_type.get(t, {}).get(metric) if isinstance(by_type.get(t, {}).get(metric), (int, float)) else 0.0
                for t in types]
        offset = (i - (len(methods) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=method,
                      color=config.METHOD_COLORS.get(method, "#a3a3a3"))
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(["Single-hop", "Multi-hop", "Global"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(config.METRIC_LABELS_VI.get(metric, metric))
    ax.set_title(f"{config.METRIC_LABELS_VI.get(metric, metric)} theo loại câu hỏi")
    ax.legend(title="Phương pháp")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
