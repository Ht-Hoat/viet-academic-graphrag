"""Test vẽ biểu đồ — bỏ qua nếu máy chưa cài matplotlib (không bắt buộc cho test lõi)."""
import pytest

from src.evaluator import load_all_results

pytest.importorskip("matplotlib", reason="matplotlib chưa cài — bỏ qua test biểu đồ")

from src.charts import by_type_chart, comparison_chart  # noqa: E402


def test_comparison_chart_writes_png(tmp_path):
    results = load_all_results()          # số liệu mẫu
    out = comparison_chart(results, out_path=tmp_path / "cmp.png")
    assert out.exists() and out.stat().st_size > 0


def test_by_type_chart_writes_png(tmp_path):
    results = load_all_results()
    out = by_type_chart(results, metric="context_recall", out_path=tmp_path / "bt.png")
    assert out.exists() and out.stat().st_size > 0
