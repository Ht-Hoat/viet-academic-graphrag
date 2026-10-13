"""Demo UI — So sánh Naive RAG · RAG+Re-ranking · GraphRAG cho hỏi-đáp tài liệu tiếng Việt.
Nhóm 7 — phần của Tuấn Anh (SV4).

    streamlit run app.py

Hai màn hình (chọn ở sidebar):
  1. 💬 Chat Demo         — hỏi 1 câu, chạy 1 hoặc so sánh cả 3 pipeline (câu trả lời, latency,
                            nguồn, và với GraphRAG thì cả loại truy vấn + đường quan hệ đồ thị).
  2. 📊 Kết quả RAGAS     — bảng + biểu đồ so sánh 4 chỉ số, TÁCH THEO LOẠI CÂU HỎI.

Khi chưa merge với 3 pipeline thật (hoặc thiếu API key), UI tự dùng pipeline MOCK để vẫn chạy mượt.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.evaluator import load_all_results
from src.pipelines import load_pipelines

st.set_page_config(page_title="GraphRAG Demo — Nhóm 7", page_icon="🕸️", layout="wide")


@st.cache_resource(show_spinner="Đang nạp pipeline & chỉ mục...")
def get_pipelines():
    pipes = load_pipelines(with_rerank=True, with_graph=True)
    return pipes


# ────────────────────────────── Chat ──────────────────────────────
def render_result(pipe, question: str):
    with st.spinner(f"Đang xử lý bằng {pipe.name}..."):
        res = pipe.query(question)

    if res.get("is_mock"):
        st.caption("🧪 Dữ liệu giả lập (mock) — pipeline thật chưa nạp hoặc thiếu API key.")

    st.markdown("**Trả lời:**")
    st.write(res["answer"] or "*(trống)*")

    meta = f"⏱️ {res['latency']}s"
    if res.get("search_type"):
        badge = "🟦 Local" if res["search_type"] == "Local" else "🟩 Global"
        meta += f" · truy vấn {badge}"
    st.caption(meta)

    if res.get("graph_paths"):
        with st.expander(f"🕸️ Đường quan hệ đồ thị ({len(res['graph_paths'])})"):
            for p in res["graph_paths"]:
                st.markdown(f"- {p}")

    with st.expander(f"📄 Nguồn ({len(res['sources'])}) & ngữ cảnh"):
        for s in res["sources"]:
            st.markdown(f"- {s}")
        if res["contexts"]:
            st.markdown("---")
            for i, c in enumerate(res["contexts"], 1):
                st.caption(f"[{i}] {c[:300]}")


def screen_chat(pipes):
    st.title("🕸️ Hỏi-Đáp Tài liệu Học thuật tiếng Việt")
    st.caption("Nhóm 7 · So sánh Naive RAG — RAG+Re-ranking — GraphRAG")

    with st.form("q"):
        question = st.text_input("💬 Nhập câu hỏi (tiếng Việt):",
                                 placeholder="VD: Tác giả nào đề xuất BERT và họ thuộc tổ chức nào?")
        compare = st.checkbox("So sánh cả 3 phương pháp", value=True)
        submitted = st.form_submit_button("Gửi câu hỏi", type="primary")

    if not submitted:
        return
    question = (question or "").strip()
    if not question:
        st.warning("Vui lòng nhập câu hỏi.")
        return

    st.divider()
    if compare:
        cols = st.columns(len(pipes))
        icons = {"Naive RAG": "🟢", "RAG + Re-ranking": "🟡", "GraphRAG": "🟣"}
        for col, (name, pipe) in zip(cols, pipes.items()):
            with col:
                st.subheader(f"{icons.get(name, '•')} {name}")
                render_result(pipe, question)
    else:
        name = st.session_state.get("single_method", config.METHODS[0])
        render_result(pipes[name], question)


# ────────────────────────── Kết quả RAGAS ──────────────────────────
def _overall_df(results):
    rows = {m: {config.METRIC_LABELS_VI[k].split(" (")[0]: results[m]["overall"].get(k)
                for k in config.RAGAS_METRICS}
            for m in config.METHODS if m in results}
    return pd.DataFrame.from_dict(rows, orient="index")


def _bar_overall(df):
    fig = go.Figure()
    for method in df.index:
        fig.add_trace(go.Bar(name=method, x=list(df.columns), y=df.loc[method].values,
                             marker_color=config.METHOD_COLORS.get(method, "#a3a3a3"),
                             text=[f"{v:.2f}" if pd.notna(v) else "" for v in df.loc[method].values],
                             textposition="outside"))
    fig.update_layout(barmode="group", title="4 chỉ số RAGAS — trung bình chung",
                      yaxis=dict(title="Điểm (0–1)", range=[0, 1.05]), height=440, margin=dict(t=50, b=30))
    return fig


def _bar_by_type(results, metric):
    fig = go.Figure()
    types = config.QUESTION_TYPES
    for method in config.METHODS:
        if method not in results:
            continue
        bt = results[method].get("by_type", {})
        ys = [bt.get(t, {}).get(metric) for t in types]
        fig.add_trace(go.Bar(name=method, x=["Single-hop", "Multi-hop", "Global"], y=ys,
                             marker_color=config.METHOD_COLORS.get(method, "#a3a3a3"),
                             text=[f"{v:.2f}" if isinstance(v, (int, float)) else "" for v in ys],
                             textposition="outside"))
    fig.update_layout(barmode="group",
                      title=f"{config.METRIC_LABELS_VI.get(metric, metric)} — theo loại câu hỏi",
                      yaxis=dict(title="Điểm (0–1)", range=[0, 1.05]), height=440, margin=dict(t=50, b=30))
    return fig


def screen_eval():
    st.title("📊 Kết quả đánh giá RAGAS")
    results = load_all_results()

    if not results:
        st.info("Chưa có kết quả. Chạy `python -m scripts.run_eval` (hoặc `--mock` để thử luồng).")
        return
    if any(r.get("source") == "sample" for r in results.values()):
        st.warning("⚠️ Đang hiển thị **số liệu MẪU** (results/sample/) để minh họa giao diện. "
                   "Chạy `python -m scripts.run_eval` để có số liệu thật.")

    st.subheader("Trung bình chung")
    df = _overall_df(results)
    st.dataframe(df.style.format("{:.2f}").background_gradient(cmap="Greens", axis=0),
                 use_container_width=True)
    st.plotly_chart(_bar_overall(df), use_container_width=True)

    st.subheader("Theo loại câu hỏi — bằng chứng cho giả thuyết")
    metric = st.selectbox("Chỉ số:", config.RAGAS_METRICS, index=config.RAGAS_METRICS.index("context_recall"),
                          format_func=lambda m: config.METRIC_LABELS_VI.get(m, m))
    st.plotly_chart(_bar_by_type(results, metric), use_container_width=True)
    st.caption("Kỳ vọng: single-hop ba phương pháp tương đương; multi-hop và global thì GraphRAG vượt lên.")


# ────────────────────────────── Main ──────────────────────────────
def main():
    pipes = get_pipelines()

    st.sidebar.title("⚙️ Cấu hình")
    screen = st.sidebar.radio("Màn hình", ["💬 Chat Demo", "📊 Kết quả RAGAS"])
    st.sidebar.divider()
    st.sidebar.radio("Phương pháp (khi không so sánh)", config.METHODS, key="single_method")
    st.sidebar.divider()
    st.sidebar.caption("Trạng thái backend:")
    for name, pipe in pipes.items():
        st.sidebar.caption(f"- {name}: {'🧪 Mock' if pipe.is_mock else '✅ Thật'}")

    if screen == "💬 Chat Demo":
        screen_chat(pipes)
    else:
        screen_eval()

    st.divider()
    st.caption("Nhóm 7 · Học Máy Nâng Cao · GraphRAG cho tài liệu tiếng Việt")


if __name__ == "__main__":
    main()
