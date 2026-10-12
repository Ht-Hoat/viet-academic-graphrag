"""Pipeline 3: GraphRAG — Local Search (vector + duyệt đồ thị) và Global Search (map-reduce trên tóm tắt cộng đồng).

Dùng trong code (UI của Tuấn Anh, evaluator RAGAS):
    from src.graph_rag import build_graph_rag
    rag = build_graph_rag(vectorstore=vectorstore, embedder=embeddings)
    result = rag.query("Tác giả của BERT làm việc ở tổ chức nào?")
    run_evaluation(rag, eval_questions, "GraphRAG")   # rag gọi được như rag_func(question)

Dòng lệnh:
    python -m src.graph_rag ask "câu hỏi" [--retrieval-only] [--faiss-dir data/faiss]
    python -m src.graph_rag eval --questions data/eval/questions.json [--retrieval-only]
"""
import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import chain, combinations, zip_longest
from pathlib import Path
from statistics import mean

from tqdm import tqdm

from src import config
from src.graph_builder import Chunk, entity_key, load_communities, source_label, to_chunk
from src.graph_store import get_store
from src.llm_utils import call_llm, get_llm, parse_json

ANALYZE_PROMPT = """Phân tích câu hỏi về kho tài liệu học thuật dưới đây.
- "type": "GLOBAL" nếu câu hỏi hỏi về xu hướng / tổng quan / chủ đề chính của cả kho tài liệu;
  "LOCAL" nếu hỏi về thực thể, khái niệm hoặc quan hệ cụ thể.
- "entities": các thực thể được nhắc tới trong câu hỏi (tên riêng, phương pháp, khái niệm, tổ chức), giữ nguyên cách viết.

Câu hỏi: {question}

Chỉ trả về JSON: {{"type": "LOCAL", "entities": ["..."]}}"""

ANSWER_PROMPT = """Dựa vào ngữ cảnh sau đây (gồm quan hệ trích từ đồ thị tri thức và trích đoạn tài liệu),
hãy trả lời câu hỏi bằng tiếng Việt.
Nếu không tìm thấy thông tin trong ngữ cảnh, hãy nói "Tôi không tìm thấy thông tin liên quan."
Luôn trích dẫn nguồn (tên file, trang) khi trả lời.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:"""

MAP_PROMPT = """Dưới đây là báo cáo tóm tắt một cụm tri thức trong kho tài liệu.

Tiêu đề: {title}
Tóm tắt: {summary}

Câu hỏi: {question}

Chỉ dựa vào báo cáo trên, viết ý trả lời một phần cho câu hỏi (2-4 câu, tiếng Việt) và chấm mức hữu ích 0-100
(0 nếu báo cáo không liên quan).
Chỉ trả về JSON: {{"answer": "...", "score": 0}}"""

REDUCE_PROMPT = """Bạn nhận được các ý trả lời một phần, mỗi ý rút ra từ một cụm tri thức trong kho tài liệu.

{partials}

Hãy tổng hợp thành câu trả lời hoàn chỉnh bằng tiếng Việt cho câu hỏi: {question}
Gộp các ý trùng nhau, sắp xếp theo mức quan trọng, ghi mã cụm [C#] sau mỗi ý.
Nếu các ý không đủ thông tin, hãy nói "Tôi không tìm thấy thông tin liên quan."

Trả lời:"""

GRAPH_CONTEXT_HEADER = "=== Quan hệ từ đồ thị tri thức ==="
DOC_CONTEXT_HEADER = "=== Trích đoạn tài liệu ==="

GLOBAL_HINTS = ("xu hướng", "tổng quan", "chủ đề chính", "các chủ đề", "nhìn chung", "toàn bộ tài liệu",
                "tổng kết", "hướng nghiên cứu chính", "những hướng nghiên cứu", "nói chung")
# Câu hỏi nhắm tới loại thực thể nào → ưu tiên đường đi chạm tới loại đó ("... tổ chức nào?" → ORG)
TYPE_HINTS = {
    "ORG": ("tổ chức", "trường", "viện", "công ty", "cơ quan", "đơn vị"),
    "PERSON": ("tác giả", "người", "nhà nghiên cứu", "nhóm nghiên cứu"),
    "PAPER": ("bài báo", "công trình", "giáo trình", "tài liệu nào"),
    "DATASET": ("bộ dữ liệu", "dataset", "độ đo", "benchmark"),
    "METHOD": ("phương pháp", "mô hình", "thuật toán", "kỹ thuật", "kiến trúc"),
    "TASK": ("bài toán", "ứng dụng", "nhiệm vụ"),
}
# Không đưa "ai" vào đây: nó vừa là từ để hỏi, vừa là một phần của tên thực thể ("Google AI Language")
STOPWORDS = {"là", "và", "của", "các", "những", "có", "không", "gì", "nào", "được", "trong", "cho", "với",
             "về", "như", "thế", "một", "này", "đó", "ở", "khi", "thì", "ra", "sao", "bao", "giờ"}


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", str(text or ""))


def _phrase_text(text: str) -> str:
    """Chuỗi đã tách từ, có khoảng trắng hai đầu — để dò cụm từ khoá mà không khớp nhầm giữa từ."""
    return " " + " ".join(re.findall(r"\w+", _normalize(text).lower())) + " "


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"\w+", _normalize(text).lower()) if len(t) > 1 and t not in STOPWORDS}


def _unique(items) -> list:
    return list(dict.fromkeys(items))


def heuristic_type(question: str) -> str:
    """Phân loại LOCAL/GLOBAL bằng từ khoá — dùng khi không có LLM hoặc LLM lỗi."""
    phrase = _phrase_text(question)
    return "GLOBAL" if any(f" {hint} " in phrase for hint in GLOBAL_HINTS) else "LOCAL"


def question_type_hints(question: str) -> set[str]:
    phrase = _phrase_text(question)
    return {etype for etype, hints in TYPE_HINTS.items() if any(f" {hint} " in phrase for hint in hints)}


def path_to_text(path: dict) -> str:
    """Viết đường đi thành câu, giữ đúng chiều và giữ đủ node trung gian."""
    names = {n["key"]: n["name"] for n in path["nodes"]}
    return "; ".join(
        f"{names.get(r['source'], r['source'])} --({r['type'].replace('_', ' ')})--> {names.get(r['target'], r['target'])}"
        for r in path["rels"]
    )


def rank_paths(paths: list[dict], seed_keys: set[str], type_hints: set[str]) -> list[dict]:
    """Bỏ trùng và xếp hạng đường đi: ưu tiên đường nối ≥2 thực thể của câu hỏi (ca đa bước), đường chạm loại
    thực thể mà câu hỏi nhắm tới, đường ngắn và cạnh xuất hiện nhiều lần."""
    best: dict[frozenset, tuple[float, dict]] = {}
    for path in paths:
        rels = path.get("rels") or []
        if not rels:
            continue
        signature = frozenset((r["source"], r["type"], r["target"]) for r in rels)
        keys = {n["key"] for n in path["nodes"]}
        weight = sum(r.get("weight") or 1 for r in rels) / len(rels)
        score = 1.0 / len(rels) + 0.05 * min(weight, 10)
        if len(keys & seed_keys) >= 2:
            score += 2.0
        if any(n["type"] in type_hints for n in path["nodes"] if n["key"] not in seed_keys):
            score += 1.0
        if signature not in best or score > best[signature][0]:
            best[signature] = (score, path)
    return [path for _, path in sorted(best.values(), key=lambda item: -item[0])]


def merge_chunks(vector_chunks: list[Chunk], graph_chunks: list[Chunk], max_chars: int) -> list[Chunk]:
    """Xen kẽ chunk từ vector search và chunk từ đồ thị, bỏ trùng, dừng khi hết ngân sách ký tự."""
    merged, seen_ids, seen_texts, total = [], set(), set(), 0
    for chunk in chain.from_iterable(zip_longest(vector_chunks, graph_chunks)):
        if chunk is None or chunk.id in seen_ids or chunk.text in seen_texts:
            continue
        if merged and total + len(chunk.text) > max_chars:
            continue
        seen_ids.add(chunk.id)
        seen_texts.add(chunk.text)
        merged.append(chunk)
        total += len(chunk.text)
    return merged


def build_context(facts: list[str], chunks: list[Chunk]) -> str:
    parts = []
    if facts:
        parts.append(GRAPH_CONTEXT_HEADER + "\n" + "\n".join(f"- {fact}" for fact in facts))
    if chunks:
        parts.append(DOC_CONTEXT_HEADER + "\n" + "\n\n".join(
            f"[{i}] Nguồn: {source_label(c.source, c.page)}\n{c.text}" for i, c in enumerate(chunks, 1)))
    return "\n\n".join(parts) or "(Không tìm thấy ngữ cảnh liên quan)"


def paths_to_subgraph(paths: list[dict], seed_keys: set[str]) -> dict:
    """Dữ liệu vẽ đồ thị cho UI."""
    nodes, edges = {}, {}
    for path in paths:
        for node in path["nodes"]:
            nodes.setdefault(node["key"], {"id": node["key"], "label": node["name"], "type": node["type"],
                                           "seed": node["key"] in seed_keys})
        for rel in path["rels"]:
            edges.setdefault((rel["source"], rel["type"], rel["target"]),
                             {"source": rel["source"], "target": rel["target"], "label": rel["type"]})
    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


def communities_to_subgraph(communities: list[dict]) -> dict:
    nodes, edges = {}, {}
    for community in communities:
        for member in community.get("members", []):
            nodes.setdefault(member["key"], {"id": member["key"], "label": member["name"],
                                             "type": member.get("type"), "community": community["id"]})
    for community in communities:
        for rel in community.get("relations", []):
            if rel["source"] in nodes and rel["target"] in nodes:
                edges.setdefault((rel["source"], rel["type"], rel["target"]),
                                 {"source": rel["source"], "target": rel["target"], "label": rel["type"]})
    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


def _unit_rows(matrix):
    """Chuẩn hoá vector về độ dài 1 để tích vô hướng chính là cosine."""
    import numpy as np

    matrix = np.asarray(matrix, dtype="float32")
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, 1e-9, None)


class EntityLinker:
    """Nối tên thực thể trong câu hỏi với node trong đồ thị: khớp tên/tên gọi khác → fulltext index → cosine
    embedding; kèm quét các tên thực thể xuất hiện nguyên văn trong câu hỏi (phòng khi LLM bỏ sót)."""

    def __init__(self, store, embedder=None):
        self.store = store
        self.embedder = embedder
        self._entities = None
        self._alias_to_key = {}
        self._pattern = None
        self._name_keys = None
        self._name_vectors = None

    def _load(self):
        if self._entities is not None:
            return
        rows = self.store.entity_index()
        self._entities = {row["key"]: row for row in rows}
        for row in rows:
            for alias in [row["name"], *(row.get("aliases") or [])]:
                key = entity_key(alias)
                if key:
                    self._alias_to_key.setdefault(key, row["key"])
        names = sorted((n for n in self._alias_to_key if len(n) >= 2), key=len, reverse=True)
        if names:
            self._pattern = re.compile(rf"(?<!\w)(?:{'|'.join(re.escape(n) for n in names)})(?!\w)", re.IGNORECASE)

    def link(self, question: str, mentions: list[str], limit: int = config.MAX_SEED_ENTITIES) -> list[dict]:
        self._load()
        keys = []
        for mention in mentions:
            key = self._match_mention(mention)
            if key and key not in keys:
                keys.append(key)
        if self._pattern:
            for match in self._pattern.finditer(_normalize(question)):
                key = self._alias_to_key.get(entity_key(match.group()))
                if key and key not in keys:
                    keys.append(key)
        return [self._entities[key] for key in keys[:limit] if key in self._entities]

    def _match_mention(self, mention: str) -> str | None:
        key = entity_key(mention)
        if not key:
            return None
        if key in self._alias_to_key:
            return self._alias_to_key[key]
        mention_tokens = _tokens(key)
        for row in self.store.search_entities(mention, 5):
            if mention_tokens and len(mention_tokens & _tokens(row["name"])) / len(mention_tokens) >= 0.5:
                return row["key"]
        return self._match_by_embedding(mention)

    def _entity_vectors(self):
        """Embedding tên thực thể, cache ra đĩa theo digest danh sách tên — chỉ tính lại khi đồ thị đổi."""
        import numpy as np

        keys = list(self._entities)
        names = [self._entities[key]["name"] for key in keys]
        digest = hashlib.sha1("|".join(names).encode("utf-8")).hexdigest()[:16]
        path = Path(config.ENTITY_VECTORS_FILE)
        if path.exists():
            try:
                cached = np.load(path, allow_pickle=False)
                if str(cached["digest"]) == digest:
                    return keys, cached["vectors"]
            except (OSError, ValueError, KeyError):
                pass  # cache hỏng thì tính lại
        vectors = _unit_rows(self.embedder.embed_documents(names))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, digest=np.array(digest), vectors=vectors)
        return keys, vectors

    def _match_by_embedding(self, mention: str) -> str | None:
        if self.embedder is None or not self._entities:
            return None
        if self._name_vectors is None:
            self._name_keys, self._name_vectors = self._entity_vectors()
        scores = self._name_vectors @ _unit_rows(self.embedder.embed_query(mention))
        best = int(scores.argmax())
        return self._name_keys[best] if scores[best] >= config.ENTITY_LINK_MIN_SIM else None


class GraphRAG:
    """Gọi được như một hàm: rag(question) → dict, để dùng thẳng với evaluator RAGAS."""

    def __init__(self, store, llm=None, vectorstore=None, embedder=None, communities=None):
        self.store = store
        self.llm = llm
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.communities = communities or []
        self.linker = EntityLinker(store, embedder)
        self._community_vectors = None

    def __call__(self, question: str) -> dict:
        return self.query(question)

    # ===== Phân tích câu hỏi =====
    def analyze(self, question: str) -> dict:
        """Gộp phân loại LOCAL/GLOBAL và trích thực thể vào 1 lần gọi LLM; lỗi thì quay về luật từ khoá."""
        fallback = {"type": heuristic_type(question), "entities": []}
        if self.llm is None:
            return fallback
        try:
            parsed = parse_json(call_llm(self.llm, ANALYZE_PROMPT.format(question=question), retries=1))
        except Exception:
            return fallback
        if not parsed:
            return fallback
        kind = str(parsed.get("type", "")).upper()
        raw_entities = parsed.get("entities") if isinstance(parsed.get("entities"), list) else []
        return {"type": kind if kind in ("LOCAL", "GLOBAL") else fallback["type"],
                "entities": [e for e in raw_entities if isinstance(e, str) and e.strip()]}

    # ===== Local search =====
    def retrieve(self, question: str, mentions: list[str] | None = None) -> dict:
        """Truy hồi local, không sinh câu trả lời: thực thể xuất phát → đường quan hệ → chunk bằng chứng,
        gộp với top-k vector. Đây là hàm để module hybrid search / context assembly gọi lại."""
        timings = {}
        start = time.perf_counter()
        seeds = self.linker.link(question, mentions or [])
        seed_keys = [seed["key"] for seed in seeds]
        paths = []
        if len(seed_keys) >= 2:
            paths += self.store.shortest_paths(list(combinations(seed_keys, 2)), max_hops=config.GRAPH_HOPS * 2)
        if seed_keys:
            paths += self.store.expand_paths(seed_keys, config.GRAPH_HOPS, config.MAX_NODE_DEGREE,
                                             config.PATHS_PER_SEED)
        paths = rank_paths(paths, set(seed_keys), question_type_hints(question))[:config.MAX_GRAPH_PATHS]
        graph_chunks = self._graph_chunks(paths, seed_keys)
        timings["graph"] = time.perf_counter() - start

        start = time.perf_counter()
        vector_chunks = ([to_chunk(doc) for doc in self.vectorstore.similarity_search(question, k=config.TOP_K_FINAL)]
                         if self.vectorstore is not None else [])
        timings["vector"] = time.perf_counter() - start

        facts = [path_to_text(path) for path in paths]
        budget = config.MAX_CONTEXT_CHARS - sum(len(fact) for fact in facts)
        return {"seeds": seeds, "paths": paths, "graph_facts": facts,
                "chunks": merge_chunks(vector_chunks, graph_chunks, budget), "timings": timings}

    def _graph_chunks(self, paths: list[dict], seed_keys: list[str],
                      limit: int = config.MAX_GRAPH_CHUNKS) -> list[Chunk]:
        """Chunk làm bằng chứng cho các cạnh trên đường đi, rồi tới chunk nhắc tới nhiều thực thể trên đường nhất."""
        evidence = _unique(cid for path in paths for rel in path["rels"] for cid in (rel.get("chunk_ids") or []))
        path_keys = _unique(chain(seed_keys, (node["key"] for path in paths for node in path["nodes"])))
        mentioned = ([row["id"] for row in self.store.chunks_for_entities(path_keys, limit)]
                     if path_keys and len(evidence) < limit else [])
        ids = _unique(chain(evidence[:limit], mentioned))[:limit]
        if not ids:
            return []
        rows = {row["id"]: row for row in self.store.chunks_by_ids(ids)}
        return [Chunk(row["id"], row["text"], row["source"], row["page"])
                for cid in ids if (row := rows.get(cid)) is not None]

    def _local_search(self, question: str, mentions: list[str], generate: bool, timings: dict) -> dict:
        retrieved = self.retrieve(question, mentions)
        timings.update(retrieved["timings"])
        chunks, facts = retrieved["chunks"], retrieved["graph_facts"]
        # Tách từng đường quan hệ thành một phần tử: RAGAS chấm precision/recall theo từng phần tử, gộp 15
        # đường vào một khối sẽ khiến một đường lạc kéo điểm của cả khối xuống
        contexts = [f"Quan hệ trong đồ thị tri thức: {fact}" for fact in facts] + [c.text for c in chunks]
        answer = ""
        if generate:
            start = time.perf_counter()
            answer = self._generate(ANSWER_PROMPT.format(context=build_context(facts, chunks), question=question))
            timings["generate"] = time.perf_counter() - start
        sources = _unique(source_label(c.source, c.page) for c in chunks)
        if facts:
            sources.append(f"Đồ thị tri thức: {len(facts)} đường quan hệ")
        return {"answer": answer, "sources": sources, "contexts": contexts, "search_type": "Local",
                "seed_entities": [seed["name"] for seed in retrieved["seeds"]], "graph_paths": facts,
                "subgraph": paths_to_subgraph(retrieved["paths"], {seed["key"] for seed in retrieved["seeds"]})}

    def _generate(self, prompt: str) -> str:
        if self.llm is None:
            raise RuntimeError("Cần LLM để sinh câu trả lời — kiểm tra GROQ_API_KEY trong .env")
        return call_llm(self.llm, prompt)

    # ===== Global search =====
    def rank_communities(self, question: str, top_k: int) -> list[dict]:
        """Chọn cụm tri thức liên quan nhất: cosine trên embedding tóm tắt, không có embedder thì đếm từ chung."""
        if not self.communities:
            return []
        texts = [f"{c['title']}. {c['summary']}" for c in self.communities]
        if self.embedder is not None:
            if self._community_vectors is None:
                self._community_vectors = _unit_rows(self.embedder.embed_documents(texts))
            scores = list(self._community_vectors @ _unit_rows(self.embedder.embed_query(question)))
        else:
            question_tokens = _tokens(question)
            scores = [len(question_tokens & _tokens(text + " " + " ".join(m["name"] for m in c.get("members", []))))
                      for c, text in zip(self.communities, texts)]
        order = sorted(range(len(self.communities)), key=lambda i: (-scores[i], -self.communities[i]["size"]))
        return [self.communities[i] for i in order[:top_k]]

    def _map_communities(self, question: str, communities: list[dict]) -> list[tuple[dict, dict]]:
        """Map: mỗi cụm cho một ý trả lời một phần kèm điểm hữu ích, bỏ ý dưới ngưỡng.

        Chạy song song nhưng vẫn giãn nhịp theo LLM_REQUESTS_PER_MINUTE, nếu không 5 request cùng lúc rất dễ
        dính 429. Mọi lượt gọi đều lỗi thì ném lỗi ra ngoài — lặng lẽ tụt về local search sẽ làm bảng so sánh
        ghi nhầm câu GLOBAL thành Local mà không ai biết."""
        failures = []

        def ask(community):
            prompt = MAP_PROMPT.format(title=community["title"], summary=community["summary"], question=question)
            try:
                parsed = parse_json(call_llm(self.llm, prompt, retries=2, throttle=True)) or {}
            except Exception as exc:
                failures.append(exc)
                return None
            try:
                score = float(parsed.get("score", 0))
            except (TypeError, ValueError):
                score = 0.0
            answer = str(parsed.get("answer") or "").strip()
            return {"answer": answer, "score": score} if answer and score >= config.GLOBAL_MIN_SCORE else None

        with ThreadPoolExecutor(max_workers=max(1, len(communities))) as pool:
            results = list(pool.map(ask, communities))
        if len(failures) == len(communities):
            raise RuntimeError("Global search: tất cả lượt gọi LLM ở bước map đều lỗi") from failures[-1]
        return sorted([(c, r) for c, r in zip(communities, results) if r], key=lambda item: -item[1]["score"])

    def _global_search(self, question: str, generate: bool, timings: dict) -> dict | None:
        """Trả None nếu chưa có community summaries hoặc không cụm nào liên quan → quay về local search."""
        start = time.perf_counter()
        selected = self.rank_communities(question, config.GLOBAL_TOP_COMMUNITIES)
        # Key riêng: khi global search thất bại và quay về local, timings["graph"] của bước local không xoá mất nó
        timings["communities"] = time.perf_counter() - start
        if not selected:
            return None
        answer, used = "", selected
        if generate:
            start = time.perf_counter()
            partials = self._map_communities(question, selected)
            if not partials:
                return None
            used = [community for community, _ in partials]
            partial_text = "\n\n".join(f"[C{c['id']}] (điểm {r['score']:.0f}) {r['answer']}" for c, r in partials)
            answer = self._generate(REDUCE_PROMPT.format(partials=partial_text, question=question))
            timings["generate"] = time.perf_counter() - start
        sources = ([f"Cụm tri thức C{c['id']}: {c['title']}" for c in used]
                   + _unique(source for c in used for source in c.get("sources", [])))
        return {"answer": answer, "sources": sources,
                "contexts": [f"{c['title']}: {c['summary']}" for c in used], "search_type": "Global",
                "seed_entities": [], "graph_paths": [], "subgraph": communities_to_subgraph(used),
                # False khi chạy --retrieval-only: ngữ cảnh chưa qua bộ lọc điểm của bước map, nên số đo trên
                # ngữ cảnh không đặt cạnh lần chạy có sinh câu trả lời để so sánh được
                "map_filtered": generate}

    # ===== Pipeline =====
    def query(self, question: str, generate: bool = True) -> dict:
        """Chạy GraphRAG cho 1 câu hỏi. generate=False chỉ truy hồi (không gọi LLM sinh) — để soi ngữ cảnh."""
        overall = time.perf_counter()
        timings = {}
        start = time.perf_counter()
        analysis = self.analyze(question)
        timings["analyze"] = time.perf_counter() - start

        result = self._global_search(question, generate, timings) if analysis["type"] == "GLOBAL" else None
        if result is None:
            result = self._local_search(question, analysis["entities"], generate, timings)
        result.update(method="GraphRAG", latency=round(time.perf_counter() - overall, 2),
                      timings={key: round(value, 3) for key, value in timings.items()})
        return result


def build_graph_rag(vectorstore=None, embedder=None, llm=None, store=None,
                    communities_path=config.COMMUNITIES_FILE) -> GraphRAG:
    """Tạo GraphRAG từ cấu hình .env. Truyền vectorstore/embedder của Naive RAG để 3 pipeline dùng chung index."""
    return GraphRAG(store or get_store(), llm or get_llm(), vectorstore=vectorstore, embedder=embedder,
                    communities=load_communities(communities_path))


# ===== Chạy thử & đo đạc =====
def evidence_hit(result: dict, expected: list[str] | None) -> float | None:
    """Tỉ lệ các mốc thông tin kỳ vọng (trường "graph_evidence" của bộ câu hỏi) xuất hiện trong ngữ cảnh truy hồi."""
    if not expected:
        return None
    haystack = _normalize(" ".join(result.get("contexts") or [])).lower()
    return round(sum(_normalize(name).lower() in haystack for name in expected) / len(expected), 2)


def run_eval(rag: GraphRAG, questions: list[dict], generate: bool = True) -> list[dict]:
    rows = []
    for item in tqdm(questions, desc="GraphRAG eval"):
        row = {"question": item["question"], "type": item.get("type"), "ground_truth": item.get("ground_truth")}
        try:
            result = rag.query(item["question"], generate=generate)
            row.update(result)
            row["evidence_hit"] = evidence_hit(result, item.get("graph_evidence"))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def summarize_eval(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("type") or "?"].append(row)
    summary = []
    for qtype, items in sorted(groups.items()):
        ok = [row for row in items if "error" not in row]
        hits = [row["evidence_hit"] for row in ok if row.get("evidence_hit") is not None]
        summary.append({
            "type": qtype,
            "n": len(items),
            "errors": len(items) - len(ok),
            "empty_context": sum(1 for row in ok if not row.get("contexts")),
            "global_search": sum(1 for row in ok if row.get("search_type") == "Global"),
            "avg_latency": round(mean(row["latency"] for row in ok), 2) if ok else None,
            "avg_evidence_hit": round(mean(hits), 2) if hits else None,
        })
    return summary


def print_result(result: dict):
    print(f"\n[{result['search_type']}] {result['latency']}s — " +
          ", ".join(f"{k} {v}s" for k, v in result["timings"].items()))
    if result.get("seed_entities"):
        print("Thực thể xuất phát: " + ", ".join(result["seed_entities"]))
    if result.get("graph_paths"):
        print("\n--- Đường quan hệ ---")
        for fact in result["graph_paths"]:
            print(f"- {fact}")
    print("\n--- Nguồn ---")
    for source in result["sources"]:
        print(f"- {source}")
    if result["answer"]:
        print("\n--- Câu trả lời ---\n" + result["answer"])
    else:
        print("\n--- Ngữ cảnh truy hồi ---")
        for i, context in enumerate(result["contexts"], 1):
            print(f"[{i}] {context[:200].strip()}...")


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--retrieval-only", action="store_true", help="Chỉ truy hồi, không gọi LLM sinh câu trả lời")
    common.add_argument("--faiss-dir", help="Thư mục FAISS index của Naive RAG (để dùng chung vector search)")
    parser = argparse.ArgumentParser(description="GraphRAG — hỏi đáp trên đồ thị tri thức")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", parents=[common], help="Hỏi một câu")
    ask.add_argument("question")
    evaluate = sub.add_parser("eval", parents=[common], help="Chạy cả bộ câu hỏi, ghi lại vết truy hồi và thời gian")
    evaluate.add_argument("--questions", default="data/eval/questions.json")
    evaluate.add_argument("--out", default="results/graphrag_trace.json")
    args = parser.parse_args(argv)

    vectorstore = embedder = None
    if args.faiss_dir:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS

        embedder = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
        vectorstore = FAISS.load_local(args.faiss_dir, embedder, allow_dangerous_deserialization=True)

    llm = get_llm() if config.GROQ_API_KEY else None
    if llm is None and not args.retrieval_only:
        parser.error("Thiếu GROQ_API_KEY trong .env — thêm key hoặc chạy với --retrieval-only")

    with get_store() as store:
        rag = GraphRAG(store, llm, vectorstore=vectorstore, embedder=embedder, communities=load_communities())
        if args.command == "ask":
            print_result(rag.query(args.question, generate=not args.retrieval_only))
        else:
            questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
            rows = run_eval(rag, questions, generate=not args.retrieval_only)
            summary = summarize_eval(rows)
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2, default=str),
                           encoding="utf-8")
            print(f"\nĐã ghi vết chạy vào {out}")
            for row in summary:
                print(f"  {row['type']}: n={row['n']}, lỗi={row['errors']}, ngữ cảnh rỗng={row['empty_context']}, "
                      f"global={row['global_search']}, latency={row['avg_latency']}s, "
                      f"đúng mốc={row['avg_evidence_hit']}")


if __name__ == "__main__":
    main()
