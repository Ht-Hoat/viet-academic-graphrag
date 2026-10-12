"""Luồng offline của GraphRAG: trích xuất thực thể–quan hệ → Knowledge Graph (Neo4j) → Leiden → tóm tắt cộng đồng.

Chạy từ thư mục gốc repo:
    python -m src.graph_builder sample                                  # nạp đồ thị mẫu, không cần LLM
    python -m src.graph_builder build --chunks data/processed/chunks.jsonl --limit 20
    python -m src.graph_builder build --pdf-dir data/pdfs               # dùng load_and_chunk của naive_rag
    python -m src.graph_builder communities [--no-llm]
    python -m src.graph_builder stats
"""
import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from src import config
from src.llm_utils import call_llm, get_llm, parse_json

PROMPT_VERSION = "v1"
MAX_NAME_LEN = 120

TYPE_ALIASES = {
    "ORGANIZATION": "ORG", "ORGANISATION": "ORG", "INSTITUTION": "ORG",
    "AUTHOR": "PERSON",
    "BOOK": "PAPER", "ARTICLE": "PAPER", "PUBLICATION": "PAPER",
    "MODEL": "METHOD", "ALGORITHM": "METHOD", "TECHNIQUE": "METHOD", "ARCHITECTURE": "METHOD",
    "PROBLEM": "TASK", "APPLICATION": "TASK",
    "METRIC": "DATASET", "BENCHMARK": "DATASET",
}
RELATION_ALIASES = {
    "is_a": "là_một", "part_of": "thành_phần_của",
    "uses": "sử_dụng", "dựa_trên": "sử_dụng", "based_on": "sử_dụng",
    "proposes": "đề_xuất", "giới_thiệu": "đề_xuất",
    "author_of": "tác_giả_của", "viết": "tác_giả_của",
    "belongs_to": "thuộc_về", "từ": "thuộc_về", "làm_việc_tại": "thuộc_về",
    "improves": "cải_tiến", "mở_rộng": "cải_tiến",
    "applied_to": "giải_quyết", "áp_dụng_cho": "giải_quyết",
    "evaluated_on": "đánh_giá_trên", "compared_with": "so_sánh_với", "related_to": "liên_quan_đến",
}
# Nhãn bị động: đảo chiều source/target rồi dùng nhãn chủ động
REVERSE_RELATION_ALIASES = {
    "được_đề_xuất_bởi": "đề_xuất", "đề_xuất_bởi": "đề_xuất", "proposed_by": "đề_xuất",
    "viết_bởi": "tác_giả_của", "written_by": "tác_giả_của",
    "được_sử_dụng_bởi": "sử_dụng", "used_by": "sử_dụng",
    "bao_gồm": "thành_phần_của", "has_part": "thành_phần_của",
    "được_cải_tiến_bởi": "cải_tiến",
}
# Từ chung chung đứng trước tên riêng: "mô hình BERT" → "BERT"
GENERIC_PREFIXES = ("mô hình ", "phương pháp ", "thuật toán ", "kỹ thuật ", "kiến trúc ", "bài báo ",
                    "khái niệm ", "tổ chức ", "bộ dữ liệu ")

EXTRACT_PROMPT = """Bạn là chuyên gia xây dựng đồ thị tri thức từ tài liệu học thuật tiếng Việt.
Từ đoạn văn dưới đây, hãy trích xuất các thực thể quan trọng và quan hệ giữa chúng.

Loại thực thể được phép:
{entity_types}

Loại quan hệ được phép (hướng: source → target):
{relation_types}

Quy tắc:
- Chỉ trích xuất thông tin có trong đoạn văn, không suy diễn.
- Giữ nguyên tên riêng / thuật ngữ như trong văn bản; bỏ các từ chung như "mô hình", "phương pháp" đứng trước tên riêng.
- "source" và "target" của quan hệ phải trùng tên với một thực thể trong "entities".
- Tối đa 15 thực thể và 20 quan hệ.

Ví dụ:
Đoạn văn: "BERT do nhóm Devlin et al. 2018 tại Google đề xuất, dựa trên kiến trúc Transformer."
Kết quả:
{{"entities": [{{"name": "BERT", "type": "METHOD"}}, {{"name": "Devlin et al. 2018", "type": "PAPER"}}, {{"name": "Google", "type": "ORG"}}, {{"name": "Transformer", "type": "METHOD"}}],
 "relations": [{{"source": "Devlin et al. 2018", "relation": "đề_xuất", "target": "BERT"}}, {{"source": "Devlin et al. 2018", "relation": "thuộc_về", "target": "Google"}}, {{"source": "BERT", "relation": "sử_dụng", "target": "Transformer"}}]}}

Đoạn văn:
\"\"\"
{text}
\"\"\"

Chỉ trả về một object JSON đúng định dạng trên, không giải thích thêm."""

COMMUNITY_PROMPT = """Bạn đang viết báo cáo tóm tắt cho một cụm tri thức trích từ kho tài liệu học thuật.

Các thực thể trong cụm:
{entities}

Các quan hệ:
{relations}

Trích đoạn tài liệu liên quan:
{excerpts}

Chỉ dựa trên thông tin trên, viết bằng tiếng Việt và trả về JSON:
{{"title": "tên chủ đề của cụm, tối đa 10 từ", "summary": "3-6 câu: chủ đề chính, các thực thể quan trọng và mối liên hệ giữa chúng"}}"""


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int | str | None = None


def chunk_id_for(text: str, source: str, page) -> str:
    """Id ổn định theo nội dung — Naive RAG và GraphRAG tính ra cùng id cho cùng một chunk."""
    return hashlib.sha1(f"{source}|{page}|{text}".encode("utf-8")).hexdigest()[:16]


def to_chunk(obj) -> Chunk:
    """Nhận LangChain Document hoặc dict {text, source, page[, chunk_id]} → Chunk."""
    if isinstance(obj, Chunk):
        return obj
    if isinstance(obj, dict):
        text = obj.get("text") or obj.get("page_content") or ""
        meta = obj.get("metadata") or obj
    else:
        text, meta = obj.page_content, obj.metadata
    # NFC ngay từ đầu: tên thực thể cũng được NFC hoá nên mới dò được trong text, và id không đổi theo cách gõ dấu
    text = unicodedata.normalize("NFC", text)
    source = str(meta.get("source", "?"))
    page = meta.get("page")
    return Chunk(str(meta.get("chunk_id") or chunk_id_for(text, source, page)), text, source, page)


# ===== Chuẩn hoá =====
def clean_name(name) -> str:
    """NFC, gộp khoảng trắng, bỏ dấu câu bao quanh, bỏ tiền tố chung chung đứng trước tên riêng."""
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFC", name)
    name = re.sub(r"\s+", " ", name).strip().strip("\"'“”‘’`.,;:()[]{}").strip()
    stripped = True
    while stripped:  # "phương pháp Mô hình BERT" → "Mô hình BERT" → "BERT"
        stripped = False
        for prefix in GENERIC_PREFIXES:
            rest = name[len(prefix):]
            if name[:len(prefix)].lower() == prefix and rest[:1] and (rest[:1].isupper() or rest[:1].isdigit()):
                name, stripped = rest, True
                break
    return name


def entity_key(name) -> str:
    return clean_name(name).lower()


def normalize_type(value) -> str:
    value = re.sub(r"[\s-]+", "_", str(value or "").strip().upper())
    value = TYPE_ALIASES.get(value, value)
    return value if value in config.ENTITY_TYPES else "CONCEPT"


def normalize_relation(label) -> tuple[str, bool]:
    """Trả (nhãn chuẩn, có_đảo_chiều). Nhãn lạ → 'liên_quan_đến'."""
    label = re.sub(r"[\s-]+", "_", unicodedata.normalize("NFC", str(label or "")).strip().lower())
    if label in REVERSE_RELATION_ALIASES:
        return REVERSE_RELATION_ALIASES[label], True
    label = RELATION_ALIASES.get(label, label)
    return (label if label in config.RELATION_TYPES else "liên_quan_đến"), False


def clean_extraction(raw) -> dict:
    """Làm sạch output LLM: bỏ mục sai định dạng, chuẩn hoá loại/quan hệ, tự thêm thực thể mà quan hệ nhắc tới."""
    raw = raw if isinstance(raw, dict) else {}
    raw_entities = raw.get("entities") if isinstance(raw.get("entities"), list) else []
    raw_relations = raw.get("relations") if isinstance(raw.get("relations"), list) else []
    entities: dict[str, dict] = {}
    relations, dropped = [], 0

    for item in raw_entities:
        name = clean_name(item.get("name")) if isinstance(item, dict) else ""
        if name and len(name) <= MAX_NAME_LEN:
            entities.setdefault(name.lower(), {"name": name, "type": normalize_type(item.get("type")), "inferred": False})

    for item in raw_relations:
        if not isinstance(item, dict):
            dropped += 1
            continue
        source, target = clean_name(item.get("source")), clean_name(item.get("target"))
        rel_type, reverse = normalize_relation(item.get("relation"))
        if not source or not target or source.lower() == target.lower() or max(len(source), len(target)) > MAX_NAME_LEN:
            dropped += 1
            continue
        if reverse:
            source, target = target, source
        for name in (source, target):
            entities.setdefault(name.lower(), {"name": name, "type": "CONCEPT", "inferred": True})
        relations.append({"source": source, "relation": rel_type, "target": target})

    return {"entities": list(entities.values()), "relations": relations,
            "raw_relations": len(raw_relations), "dropped_relations": dropped}


# ===== Trích xuất =====
def group_chunks(chunks: list[Chunk], max_chars: int = config.EXTRACT_UNIT_CHARS) -> list[list[Chunk]]:
    """Gộp các chunk liền nhau cùng file thành đơn vị trích xuất ≤ max_chars để giảm số lần gọi LLM."""
    units, current, size = [], [], 0
    for chunk in chunks:
        if current and (current[-1].source != chunk.source or size + len(chunk.text) > max_chars):
            units.append(current)
            current, size = [], 0
        current.append(chunk)
        size += len(chunk.text)
    if current:
        units.append(current)
    return units


def unit_id_for(unit: list[Chunk], model: str) -> str:
    key = "|".join(c.id for c in unit) + f"|{PROMPT_VERSION}|{model}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


class ExtractionCache:
    """Cache kết quả trích xuất theo từng đơn vị (JSONL ghi nối tiếp) — chạy lại không tốn thêm quota."""

    def __init__(self, path: Path | str = config.EXTRACTION_CACHE):
        self.path = Path(path)
        self.items: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # dòng ghi dở khi bị ngắt giữa chừng
                self.items[row["unit_id"]] = row["result"]

    def get(self, unit_id: str) -> dict | None:
        return self.items.get(unit_id)

    def put(self, unit_id: str, chunk_ids: list[str], result: dict):
        self.items[unit_id] = result
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"unit_id": unit_id, "chunk_ids": chunk_ids, "result": result}, ensure_ascii=False) + "\n")


def build_extract_prompt(text: str) -> str:
    return EXTRACT_PROMPT.format(
        entity_types="\n".join(f"- {name}: {desc}" for name, desc in config.ENTITY_TYPES.items()),
        relation_types="\n".join(f"- {name}: {desc}" for name, desc in config.RELATION_TYPES.items()),
        text=text,
    )


def extract_units(units: list[list[Chunk]], llm, cache: ExtractionCache, model: str) -> list[tuple[list[Chunk], dict | None]]:
    """Trích xuất từng đơn vị (dùng cache nếu có). Lỗi gọi LLM → None (không cache, lần sau chạy lại);
    output không phải JSON → cache {"error": "invalid_json"} để không tốn quota lặp lại."""
    results = []
    for unit in tqdm(units, desc="Trích xuất thực thể"):
        unit_id = unit_id_for(unit, model)
        result = cache.get(unit_id)
        if result is None:
            try:
                content = call_llm(llm, build_extract_prompt("\n\n".join(c.text for c in unit)), throttle=True)
            except Exception as exc:
                print(f"Lỗi gọi LLM ở đơn vị {unit_id}: {exc}", file=sys.stderr)
                results.append((unit, None))
                continue
            result = parse_json(content) or {"error": "invalid_json"}
            cache.put(unit_id, [c.id for c in unit], result)
        results.append((unit, result))
    return results


# ===== Gộp thành đồ thị =====
def _mentioning(unit: list[Chunk], lowered_texts: list[str], *names: str) -> list[str]:
    return [c.id for c, text in zip(unit, lowered_texts) if all(n.lower() in text for n in names)]


def aggregate(unit_results: list[tuple[list[Chunk], dict | None]]) -> dict:
    """Gộp kết quả mọi đơn vị: gộp thực thể trùng key, cộng trọng số quan hệ, gắn chunk nguồn cho thực thể và cạnh.

    Trả {"chunks", "entities", "relations", "stats"} ở dạng sẵn sàng ghi Neo4j.
    """
    chunks: dict[str, Chunk] = {}
    entity_chunks, name_votes, type_votes = defaultdict(set), defaultdict(Counter), defaultdict(Counter)
    relations: dict[tuple, dict] = {}
    stats = Counter(units=len(unit_results))

    for unit, raw in unit_results:
        for chunk in unit:
            chunks[chunk.id] = chunk
        if raw is None or raw.get("error"):
            stats["failed_units"] += 1
            continue
        data = clean_extraction(raw)
        stats["raw_relations"] += data["raw_relations"]
        stats["dropped_relations"] += data["dropped_relations"]
        lowered = [c.text.lower() for c in unit]
        all_ids = [c.id for c in unit]

        for ent in data["entities"]:
            key = entity_key(ent["name"])
            name_votes[key][ent["name"]] += 1
            if not ent["inferred"]:
                type_votes[key][ent["type"]] += 1
            entity_chunks[key].update(_mentioning(unit, lowered, ent["name"]) or all_ids)

        for rel in data["relations"]:
            triple = (entity_key(rel["source"]), rel["relation"], entity_key(rel["target"]))
            record = relations.setdefault(triple, {"weight": 0, "chunk_ids": set()})
            record["weight"] += 1
            record["chunk_ids"].update(
                _mentioning(unit, lowered, rel["source"], rel["target"])
                or _mentioning(unit, lowered, rel["source"]) + _mentioning(unit, lowered, rel["target"])
                or all_ids
            )

    entities = [{
        "key": key,
        "name": votes.most_common(1)[0][0],
        "type": type_votes[key].most_common(1)[0][0] if type_votes[key] else "CONCEPT",
        "aliases": sorted(votes),
        "mentions": sum(votes.values()),
        "chunk_ids": sorted(entity_chunks[key]),
    } for key, votes in name_votes.items()]
    relation_rows = [{"source": s, "type": t, "target": d, "weight": r["weight"], "chunk_ids": sorted(r["chunk_ids"])}
                     for (s, t, d), r in relations.items()]
    chunk_rows = [{"id": c.id, "text": c.text, "source": c.source, "page": c.page} for c in chunks.values()]

    stats.update(entities=len(entities), relations=len(relation_rows), chunks=len(chunk_rows))
    stats["drop_rate"] = round(stats["dropped_relations"] / stats["raw_relations"], 3) if stats["raw_relations"] else 0.0
    return {"chunks": chunk_rows, "entities": entities, "relations": relation_rows, "stats": dict(stats)}


def build_knowledge_graph(chunks: list, store, llm=None, reset: bool = True,
                          cache_path: Path | str = config.EXTRACTION_CACHE) -> dict:
    """Trích xuất toàn bộ chunks (qua cache) và ghi vào Neo4j. Trả thống kê.

    reset=True xoá đồ thị GraphRAG cũ trước khi ghi — an toàn vì kết quả trích xuất đã nằm trong cache.
    """
    chunk_list = [to_chunk(c) for c in chunks]
    llm = llm or get_llm(config.EXTRACT_MODEL, temperature=0)
    model = getattr(llm, "model_name", None) or config.EXTRACT_MODEL
    unit_results = extract_units(group_chunks(chunk_list), llm, ExtractionCache(cache_path), model)
    graph = aggregate(unit_results)
    if reset:
        store.clear()
    store.write_graph(graph["chunks"], graph["entities"], graph["relations"])
    return graph["stats"]


# ===== Phân cụm & tóm tắt cộng đồng =====
def detect_communities(entity_keys: list[str], edges: list[tuple[str, str, float]],
                       resolution: float = config.LEIDEN_RESOLUTION,
                       min_size: int = config.MIN_COMMUNITY_SIZE, seed: int = 42) -> dict[str, int]:
    """Leiden (tối đa hoá modularity) trên đồ thị vô hướng có trọng số.

    Trả entity key → community id; chỉ giữ cộng đồng ≥ min_size, id đánh số theo kích thước giảm dần.
    """
    import igraph as ig
    import leidenalg

    if not entity_keys:
        return {}
    index = {key: i for i, key in enumerate(entity_keys)}
    pairs, weights = [], []
    for source, target, weight in edges:
        if source in index and target in index and source != target:
            pairs.append((index[source], index[target]))
            weights.append(float(weight or 1))
    graph = ig.Graph(n=len(entity_keys), edges=pairs, directed=False)
    graph.es["weight"] = weights
    graph.simplify(combine_edges={"weight": "sum"})
    partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, weights="weight",
                                         resolution_parameter=resolution, seed=seed)
    groups = sorted((sorted(group) for group in partition if len(group) >= min_size), key=lambda g: (-len(g), g))
    return {entity_keys[v]: cid for cid, group in enumerate(groups) for v in group}


def source_label(source, page) -> str:
    return f"{Path(str(source)).name} — trang {page}" if page is not None else Path(str(source)).name


def fallback_summary(members: list[dict], relations: list[dict]) -> dict:
    """Tóm tắt không cần LLM: liệt kê thực thể chính và quan hệ — dùng khi LLM lỗi hoặc chạy --no-llm."""
    title = ", ".join(m["name"] for m in members[:3])
    lines = [f"{r['source_name']} {r['type'].replace('_', ' ')} {r['target_name']}" for r in relations[:10]]
    summary = f"Cụm gồm {len(members)} thực thể, nổi bật: {', '.join(m['name'] for m in members[:8])}."
    if lines:
        summary += " Quan hệ chính: " + "; ".join(lines) + "."
    return {"title": title, "summary": summary}


def build_communities(store, llm=None, path: Path | str = config.COMMUNITIES_FILE,
                      max_entities: int = 30, max_relations: int = 40, max_excerpts: int = 3) -> list[dict]:
    """Chạy Leiden, tóm tắt từng cộng đồng bằng LLM (tái dùng tóm tắt cũ nếu nội dung cụm không đổi), lưu JSON + Neo4j."""
    path = Path(path)
    previous = {}
    if path.exists():
        previous = {c["content_hash"]: c for c in json.loads(path.read_text(encoding="utf-8"))}

    entity_keys = [row["key"] for row in store.entity_index()]
    assignment = detect_communities(entity_keys, store.fetch_edges())
    groups = defaultdict(list)
    for key, cid in assignment.items():
        groups[cid].append(key)

    communities = []
    for cid in tqdm(sorted(groups), desc="Tóm tắt cộng đồng"):
        member_keys = sorted(groups[cid])
        members = store.entities_by_keys(member_keys)[:max_entities]
        relations = store.relations_among(member_keys, max_relations)
        excerpts = store.chunks_for_entities(member_keys, max_excerpts)
        content_hash = hashlib.sha1(json.dumps(
            [member_keys, [(r["source"], r["type"], r["target"]) for r in relations]], ensure_ascii=False
        ).encode("utf-8")).hexdigest()[:16]

        report = previous.get(content_hash)
        if report is not None and llm is not None and report.get("generated_by") != "llm":
            report = None  # tóm tắt cũ là bản dự phòng (--no-llm) → tạo lại bằng LLM
        if report is None:
            report = dict(fallback_summary(members, relations), generated_by="fallback")
            if llm is not None:
                prompt = COMMUNITY_PROMPT.format(
                    entities="\n".join(f"- {m['name']} ({m['type']})" for m in members),
                    relations="\n".join(f"- {r['source_name']} --({r['type']})--> {r['target_name']}" for r in relations) or "(không có)",
                    excerpts="\n\n".join(f"[{source_label(e['source'], e['page'])}] {e['text'][:500]}" for e in excerpts),
                )
                try:
                    parsed = parse_json(call_llm(llm, prompt, throttle=True)) or {}
                    if parsed.get("summary"):
                        report = {"title": str(parsed.get("title") or report["title"]),
                                  "summary": str(parsed["summary"]), "generated_by": "llm"}
                except Exception as exc:
                    print(f"Lỗi tóm tắt cộng đồng {cid}, dùng tóm tắt dự phòng: {exc}", file=sys.stderr)

        communities.append({
            "id": cid,
            "title": report["title"],
            "summary": report["summary"],
            "generated_by": report.get("generated_by", "fallback"),
            "size": len(member_keys),
            "member_keys": member_keys,
            "members": [{"key": m["key"], "name": m["name"], "type": m["type"]} for m in members],
            "relations": [{"source": r["source"], "type": r["type"], "target": r["target"]} for r in relations],
            "sources": sorted({source_label(e["source"], e["page"]) for e in excerpts}),
            "content_hash": content_hash,
        })

    store.set_communities(assignment, communities)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(communities, ensure_ascii=False, indent=2), encoding="utf-8")
    return communities


def load_communities(path: Path | str = config.COMMUNITIES_FILE) -> list[dict]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


# ===== Nguồn chunks =====
def load_chunks_jsonl(path: Path | str) -> list[Chunk]:
    """Đọc chunks từ JSONL — mỗi dòng {text, source, page[, chunk_id]}. Định dạng trao đổi với module chunking."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [to_chunk(json.loads(line)) for line in lines if line.strip()]


def save_chunks_jsonl(chunks: list, path: Path | str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in map(to_chunk, chunks):
            f.write(json.dumps({"chunk_id": chunk.id, "text": chunk.text, "source": chunk.source, "page": chunk.page},
                               ensure_ascii=False) + "\n")


def _print_stats(title: str, stats: dict):
    print(f"\n=== {title} ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Dựng Knowledge Graph cho GraphRAG")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sample", help="Nạp đồ thị mẫu BERT/GPT (không cần LLM)")
    build = sub.add_parser("build", help="Trích xuất thực thể từ chunks và ghi vào Neo4j")
    source = build.add_mutually_exclusive_group(required=True)
    source.add_argument("--chunks", help="File JSONL chunks {text, source, page[, chunk_id]}")
    source.add_argument("--pdf-dir", help="Thư mục PDF (dùng load_and_chunk của src.naive_rag)")
    build.add_argument("--limit", type=int, help="Chỉ xử lý N chunk đầu (chạy thử)")
    build.add_argument("--no-reset", action="store_true", help="Không xoá đồ thị cũ trước khi ghi")
    communities = sub.add_parser("communities", help="Chạy Leiden + tóm tắt cộng đồng")
    communities.add_argument("--no-llm", action="store_true", help="Dùng tóm tắt dự phòng, không gọi LLM")
    sub.add_parser("stats", help="Thống kê đồ thị trong Neo4j")
    args = parser.parse_args(argv)

    from src.graph_store import get_store

    with get_store() as store:
        if args.command == "sample":
            from src.graph_sample import sample_unit_results

            graph = aggregate(sample_unit_results())
            store.clear()
            store.write_graph(graph["chunks"], graph["entities"], graph["relations"])
            _print_stats("Đồ thị mẫu", graph["stats"])
        elif args.command == "build":
            if args.chunks:
                chunks = load_chunks_jsonl(args.chunks)
            else:
                try:
                    from src.naive_rag import load_and_chunk
                except ImportError:
                    parser.error("Chưa có src/naive_rag.py (module chunking của Hoạt). Tạm dùng "
                                 "--chunks data/processed/chunks.jsonl (xuất bằng save_chunks_jsonl).")
                chunks = load_and_chunk(args.pdf_dir)
            chunks = chunks[:args.limit] if args.limit else chunks
            _print_stats("Kết quả trích xuất", build_knowledge_graph(chunks, store, reset=not args.no_reset))
        elif args.command == "communities":
            llm = None if args.no_llm else get_llm(config.LLM_MODEL, temperature=0)
            result = build_communities(store, llm)
            print(f"\nĐã lưu {len(result)} cộng đồng vào {config.COMMUNITIES_FILE}")
            for c in result:
                print(f"  C{c['id']} ({c['size']} thực thể): {c['title']}")
        _print_stats(f"Đồ thị ({config.GRAPH_BACKEND})", store.stats())


if __name__ == "__main__":
    main()
