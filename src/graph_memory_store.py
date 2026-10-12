"""Backend đồ thị chạy trong RAM (NetworkX + file JSON) — cùng giao diện với Neo4jGraphStore.

Dùng khi không chạy được Neo4j (máy yếu, không cài được Docker) hoặc khi chạy test nhanh:

    GRAPH_BACKEND=memory python -m src.graph_builder sample
    GRAPH_BACKEND=memory python -m src.graph_rag ask "Tác giả của BERT làm việc ở tổ chức nào?" --retrieval-only

Neo4j vẫn là backend chính của đề tài (Cypher và giao diện đồ thị dùng để demo); file này để không ai bị chặn.
"""
import json
import re
from collections import defaultdict
from itertools import islice
from pathlib import Path

from src import config


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", str(text or "").lower()))


class InMemoryGraphStore:
    def __init__(self, path=config.GRAPH_FILE, autoload: bool = True):
        self.path = Path(path)
        self.chunks: dict[str, dict] = {}
        self.entities: dict[str, dict] = {}
        self.relations: dict[tuple, dict] = {}
        self.communities: list[dict] = []
        self._graph_cache = None
        if autoload and self.path.exists():
            self.load()

    # ===== Vòng đời =====
    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def ensure_schema(self):
        pass

    def wait_for_indexes(self, timeout_seconds: int = 60):
        pass

    def load(self):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.chunks = data.get("chunks", {})
        self.entities = data.get("entities", {})
        self.relations = {(r["source"], r["type"], r["target"]): r for r in data.get("relations", [])}
        self.communities = data.get("communities", [])
        self._graph_cache = None

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "chunks": self.chunks, "entities": self.entities,
            "relations": list(self.relations.values()), "communities": self.communities,
        }, ensure_ascii=False), encoding="utf-8")

    def clear(self):
        self.chunks, self.entities, self.relations, self.communities = {}, {}, {}, []
        self._graph_cache = None
        self.save()

    # ===== Ghi =====
    def write_graph(self, chunks: list[dict], entities: list[dict], relations: list[dict]):
        for chunk in chunks:
            self.chunks[chunk["id"]] = dict(chunk)
        for entity in entities:
            record = self.entities.setdefault(entity["key"], {})
            previous_chunks = record.get("chunk_ids", [])
            record.update(entity)
            # Neo4j dùng MERGE cho (:Entity)-[:MENTIONED_IN]->(:Chunk) nên nguồn cũ không mất khi build tiếp
            record["chunk_ids"] = sorted({cid for cid in [*previous_chunks, *entity.get("chunk_ids", [])]
                                          if cid in self.chunks})
        for relation in relations:
            if relation["source"] in self.entities and relation["target"] in self.entities:
                self.relations[(relation["source"], relation["type"], relation["target"])] = dict(relation)
        degrees = defaultdict(int)
        for source, _, target in self.relations:
            degrees[source] += 1
            degrees[target] += 1
        for key, entity in self.entities.items():
            entity["degree"] = degrees[key]
        self._graph_cache = None
        self.save()

    def set_communities(self, assignment: dict[str, int], communities: list[dict]):
        for key, entity in self.entities.items():
            entity["community"] = assignment.get(key)
        self.communities = [dict(community) for community in communities]
        self.save()

    def stats(self) -> dict:
        return {"entities": len(self.entities), "relations": len(self.relations), "chunks": len(self.chunks),
                "mentions": sum(len(e.get("chunk_ids", [])) for e in self.entities.values()),
                "communities": len(self.communities)}

    # ===== Đọc =====
    def entity_index(self) -> list[dict]:
        return [{"key": e["key"], "name": e["name"], "type": e["type"], "aliases": e.get("aliases", []),
                 "degree": e.get("degree", 0)} for e in self.entities.values()]

    def fetch_edges(self) -> list[tuple[str, str, float]]:
        return [(r["source"], r["target"], r.get("weight") or 1) for r in self.relations.values()]

    def entities_by_keys(self, keys: list[str]) -> list[dict]:
        rows = [{"key": e["key"], "name": e["name"], "type": e["type"], "degree": e.get("degree", 0)}
                for key in keys if (e := self.entities.get(key))]
        return sorted(rows, key=lambda row: (-row["degree"], row["key"]))

    def relations_among(self, keys: list[str], limit: int) -> list[dict]:
        keyset = set(keys)
        rows = [{"source": r["source"], "source_name": self.entities[r["source"]]["name"], "type": r["type"],
                 "target": r["target"], "target_name": self.entities[r["target"]]["name"],
                 "weight": r.get("weight") or 1}
                for r in self.relations.values() if r["source"] in keyset and r["target"] in keyset]
        return sorted(rows, key=lambda row: (-row["weight"], row["source"], row["target"]))[:limit]

    def search_entities(self, text: str, limit: int = 5) -> list[dict]:
        """Thay cho fulltext index của Neo4j: chấm điểm theo số từ trùng với tên và các tên gọi khác."""
        query = _tokens(text)
        if not query:
            return []
        scored = []
        for entity in self.entities.values():
            names = [entity["name"], *entity.get("aliases", [])]
            overlap = max(len(query & _tokens(name)) for name in names)
            if overlap:
                bonus = 0.5 if any(str(text).lower() in name.lower() for name in names) else 0.0
                scored.append({"key": entity["key"], "name": entity["name"], "type": entity["type"],
                               "score": overlap / len(query) + bonus})
        return sorted(scored, key=lambda row: (-row["score"], row["key"]))[:limit]

    def chunks_by_ids(self, ids: list[str]) -> list[dict]:
        return [dict(self.chunks[cid]) for cid in ids if cid in self.chunks]

    def chunks_for_entities(self, keys: list[str], limit: int) -> list[dict]:
        hits = defaultdict(int)
        for key in set(keys):
            for chunk_id in self.entities.get(key, {}).get("chunk_ids", []):
                hits[chunk_id] += 1
        ordered = sorted(hits.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return [dict(self.chunks[cid], hits=count) for cid, count in ordered if cid in self.chunks]

    # ===== Duyệt đồ thị =====
    def _graph(self):
        import networkx as nx

        if self._graph_cache is None:
            graph = nx.MultiGraph()
            graph.add_nodes_from(self.entities)
            for relation in self.relations.values():
                graph.add_edge(relation["source"], relation["target"], rel=relation)
            self._graph_cache = graph
        return self._graph_cache

    def _to_path(self, start: str, edge_path) -> dict:
        graph = self._graph()
        keys = [start] + [target for _, target, _ in edge_path]
        rels = [graph[source][target][index]["rel"] for source, target, index in edge_path]
        nodes = [{"key": key, "name": self.entities[key]["name"], "type": self.entities[key]["type"]} for key in keys]
        return {"nodes": nodes, "rels": [dict(rel) for rel in rels]}

    def expand_paths(self, keys: list[str], hops: int, max_degree: int, limit: int) -> list[dict]:
        graph = self._graph()
        hops = max(1, min(int(hops), 3))
        paths = []
        for key in keys:
            if key not in graph:
                continue
            by_length = defaultdict(list)
            self._walk(graph, key, [key], [], hops, max_degree, limit, by_length)
            for length in sorted(by_length):
                ranked = sorted(by_length[length], key=lambda item: -item[0])[:limit]
                paths.extend(path for _, path in ranked)
        return paths

    def _walk(self, graph, start, nodes, edges, hops, max_degree, limit, by_length):
        """DFS có cắt tỉa: dừng ngay khi chạm node bậc cao, giống điều kiện degree trong câu Cypher.

        Liệt kê hết rồi mới lọc như trước sẽ nổ số đường trên đồ thị thật."""
        if len(edges) >= hops or sum(map(len, by_length.values())) > limit * 50:
            return
        current = nodes[-1]
        if len(nodes) > 1 and self.entities[current].get("degree", 0) > max_degree:
            return  # không đi xuyên hub (hub vẫn được phép là điểm cuối)
        for neighbor in graph[current]:
            if neighbor in nodes:
                continue
            for index in graph[current][neighbor]:
                edge_path = edges + [(current, neighbor, index)]
                path = self._to_path(start, edge_path)
                by_length[len(edge_path)].append((sum(rel.get("weight") or 1 for rel in path["rels"]), path))
                self._walk(graph, start, nodes + [neighbor], edge_path, hops, max_degree, limit, by_length)

    def shortest_paths(self, pairs, max_hops: int, limit: int = 3) -> list[dict]:
        import networkx as nx

        graph = self._graph()
        paths = []
        for source, target in pairs:
            if source not in graph or target not in graph:
                continue
            try:
                # islice: chỉ dựng đúng `limit` đường thay vì liệt kê toàn bộ đường ngắn nhất
                for node_path in islice(nx.all_shortest_paths(graph, source, target), limit):
                    if len(node_path) - 1 > max_hops:
                        break
                    edge_path = [(u, v, max(graph[u][v], key=lambda i: graph[u][v][i]["rel"].get("weight") or 1))
                                 for u, v in zip(node_path, node_path[1:])]
                    paths.append(self._to_path(source, edge_path))
            except nx.NetworkXNoPath:
                continue
        return paths
