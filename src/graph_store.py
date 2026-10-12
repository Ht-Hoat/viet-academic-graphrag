"""Lớp truy cập Neo4j cho GraphRAG — toàn bộ Cypher nằm ở đây.

Schema:
    (:Chunk {id, text, source, page})
    (:Entity {key, name, type, aliases, aliases_text, mentions, degree, community})
    (:Entity)-[:REL {type, weight, chunk_ids}]->(:Entity)
    (:Entity)-[:MENTIONED_IN]->(:Chunk)
    (:Entity)-[:IN_COMMUNITY]->(:Community {id, title, summary, size})

Yêu cầu Neo4j >= 5.23 (cú pháp CALL (x) { ... }).
"""
import re

from neo4j import GraphDatabase

from src import config

BATCH_SIZE = 500       # số dòng mỗi lần UNWIND
TX_BATCH_SIZE = 10000  # số dòng mỗi giao dịch con của CALL { ... } IN TRANSACTIONS
_LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')

_PATH_RETURN = """
RETURN [n IN nodes(p) | n {.key, .name, .type}] AS nodes,
       [r IN relationships(p) | {type: r.type, source: startNode(r).key, target: endNode(r).key,
                                 weight: r.weight, chunk_ids: r.chunk_ids}] AS rels
"""


def get_store(backend: str | None = None):
    """Tạo store theo cấu hình: "neo4j" (mặc định) hoặc "memory" (NetworkX + JSON, xem graph_memory_store.py)."""
    backend = (backend or config.GRAPH_BACKEND).lower()
    if backend == "memory":
        from src.graph_memory_store import InMemoryGraphStore

        return InMemoryGraphStore()
    if backend != "neo4j":
        raise ValueError(f"GRAPH_BACKEND không hợp lệ: {backend} (chỉ nhận 'neo4j' hoặc 'memory')")
    return Neo4jGraphStore()


def _batches(rows: list, size: int = BATCH_SIZE):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


class Neo4jGraphStore:
    def __init__(self, uri=None, user=None, password=None, database=None, driver=None):
        self.driver = driver or GraphDatabase.driver(
            uri or config.NEO4J_URI,
            auth=(user or config.NEO4J_USER, password or config.NEO4J_PASSWORD),
        )
        self.database = database or config.NEO4J_DATABASE

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def run(self, cypher: str, **params) -> list[dict]:
        # Tên tham số là "cypher" để không đụng với tham số Cypher tên $query
        result = self.driver.execute_query(cypher, parameters_=params, database_=self.database)
        return [record.data() for record in result.records]

    # ===== Dựng đồ thị =====
    def ensure_schema(self):
        self.run("CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE")
        self.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        self.run("CREATE CONSTRAINT community_id IF NOT EXISTS FOR (c:Community) REQUIRE c.id IS UNIQUE")
        self.run("CREATE FULLTEXT INDEX entity_name_ft IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.aliases_text]")

    def run_autocommit(self, cypher: str, **params):
        """Chạy ở chế độ auto-commit — bắt buộc với CALL { ... } IN TRANSACTIONS (execute_query không cho phép)."""
        with self.driver.session(database=self.database) as session:
            return session.run(cypher, params).consume()

    def wait_for_indexes(self, timeout_seconds: int = 60):
        self.run("CALL db.awaitIndexes($timeout)", timeout=timeout_seconds)

    def clear(self):
        """Chỉ xoá dữ liệu của GraphRAG (Entity, Chunk, Community), không đụng tới node khác trong DB.

        Chia nhỏ giao dịch để đồ thị vài nghìn node không làm vỡ heap 1 GB của container."""
        self.run_autocommit(f"""
            MATCH (n) WHERE n:Entity OR n:Chunk OR n:Community
            CALL (n) {{ DETACH DELETE n }} IN TRANSACTIONS OF {TX_BATCH_SIZE} ROWS
        """)

    def write_graph(self, chunks: list[dict], entities: list[dict], relations: list[dict]):
        """Ghi theo lô bằng UNWIND. chunks: {id, text, source, page}; entities: {key, name, type, aliases,
        mentions, chunk_ids}; relations: {source, target, type, weight, chunk_ids} (source/target là key)."""
        self.ensure_schema()
        for rows in _batches(chunks):
            self.run("""
                UNWIND $rows AS row
                MERGE (c:Chunk {id: row.id})
                SET c.text = row.text, c.source = row.source, c.page = row.page
            """, rows=rows)
        for rows in _batches(entities):
            self.run("""
                UNWIND $rows AS row
                MERGE (e:Entity {key: row.key})
                SET e.name = row.name, e.type = row.type, e.aliases = row.aliases,
                    e.aliases_text = reduce(s = '', a IN row.aliases | s + ' | ' + a), e.mentions = row.mentions
                WITH e, row
                UNWIND row.chunk_ids AS cid
                MATCH (c:Chunk {id: cid})
                MERGE (e)-[:MENTIONED_IN]->(c)
            """, rows=rows)
        for rows in _batches(relations):
            self.run("""
                UNWIND $rows AS row
                MATCH (a:Entity {key: row.source}), (b:Entity {key: row.target})
                MERGE (a)-[r:REL {type: row.type}]->(b)
                SET r.weight = row.weight, r.chunk_ids = row.chunk_ids
            """, rows=rows)
        self.run_autocommit(f"""
            MATCH (e:Entity)
            CALL (e) {{ SET e.degree = COUNT {{ (e)-[:REL]-() }} }} IN TRANSACTIONS OF {TX_BATCH_SIZE} ROWS
        """)
        self.wait_for_indexes()

    def set_communities(self, assignment: dict[str, int], communities: list[dict]):
        """assignment: entity key → community id; communities: {id, title, summary, size, member_keys}."""
        self.run_autocommit(f"""
            MATCH (c:Community)
            CALL (c) {{ DETACH DELETE c }} IN TRANSACTIONS OF {TX_BATCH_SIZE} ROWS
        """)
        self.run_autocommit(f"""
            MATCH (e:Entity) WHERE e.community IS NOT NULL
            CALL (e) {{ REMOVE e.community }} IN TRANSACTIONS OF {TX_BATCH_SIZE} ROWS
        """)
        rows = [{"key": key, "community": cid} for key, cid in assignment.items()]
        for batch in _batches(rows):
            self.run("UNWIND $rows AS row MATCH (e:Entity {key: row.key}) SET e.community = row.community", rows=batch)
        self.run("""
            UNWIND $rows AS row
            CREATE (c:Community {id: row.id, title: row.title, summary: row.summary, size: row.size})
            WITH c, row
            UNWIND row.member_keys AS key
            MATCH (e:Entity {key: key})
            MERGE (e)-[:IN_COMMUNITY]->(c)
        """, rows=communities)

    def stats(self) -> dict:
        return self.run("""
            CALL () { MATCH (e:Entity) RETURN count(e) AS entities }
            CALL () { MATCH ()-[r:REL]->() RETURN count(r) AS relations }
            CALL () { MATCH (c:Chunk) RETURN count(c) AS chunks }
            CALL () { MATCH ()-[m:MENTIONED_IN]->() RETURN count(m) AS mentions }
            CALL () { MATCH (c:Community) RETURN count(c) AS communities }
            RETURN entities, relations, chunks, mentions, communities
        """)[0]

    # ===== Đọc đồ thị =====
    def entity_index(self) -> list[dict]:
        return self.run("""
            MATCH (e:Entity)
            RETURN e.key AS key, e.name AS name, e.type AS type, e.aliases AS aliases, e.degree AS degree
        """)

    def fetch_edges(self) -> list[tuple[str, str, float]]:
        rows = self.run("MATCH (a:Entity)-[r:REL]->(b:Entity) RETURN a.key AS source, b.key AS target, r.weight AS weight")
        return [(row["source"], row["target"], row["weight"] or 1) for row in rows]

    def entities_by_keys(self, keys: list[str]) -> list[dict]:
        return self.run("""
            MATCH (e:Entity) WHERE e.key IN $keys
            RETURN e.key AS key, e.name AS name, e.type AS type, e.degree AS degree
            ORDER BY e.degree DESC, e.key
        """, keys=keys)

    def relations_among(self, keys: list[str], limit: int) -> list[dict]:
        return self.run("""
            MATCH (a:Entity)-[r:REL]->(b:Entity) WHERE a.key IN $keys AND b.key IN $keys
            RETURN a.key AS source, a.name AS source_name, r.type AS type, b.key AS target, b.name AS target_name,
                   r.weight AS weight
            ORDER BY r.weight DESC, a.key, b.key
            LIMIT $limit
        """, keys=keys, limit=limit)

    def search_entities(self, text: str, limit: int = 5) -> list[dict]:
        """Tìm thực thể theo fulltext index trên tên + tên gọi khác."""
        query = _LUCENE_SPECIAL.sub(r"\\\1", text).strip()
        if not query:
            return []
        return self.run("""
            CALL db.index.fulltext.queryNodes('entity_name_ft', $query) YIELD node, score
            RETURN node.key AS key, node.name AS name, node.type AS type, score
            LIMIT $limit
        """, query=query, limit=limit)

    def expand_paths(self, keys: list[str], hops: int, max_degree: int, limit: int) -> list[dict]:
        """Đường đi 1..hops bước từ các thực thể xuất phát. Mỗi độ dài lấy tối đa `limit` đường (ưu tiên cạnh
        có trọng số cao) để đường 2 bước không bị các láng giềng 1 bước chiếm hết chỗ. Không đi xuyên hub."""
        hops = max(1, min(int(hops), 3))
        branches = "\nUNION\n".join(f"""
            MATCH p = (s)-[:REL*{h}..{h}]-(t:Entity)
            WHERE t <> s AND all(n IN nodes(p)[1..-1] WHERE n.degree <= $max_degree)
            RETURN p, reduce(w = 0.0, r IN relationships(p) | w + coalesce(r.weight, 1)) AS weight
            ORDER BY weight DESC
            LIMIT $limit""" for h in range(1, hops + 1))
        return self.run(f"""
            UNWIND $keys AS key
            MATCH (s:Entity {{key: key}})
            CALL (s) {{ {branches} }}
            {_PATH_RETURN}
        """, keys=keys, max_degree=max_degree, limit=limit)

    def shortest_paths(self, pairs: list[tuple[str, str]], max_hops: int, limit: int = 3) -> list[dict]:
        """Các đường ngắn nhất nối từng cặp thực thể — lõi của câu hỏi đa bước."""
        max_hops = max(1, min(int(max_hops), 6))
        return self.run(f"""
            UNWIND $pairs AS pair
            MATCH (a:Entity {{key: pair[0]}}), (b:Entity {{key: pair[1]}})
            CALL (a, b) {{
                MATCH p = allShortestPaths((a)-[:REL*..{max_hops}]-(b))
                RETURN p LIMIT $limit
            }}
            {_PATH_RETURN}
        """, pairs=[list(pair) for pair in pairs], limit=limit)

    def chunks_by_ids(self, ids: list[str]) -> list[dict]:
        return self.run("""
            MATCH (c:Chunk) WHERE c.id IN $ids
            RETURN c.id AS id, c.text AS text, c.source AS source, c.page AS page
        """, ids=ids)

    def chunks_for_entities(self, keys: list[str], limit: int) -> list[dict]:
        """Chunk nhắc tới nhiều thực thể trong `keys` nhất."""
        return self.run("""
            MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk) WHERE e.key IN $keys
            WITH c, count(DISTINCT e) AS hits
            RETURN c.id AS id, c.text AS text, c.source AS source, c.page AS page, hits
            ORDER BY hits DESC, c.id
            LIMIT $limit
        """, keys=keys, limit=limit)
