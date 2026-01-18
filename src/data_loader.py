# src/data_loader.py
import os
import json
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class FB15K237Graph:
    """
    FB15K237 loader with OpenKE entity-id mapping alignment.

    - If entity2id_path is provided, we use that mapping (OpenKE compatible).
    - Also load valid.txt if exists (recommended).
    """

    def __init__(
        self,
        data_dir="data/fb15k237",
        entity2id_path=None,
        relation2id_path=None
    ):
        self.data_dir = data_dir

        # ---- load splits (include valid if exists) ----
        self.train_triples = self.load_triples(os.path.join(data_dir, "train.txt"))
        self.valid_triples = self.load_triples(os.path.join(data_dir, "valid.txt"), allow_missing=True)
        self.test_triples = self.load_triples(os.path.join(data_dir, "test.txt"))

        # ---- entity mapping ----
        if entity2id_path is not None and os.path.exists(entity2id_path):
            self.entities, self.entity2id, self.id2entity = self.load_entity2id_openke(entity2id_path)
        else:
            # fallback: build from seen entities (NOT compatible with OpenKE ckpt)
            entities = set()
            for h, r, t in (self.train_triples + self.valid_triples + self.test_triples):
                entities.add(h); entities.add(t)
            self.entities = sorted(list(entities))
            self.entity2id = {e: i for i, e in enumerate(self.entities)}
            self.id2entity = {i: e for i, e in enumerate(self.entities)}
            
        # --------- Relations ---------
        if relation2id_path is not None and os.path.exists(relation2id_path):
            self.relations, self.relation2id, self.id2relation = self.load_relation2id_openke(relation2id_path)
        else:
            relations = set(r for _, r, _ in (self.train_triples + self.valid_triples + self.test_triples))
            self.relations = sorted(list(relations))
            self.relation2id = {r:i for i,r in enumerate(self.relations)}
            self.id2relation = {i:r for r,i in self.relation2id.items()}




        # ---- entity descriptions ----
        desc_path = os.path.join(data_dir, "entity2text.json")
        if os.path.exists(desc_path):
            with open(desc_path, "r") as f:
                self.entity2text = json.load(f)
        else:
            self.entity2text = {e: "" for e in self.entities}

        # ---- adjacency (undirected) from TRAIN only ----
        self.adj = defaultdict(list)
        for h, r, t in self.train_triples:
            if h not in self.entity2id or t not in self.entity2id:
                continue
            hi = self.entity2id[h]
            ti = self.entity2id[t]
            self.adj[hi].append((r, ti))
            self.adj[ti].append((f"inv_{r}", hi))  # nếu muốn undirected vẫn giữ “hướng” bằng inv_

    @staticmethod
    def load_entity2id_openke(path):
        """
        OpenKE entity2id.txt format:
          first line: <num_entities>
          next lines: <entity>\t<id>   (sometimes space-separated)
        """
        entity2id = {}
        with open(path, "r") as f:
            first = f.readline().strip()
            try:
                n = int(first)
            except:
                raise ValueError(f"entity2id.txt first line must be integer, got: {first}")

            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                ent = parts[0]
                idx = int(parts[1])
                entity2id[ent] = idx

        # build id2entity list of length n
        id2entity = [None] * n
        for ent, idx in entity2id.items():
            if 0 <= idx < n:
                id2entity[idx] = ent

        # sanity: fill None if any (rare)
        for i in range(n):
            if id2entity[i] is None:
                id2entity[i] = f"__missing_entity_{i}__"

        entities = id2entity[:]  # in id order
        return entities, entity2id, {i: e for i, e in enumerate(entities)}


    def load_relation2id_openke(self, path):
        with open(path, "r") as f:
            first = f.readline().strip()
            try:
                n = int(first)
            except:
                # nếu file không có dòng count
                f.seek(0)
                lines = f.readlines()
                rels = []
                rel2id = {}
                id2rel = {}
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    r, idx = parts[0], int(parts[1])
                    rel2id[r] = idx
                    id2rel[idx] = r
                n = max(id2rel.keys()) + 1 if id2rel else 0
                rels = [id2rel[i] if i in id2rel else f"__missing_rel_{i}__" for i in range(n)]
                return rels, rel2id, id2rel

            id2rel = [None] * n
            rel2id = {}
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                r, idx = parts[0], int(parts[1])
                rel2id[r] = idx
                if 0 <= idx < n:
                    id2rel[idx] = r
            for i in range(n):
                if id2rel[i] is None:
                    id2rel[i] = f"__missing_rel_{i}__"
            return id2rel, rel2id, {i:r for i,r in enumerate(id2rel)}


    @staticmethod
    def load_triples(file_path, allow_missing=False):
        triples = []
        if allow_missing and (not os.path.exists(file_path)):
            return triples
        with open(file_path, "r") as f:
            for line in f:
                h, r, t = line.strip().split()
                triples.append((h, r, t))
        return triples

    def build_pseudo_doc_for_subgraph(
        self,
        head_idx: int,
        subgraph_nodes: list[int],
        max_edges: int = 40,
        max_chars: int = 800,
        use_entity_desc: bool = True,
    ) -> str:
        head_ent = self.id2entity[head_idx]
        parts = []

        if use_entity_desc:
            head_desc = self.entity2text.get(head_ent, "")
            head_desc = head_desc.strip() if isinstance(head_desc, str) else ""
            if head_desc:
                parts.append(f"{head_ent}: {head_desc}")

        edge_count = 0
        nodes_set = set(subgraph_nodes)

        for u in sorted(nodes_set):
            u_name = self.id2entity[u]
            for r, v in self.adj[u]:
                if v in nodes_set:
                    v_name = self.id2entity[v]
                    parts.append(f"{u_name} {r} {v_name}.")
                    edge_count += 1
                    if edge_count >= max_edges:
                        break
            if edge_count >= max_edges:
                break


        if not parts:
            return "This entity has limited connections in the knowledge graph."

        doc = " ".join(parts)
        if len(doc) > max_chars:
            doc = doc[:max_chars]
        return doc


class LocalAlignmentDataset(Dataset):
    def __init__(
        self,
        graph_obj: FB15K237Graph,
        k_hop: int = 2,
        max_edges: int = 40,
        max_chars: int = 800,
        use_entity_desc: bool = True,
        cache_by_head: bool = True,
    ):
        self.graph = graph_obj
        self.triples = self.graph.train_triples
        self.k_hop = k_hop

        self.max_edges = max_edges
        self.max_chars = max_chars
        self.use_entity_desc = use_entity_desc

        self.cache_by_head = cache_by_head
        self._subgraph_cache = {}
        self._doc_cache = {}

    def __len__(self):
        return len(self.triples)

    def get_k_hop_neighbors(self, node_idx: int) -> list[int]:
        if self.cache_by_head and node_idx in self._subgraph_cache:
            return self._subgraph_cache[node_idx]

        visited = {node_idx}
        frontier = {node_idx}

        for _ in range(self.k_hop):
            nxt = set()
            for u in frontier:
                nxt.update(v for _, v in self.graph.adj[u])
            nxt -= visited
            visited |= nxt
            frontier = nxt
            if not frontier:
                break

        out = list(visited)
        if self.cache_by_head:
            self._subgraph_cache[node_idx] = out
        return out

    def get_pseudo_doc(self, head_idx: int, subgraph_nodes: list[int]) -> str:
        if self.cache_by_head and head_idx in self._doc_cache:
            return self._doc_cache[head_idx]

        doc = self.graph.build_pseudo_doc_for_subgraph(
            head_idx=head_idx,
            subgraph_nodes=subgraph_nodes,
            max_edges=self.max_edges,
            max_chars=self.max_chars,
            use_entity_desc=self.use_entity_desc,
        )

        if self.cache_by_head:
            self._doc_cache[head_idx] = doc
        return doc

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        h_idx = self.graph.entity2id[h]
        t_idx = self.graph.entity2id[t]
        r_idx = self.graph.relation2id.get(r, 0)


        subgraph_nodes = self.get_k_hop_neighbors(h_idx)
        subgraph_doc = self.get_pseudo_doc(h_idx, subgraph_nodes)

        head_ent = self.graph.id2entity[h_idx]
        head_text = self.graph.entity2text.get(head_ent, head_ent)

        return {
            "head": h_idx,
            "relation": r_idx,
            "tail": t_idx,
            "subgraph": subgraph_nodes,
            "subgraph_doc": subgraph_doc,
            "head_text": head_text,   # 🔥 ADD
        }



def collate_fn(batch):
    out = {}
    for k in batch[0].keys():
        if k in ("subgraph", "subgraph_doc", "head_text"):
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
    return out

