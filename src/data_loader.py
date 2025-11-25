# src/data_loader.py
import os
import json
import torch
from torch.utils.data import Dataset
from collections import defaultdict

class FB15K237Graph:
    def __init__(self, data_dir="data/fb15k237"):
        self.data_dir = data_dir
        self.train_triples = self.load_triples(os.path.join(data_dir, "train.txt"))
        self.test_triples = self.load_triples(os.path.join(data_dir, "test.txt"))

        # Entities
        entities = set()
        for h,r,t in self.train_triples + self.test_triples:
            entities.add(h)
            entities.add(t)
        self.entities = sorted(list(entities))
        self.entity2id = {e:i for i,e in enumerate(self.entities)}
        self.id2entity = {i:e for i,e in enumerate(self.entities)}

        # Load entity descriptions
        desc_path = os.path.join(data_dir, "entity2text.json")
        if os.path.exists(desc_path):
            with open(desc_path, "r") as f:
                self.entity2text = json.load(f)
        else:
            self.entity2text = {e:"" for e in self.entities}

        # Build adjacency list for 2-hop subgraph queries
        self.adj = defaultdict(set)
        for h,r,t in self.train_triples:
            hi = self.entity2id[h]
            ti = self.entity2id[t]
            self.adj[hi].add(ti)
            self.adj[ti].add(hi)  # assuming undirected

    def load_triples(self, file_path):
        triples = []
        with open(file_path, 'r') as f:
            for line in f:
                h,r,t = line.strip().split()
                triples.append((h,r,t))
        return triples


class LocalAlignmentDataset(Dataset):
    """
    Dataset cho HKA:
    - Trả về head, tail, relation index.
    - Trả về 2-hop subgraph node indices cho head (dạng list)
    """
    def __init__(self, graph_obj, k_hop=2):
        self.graph = graph_obj
        self.triples = self.graph.train_triples
        self.k_hop = k_hop

    def __len__(self):
        return len(self.triples)

    def get_k_hop_neighbors(self, node_idx):
        visited = {node_idx}
        frontier = {node_idx}
        for _ in range(self.k_hop):
            next_frontier = set()
            for u in frontier:
                next_frontier.update(self.graph.adj[u])
            next_frontier -= visited
            visited.update(next_frontier)
            frontier = next_frontier
        return list(visited)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        h_idx = self.graph.entity2id[h]
        t_idx = self.graph.entity2id[t]
        r_idx = 0  # placeholder if needed
        subgraph_nodes = self.get_k_hop_neighbors(h_idx)
        return {
            "head": h_idx,
            "relation": r_idx,
            "tail": t_idx,
            "subgraph": subgraph_nodes  # keep as list
        }


# collate_fn tùy chỉnh cho DataLoader
def collate_fn(batch):
    batch_out = {}
    for key in batch[0]:
        if key == "subgraph":
            batch_out[key] = [d[key] for d in batch]  # list of lists
        else:
            batch_out[key] = torch.tensor([d[key] for d in batch])
    return batch_out


# --- Dataloader example ---
# dataset = LocalAlignmentDataset(graph_obj)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)