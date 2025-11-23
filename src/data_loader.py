import os
import json
import torch
from torch.utils.data import Dataset

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

        # Load entity descriptions if exist
        desc_path = os.path.join(data_dir, "entity2text.json")
        if os.path.exists(desc_path):
            with open(desc_path, "r") as f:
                self.entity2text = json.load(f)
        else:
            self.entity2text = {e:"" for e in self.entities}

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
    - Trả về head, relation, tail dưới dạng **index**.
    """
    def __init__(self, graph_obj):
        self.graph = graph_obj
        self.triples = self.graph.train_triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        h_idx = self.graph.entity2id[h]
        t_idx = self.graph.entity2id[t]
        r_idx = 0  # Nếu cần relation index, hiện placeholder
        return {
            "head": h_idx,
            "relation": r_idx,
            "tail": t_idx
        }
