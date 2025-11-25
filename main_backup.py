import torch
from torch.utils.data import DataLoader
from torch import optim, nn
import time

from src.data_loader import FB15K237Graph, LocalAlignmentDataset
from src.graph_encoder import GraphTransformerEncoder
from src.text_encoder import TextEncoder
from src.hka_trainer import train_hka
from src.knowledge_adapter import KnowledgeAdapter
from src.metrics import evaluate_retrieval

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# --- collate_fn để subgraph variable length ---
def collate_fn(batch):
    out = {}
    for key in batch[0]:
        if key == "subgraph":
            out[key] = [d[key] for d in batch]  # list of tensors
        else:
            out[key] = torch.tensor([d[key] for d in batch])
    return out

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}")

    # --- Load KG ---
    graph_obj = FB15K237Graph(data_dir="data/fb15k237")
    log(f"Loaded KG: {len(graph_obj.train_triples)} train triples, "
        f"{len(graph_obj.test_triples)} test triples, {len(graph_obj.entities)} entities")

    edges = [[graph_obj.entity2id[h], graph_obj.entity2id[t]] for h,r,t in graph_obj.train_triples]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    node_feats = torch.randn(len(graph_obj.entities), 128)

    # --- Graph Encoder + Adapter ---
    graph_encoder = GraphTransformerEncoder(
        in_dim=128, hidden_dim=128, out_dim=128, n_heads=4, n_layers=2,
        dropout=0.1,
        pretrained_path="pretrained_graph_transformer.pth"
    ).to(device)
    graph_encoder.node_feats = node_feats
    graph_encoder.edge_index = edge_index

    adapter = KnowledgeAdapter(in_dim=128, hidden_dim=64, out_dim=128).to(device)

    # --- Text Encoder + Adapter ---
    text_encoder = TextEncoder(entity2text=graph_obj.entity2text,
                               vocab_size=30522, hidden_dim=128, n_layers=2,
                               n_heads=4, dropout=0.1,
                               pretrained_path="pretrained_text_encoder.pth")
    text_adapter = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 128)
    ).to(device)

    # --- Dataset & Dataloader ---
    dataset = LocalAlignmentDataset(graph_obj)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    log(f"Dataloader created: {len(dataloader)} batches (batch_size=2)")

    # ================= Mode 1: Pretrained only =================
    log("--- Mode 1: Pretrained only ---")
    for param in graph_encoder.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False
    for param in adapter.parameters():
        param.requires_grad = False
    for param in text_adapter.parameters():
        param.requires_grad = False

    metrics = evaluate_retrieval(graph_encoder, text_encoder, graph_obj.test_triples,
                                 id2entity=graph_obj.id2entity, entity2id=graph_obj.entity2id,
                                 adapter=adapter, text_adapter=text_adapter,
                                 device=device, batch_size=2)
    log(f"Pretrained metrics: {metrics}")

    # ================= Mode 2: Train HKA only =================
    log("--- Mode 2: Train HKA only ---")
    for param in graph_encoder.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False
    for param in adapter.parameters():
        param.requires_grad = True
    for param in text_adapter.parameters():
        param.requires_grad = True

    optimizer_hka = optim.Adam(list(adapter.parameters()) + list(text_adapter.parameters()), lr=2e-3)

    train_hka(adapter, graph_encoder, text_encoder, dataloader, optimizer_hka,
              device=device, text_adapter=text_adapter, id2entity=graph_obj.id2entity,
              tau=0.07, num_epochs=3, warmup_ratio=0.03, log_interval=10)

    metrics = evaluate_retrieval(graph_encoder, text_encoder, graph_obj.test_triples,
                                 id2entity=graph_obj.id2entity, entity2id=graph_obj.entity2id,
                                 adapter=adapter, text_adapter=text_adapter,
                                 device=device, batch_size=2)
    log(f"HKA only metrics: {metrics}")

    # ================= Mode 3: Train Adapter only =================
    log("--- Mode 3: Train Adapter only ---")
    for param in graph_encoder.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False
    for param in adapter.parameters():
        param.requires_grad = True
    for param in text_adapter.parameters():
        param.requires_grad = True

    optimizer_adapter = optim.Adam(list(adapter.parameters()) + list(text_adapter.parameters()), lr=2e-3)

    train_hka(adapter, graph_encoder, text_encoder, dataloader, optimizer_adapter,
              device=device, text_adapter=text_adapter, id2entity=graph_obj.id2entity,
              tau=0.07, num_epochs=3, warmup_ratio=0.03, log_interval=10)

    metrics = evaluate_retrieval(graph_encoder, text_encoder, graph_obj.test_triples,
                                 id2entity=graph_obj.id2entity, entity2id=graph_obj.entity2id,
                                 adapter=adapter, text_adapter=text_adapter,
                                 device=device, batch_size=2)
    log(f"Adapter only metrics: {metrics}")

    # ================= Mode 4: Train HKA + Adapter =================
    log("--- Mode 4: Train HKA + Adapter ---")
    for param in graph_encoder.parameters():
        param.requires_grad = True
    for param in text_encoder.parameters():
        param.requires_grad = True
    for param in adapter.parameters():
        param.requires_grad = True
    for param in text_adapter.parameters():
        param.requires_grad = True

    optimizer_all = optim.Adam(list(graph_encoder.parameters()) +
                               list(text_encoder.parameters()) +
                               list(adapter.parameters()) +
                               list(text_adapter.parameters()), lr=2e-3)

    train_hka(adapter, graph_encoder, text_encoder, dataloader, optimizer_all,
              device=device, text_adapter=text_adapter, id2entity=graph_obj.id2entity,
              tau=0.07, num_epochs=3, warmup_ratio=0.03, log_interval=10)

    metrics = evaluate_retrieval(graph_encoder, text_encoder, graph_obj.test_triples,
                                 id2entity=graph_obj.id2entity, entity2id=graph_obj.entity2id,
                                 adapter=adapter, text_adapter=text_adapter,
                                 device=device, batch_size=2)
    log(f"HKA + Adapter metrics: {metrics}")


if __name__ == "__main__":
    main()
