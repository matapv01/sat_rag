import torch
from torch.utils.data import DataLoader
from torch import optim, nn
import time

from src.data_loader import FB15K237Graph, LocalAlignmentDataset
from src.graph_encoder import GraphEncoder
from src.text_encoder import TextEncoder
from src.hka_trainer import train_hka
from src.knowledge_adapter import KnowledgeAdapter
from src.metrics import evaluate_retrieval

def log(msg):
    # log với timestamp
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

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
    graph_encoder = GraphEncoder(in_dim=128, hidden_dim=256, out_dim=256).to(device)
    graph_encoder.node_feats = node_feats
    graph_encoder.edge_index = edge_index
    adapter = KnowledgeAdapter(in_dim=256, hidden_dim=128, out_dim=256).to(device)

    # --- Text Encoder + projection ---
    text_encoder = TextEncoder(entity2text=graph_obj.entity2text)
    text_adapter = nn.Linear(384, 256).to(device)

    # --- Dataset & Dataloader ---
    dataset = LocalAlignmentDataset(graph_obj)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    log(f"Dataloader created: {len(dataloader)} batches")

    # --- Mode 1: Pretrained only ---
    log("--- Mode 1: Pretrained only ---")
    for param in graph_encoder.parameters():
        param.requires_grad = False
    for param in text_encoder.encoder.parameters():
        param.requires_grad = False
    for param in adapter.parameters():
        param.requires_grad = False
    for param in text_adapter.parameters():
        param.requires_grad = False

    metrics = evaluate_retrieval(graph_encoder, text_encoder, graph_obj.test_triples,
                                 id2entity=graph_obj.id2entity, entity2id=graph_obj.entity2id,
                                 adapter=adapter, text_adapter=text_adapter,
                                 device=device, batch_size=32)
    log(f"Pretrained metrics: {metrics}")

    # --- Mode 2: Train HKA only ---
    log("--- Mode 2: Train HKA only ---")
    for param in adapter.parameters():
        param.requires_grad = False
    for param in text_adapter.parameters():
        param.requires_grad = False
    for param in graph_encoder.parameters():
        param.requires_grad = True
    for param in text_encoder.encoder.parameters():
        param.requires_grad = True

    optimizer_hka = optim.Adam(list(graph_encoder.parameters()) + list(text_encoder.encoder.parameters()), lr=1e-4)

    for epoch in range(5):
        log(f"=== Epoch {epoch+1} ===")
        loss = train_hka(graph_encoder, text_encoder, dataloader, optimizer_hka,
                         device, adapter, text_adapter=text_adapter, id2entity=graph_obj.id2entity,
                         log_interval=100)
        metrics = evaluate_retrieval(graph_encoder, text_encoder, graph_obj.test_triples,
                                     id2entity=graph_obj.id2entity, entity2id=graph_obj.entity2id,
                                     adapter=adapter, text_adapter=text_adapter,
                                     device=device, batch_size=32)
        log(f"HKA only metrics: {metrics}, loss={loss:.4f}")

    # --- Mode 3: Train Adapter only ---
    log("--- Mode 3: Train Adapter only ---")
    for param in graph_encoder.parameters():
        param.requires_grad = False
    for param in text_encoder.encoder.parameters():
        param.requires_grad = False
    for param in adapter.parameters():
        param.requires_grad = True
    for param in text_adapter.parameters():
        param.requires_grad = True

    optimizer_adapter = optim.Adam(list(adapter.parameters()) + list(text_adapter.parameters()), lr=1e-4)

    for epoch in range(5):
        log(f"=== Epoch {epoch+1} ===")
        loss = train_hka(graph_encoder, text_encoder, dataloader, optimizer_adapter,
                         device, adapter, text_adapter=text_adapter, id2entity=graph_obj.id2entity,
                         log_interval=100)
        metrics = evaluate_retrieval(graph_encoder, text_encoder, graph_obj.test_triples,
                                     id2entity=graph_obj.id2entity, entity2id=graph_obj.entity2id,
                                     adapter=adapter, text_adapter=text_adapter,
                                     device=device, batch_size=32)
        log(f"Adapter only metrics: {metrics}, loss={loss:.4f}")

    # --- Mode 4: Train HKA + Adapter ---
    log("--- Mode 4: Train HKA + Adapter ---")
    for param in graph_encoder.parameters():
        param.requires_grad = True
    for param in text_encoder.encoder.parameters():
        param.requires_grad = True
    for param in adapter.parameters():
        param.requires_grad = True
    for param in text_adapter.parameters():
        param.requires_grad = True

    optimizer_all = optim.Adam(list(graph_encoder.parameters()) +
                               list(text_encoder.encoder.parameters()) +
                               list(adapter.parameters()) +
                               list(text_adapter.parameters()), lr=1e-4)

    for epoch in range(5):
        log(f"=== Epoch {epoch+1} ===")
        loss = train_hka(graph_encoder, text_encoder, dataloader, optimizer_all,
                         device, adapter, text_adapter=text_adapter, id2entity=graph_obj.id2entity,
                         log_interval=100)
        metrics = evaluate_retrieval(graph_encoder, text_encoder, graph_obj.test_triples,
                                     id2entity=graph_obj.id2entity, entity2id=graph_obj.entity2id,
                                     adapter=adapter, text_adapter=text_adapter,
                                     device=device, batch_size=32)
        log(f"HKA + Adapter metrics: {metrics}, loss={loss:.4f}")

if __name__ == "__main__":
    main()
