import torch
from torch.utils.data import DataLoader
from torch import optim, nn
import time

from src.data_loader import FB15K237Graph, LocalAlignmentDataset
from src.graph_encoder import GraphormerEncoder
from src.text_encoder import TextEncoderPretrained
from src.hka_trainer import train_hka
from src.knowledge_adapter import KnowledgeAdapter
from src.metrics import evaluate_retrieval

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# --- collate_fn ---
def collate_fn(batch):
    out = {}
    for key in batch[0]:
        if key == "subgraph":
            out[key] = [d[key] for d in batch]
        else:
            out[key] = torch.tensor([d[key] for d in batch])
    return out

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}")

    # ---------------- Load KG ----------------
    graph_obj = FB15K237Graph(data_dir="data/fb15k237")
    log(f"Loaded KG: {len(graph_obj.train_triples)} train triples, "
        f"{len(graph_obj.test_triples)} test triples, {len(graph_obj.entities)} entities")

    num_entities = len(graph_obj.entities)

    # ---------------- Graph Encoder (Pretrained embeddings only) ----------------
    log("Loading Graphormer pretrained model...")
    graph_encoder = GraphormerEncoder(
        model_name="clefourrier/graphormer-base-pcqm4mv2",
        use_pretrained=True,
        device=device
    ).to(device)

    # Load pretrained node embeddings (no forward)
    pretrained_dim = graph_encoder.model.config.hidden_size
    graph_encoder.load_lambdaKG_embeddings(torch.randn(num_entities, pretrained_dim).to(device))

    # Graph Encoder always frozen
    for p in graph_encoder.parameters():
        p.requires_grad = False

    # ---------------- Adapter ----------------
    adapter = KnowledgeAdapter(
        in_dim=pretrained_dim,
        hidden_dim=64,
        out_dim=128
    ).to(device)

    # ---------------- Text Encoder ----------------
    text_encoder = TextEncoderPretrained(entity2text=graph_obj.entity2text).to(device)
    text_dim = 128  # output dimension after TextEncoder
    text_adapter = nn.Sequential(
        nn.Linear(text_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 128)
    ).to(device)

    # ---------------- Dataset ----------------
    dataset = LocalAlignmentDataset(graph_obj)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    log(f"Dataloader created: {len(dataloader)} batches")

    # =====================================================
    # Mode 1: Pretrained only
    # =====================================================
    log("--- Mode 1: Pretrained only ---")
    for m in [adapter, text_adapter, text_encoder]:
        for p in m.parameters():
            p.requires_grad = False

    metrics = evaluate_retrieval(
        graph_encoder, text_encoder, graph_obj.test_triples,
        id2entity=graph_obj.id2entity,
        entity2id=graph_obj.entity2id,
        adapter=adapter,        # adapter applied to node embeddings
        text_adapter=text_adapter,
        device=device, batch_size=2
    )
    log(f"Pretrained metrics: {metrics}")

    # =====================================================
    # Mode 2: Train HKA only
    # =====================================================
    log("--- Mode 2: Train HKA only ---")
    for p in adapter.parameters():
        p.requires_grad = True
    for p in text_adapter.parameters():
        p.requires_grad = True

    optimizer_hka = optim.Adam(
        list(adapter.parameters()) + list(text_adapter.parameters()),
        lr=2e-3
    )

    train_hka(
        adapter, graph_encoder, text_encoder, dataloader,
        optimizer_hka, device=device,
        text_adapter=text_adapter, id2entity=graph_obj.id2entity,
        tau=0.07, num_epochs=3, warmup_ratio=0.03,
        log_interval=10
    )

    metrics = evaluate_retrieval(
        graph_encoder, text_encoder, graph_obj.test_triples,
        id2entity=graph_obj.id2entity,
        entity2id=graph_obj.entity2id,
        adapter=adapter,
        text_adapter=text_adapter,
        device=device, batch_size=2
    )
    log(f"HKA only metrics: {metrics}")

    # =====================================================
    # Mode 3: Train Adapter only
    # =====================================================
    log("--- Mode 3: Train Adapter only ---")
    optimizer_adapter = optim.Adam(
        list(adapter.parameters()) + list(text_adapter.parameters()),
        lr=2e-3
    )

    train_hka(
        adapter, graph_encoder, text_encoder, dataloader,
        optimizer_adapter, device=device,
        text_adapter=text_adapter, id2entity=graph_obj.id2entity,
        tau=0.07, num_epochs=3, warmup_ratio=0.03,
        log_interval=10
    )

    metrics = evaluate_retrieval(
        graph_encoder, text_encoder, graph_obj.test_triples,
        id2entity=graph_obj.id2entity,
        entity2id=graph_obj.entity2id,
        adapter=adapter,
        text_adapter=text_adapter,
        device=device, batch_size=2
    )
    log(f"Adapter only metrics: {metrics}")

    # =====================================================
    # Mode 4: Train HKA + Adapter + Text Encoder
    # =====================================================
    log("--- Mode 4: Train HKA + Adapter + Text Encoder ---")
    for p in text_encoder.parameters():
        p.requires_grad = True
    for p in adapter.parameters():
        p.requires_grad = True
    for p in text_adapter.parameters():
        p.requires_grad = True

    optimizer_all = optim.Adam(
        list(text_encoder.parameters()) +
        list(adapter.parameters()) +
        list(text_adapter.parameters()), lr=2e-3
    )

    train_hka(
        adapter, graph_encoder, text_encoder, dataloader,
        optimizer_all, device=device,
        text_adapter=text_adapter, id2entity=graph_obj.id2entity,
        tau=0.07, num_epochs=3, warmup_ratio=0.03,
        log_interval=10
    )

    metrics = evaluate_retrieval(
        graph_encoder, text_encoder, graph_obj.test_triples,
        id2entity=graph_obj.id2entity,
        entity2id=graph_obj.entity2id,
        adapter=adapter,
        text_adapter=text_adapter,
        device=device, batch_size=2
    )
    log(f"HKA + Adapter + Text metrics: {metrics}")


if __name__ == "__main__":
    main()
