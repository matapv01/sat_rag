import torch
import torch.nn.functional as F
import time

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def compute_similarity_matrix(H, D, tau=0.07):
    H = F.normalize(H, p=2, dim=1)
    D = F.normalize(D, p=2, dim=1)
    return torch.matmul(H, D.t()) / tau

def local_alignment_loss(node_emb, text_emb, tau=0.07):
    Lambda = compute_similarity_matrix(node_emb, text_emb, tau)
    N = node_emb.size(0)
    labels = torch.arange(N, device=node_emb.device)
    loss_i = F.cross_entropy(Lambda, labels)
    loss_j = F.cross_entropy(Lambda.t(), labels)
    return 0.5 * (loss_i + loss_j)

def global_alignment_loss(subgraph_emb, doc_emb, tau=0.07):
    Lambda = compute_similarity_matrix(subgraph_emb, doc_emb, tau)
    N = subgraph_emb.size(0)
    labels = torch.arange(N, device=subgraph_emb.device)
    loss_i = F.cross_entropy(Lambda, labels)
    loss_j = F.cross_entropy(Lambda.t(), labels)
    return 0.5 * (loss_i + loss_j)

def train_hka(graph_encoder, text_encoder, dataloader, optimizer, device, adapter,
              text_adapter=None, id2entity=None, tau=0.07, log_interval=50):
    """Train HKA and return avg loss (optionally with optimizer)."""
    graph_encoder.train()
    adapter.train()
    text_encoder.train()
    if text_adapter:
        text_adapter.train()

    total_loss = 0.0

    for i, batch in enumerate(dataloader):
        if optimizer:
            optimizer.zero_grad()

        # Graph embeddings
        node_feats = graph_encoder.node_feats.to(device)
        edge_index = graph_encoder.edge_index.to(device)
        node_emb = graph_encoder(node_feats, edge_index)
        node_emb = adapter(node_emb)

        # Text embeddings
        entity_names = [text_encoder.entity2text[id2entity[batch["head"][j].item()]]
                        for j in range(len(batch["head"]))]
        text_emb = text_encoder.encode(entity_names).to(device)
        if text_adapter and optimizer:
            text_emb = text_adapter(text_emb)

        # Loss
        L_local = local_alignment_loss(node_emb[batch["head"]], text_emb, tau)
        subgraph_emb = node_emb[batch["head"]].mean(dim=0, keepdim=True)
        doc_emb = text_emb.mean(dim=0, keepdim=True)
        L_global = global_alignment_loss(subgraph_emb, doc_emb, tau)
        L_HKA = L_local + L_global

        if optimizer:
            L_HKA.backward()
            optimizer.step()

        total_loss += L_HKA.item()

        if (i+1) % log_interval == 0:
            log(f"[Batch {i+1}/{len(dataloader)}] "
                f"Loss={L_HKA.item():.4f} "
                f"Device={device}")

    avg_loss = total_loss / len(dataloader)
    log(f"Avg loss: {avg_loss:.4f} on device {device}")
    return avg_loss
