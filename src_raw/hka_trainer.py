# train_hka.py
import torch
import torch.nn.functional as F
import time
from torch.optim.lr_scheduler import LambdaLR

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def compute_similarity_matrix(H, D, tau=0.07):
    """
    H: [batch_size, dim] node embeddings
    D: [batch_size, dim] text embeddings
    """
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


def train_hka(adapter, graph_encoder, text_encoder, dataloader, optimizer,
              device="cuda", text_adapter=None, id2entity=None,
              tau=0.07, num_epochs=3, warmup_ratio=0.03, log_interval=1):
    """
    Adapter-only HKA training (Local -> Global sequential).
    - Handles variable-length subgraphs
    - Graph/Text encoders frozen
    """
    adapter.train()
    graph_encoder.eval()
    text_encoder.eval()
    if text_adapter:
        text_adapter.train()

    total_steps = len(dataloader) * num_epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min((step+1)/warmup_steps, 1.0))

    log(f"Starting HKA training for {num_epochs} epochs on device {device}")

    for epoch in range(num_epochs):
        log(f"=== Epoch {epoch+1}/{num_epochs} ===")
        epoch_loss = 0.0

        # --- Local alignment ---
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # Graph embeddings
            node_feats = graph_encoder.node_feats.to(device)
            edge_index = graph_encoder.edge_index.to(device)
            node_emb = graph_encoder(node_feats, edge_index)
            node_emb = adapter(node_emb)

            # Text embeddings
            entity_names = [text_encoder.entity2text[id2entity[batch["head"][j].item()]]
                            for j in range(len(batch["head"]))]
            text_emb = text_encoder.encode(entity_names)
            if text_adapter:
                text_emb = text_adapter(text_emb)

            # Local alignment: mean pooling per subgraph
            subgraph_embs = []
            for s in batch["subgraph"]:
                # s = s.to(device)
                s = torch.tensor(s, dtype=torch.long, device=device)  # <- chuyển list -> tensor
                subgraph_embs.append(node_emb[s].mean(dim=0, keepdim=True))
            subgraph_emb = torch.cat(subgraph_embs, dim=0)
            L_local = local_alignment_loss(subgraph_emb, text_emb, tau)

            L_local.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += L_local.item()
            if (i+1) % log_interval == 0:
                log(f"[Epoch {epoch+1} Batch {i+1}/{len(dataloader)}] L_local={L_local.item():.4f}")

        avg_epoch_loss = epoch_loss / len(dataloader)
        log(f"Local alignment finished. Avg loss: {avg_epoch_loss:.4f}")

        # --- Global alignment ---
        epoch_loss = 0.0
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            node_feats = graph_encoder.node_feats.to(device)
            edge_index = graph_encoder.edge_index.to(device)
            node_emb = graph_encoder(node_feats, edge_index)
            node_emb = adapter(node_emb)

            entity_names = [text_encoder.entity2text[id2entity[batch["head"][j].item()]]
                            for j in range(len(batch["head"]))]
            text_emb = text_encoder.encode(entity_names)
            if text_adapter:
                text_emb = text_adapter(text_emb)

            # Global alignment: mean over each subgraph, then mean of text embeddings
            subgraph_embs = []
            for s in batch["subgraph"]:
                s = s.to(device)
                subgraph_embs.append(node_emb[s].mean(dim=0, keepdim=True))
            subgraph_emb = torch.cat(subgraph_embs, dim=0)
            doc_emb = text_emb.mean(dim=0, keepdim=True)
            L_global = global_alignment_loss(subgraph_emb, doc_emb, tau)

            L_global.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += L_global.item()
            if (i+1) % log_interval == 0:
                log(f"[Epoch {epoch+1} Batch {i+1}/{len(dataloader)}] L_global={L_global.item():.4f}")

        avg_epoch_loss = epoch_loss / len(dataloader)
        log(f"Global alignment finished. Avg loss: {avg_epoch_loss:.4f}")

    log("HKA training completed.")