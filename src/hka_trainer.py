# train_hka.py
import torch
import torch.nn.functional as F
import time
from torch.optim.lr_scheduler import LambdaLR


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def compute_similarity_matrix(A, B, tau=0.07):
    """
    Compute cosine similarity matrix and scale by tau
    A: [B, dim]
    B: [B, dim]
    """
    A = F.normalize(A, p=2, dim=1)
    B = F.normalize(B, p=2, dim=1)
    return A @ B.t() / tau


def contrastive_loss(A, B, tau=0.07):
    """
    Symmetric contrastive loss between A and B
    """
    sim = compute_similarity_matrix(A, B, tau)
    N = A.size(0)
    labels = torch.arange(N, device=A.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))


def train_hka(adapter, graph_encoder, text_encoder, dataloader, optimizer,
              device="cuda", text_adapter=None, id2entity=None,
              tau=0.07, num_epochs=3, warmup_ratio=0.03, log_interval=1):
    """
    HKA Training: Local -> Global alignment
    - graph_encoder.embeddings: pretrained node embeddings
    - Adapter maps Graphormer output -> 128d
    - TextEncoder + optional text_adapter -> 128d
    """

    graph_encoder.eval()     # Graphormer frozen
    text_encoder.eval()      # LM frozen
    adapter.train()
    if text_adapter:
        text_adapter.train()

    # Warmup scheduler
    total_steps = len(dataloader) * num_epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda s: min((s + 1) / warmup_steps, 1.0)
    )

    log(f"HKA Training start: {num_epochs} epochs on {device}")

    step = 0
    for epoch in range(num_epochs):
        log(f"=== Epoch {epoch+1}/{num_epochs} ===")

        # ----------------- LOCAL ALIGNMENT -----------------
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            heads = batch["head"].to(device)
            subgraphs = batch["subgraph"]

            # --- Graph embeddings ---
            node_emb = graph_encoder.embeddings.to(device)  # [num_entities, dim]
            node_emb = adapter(node_emb)                            # [num_entities, 128]

            # --- Text embeddings ---
            names = [text_encoder.entity2text[id2entity[h.item()]] for h in heads]
            text_emb = text_encoder.encode(names, device=device)
            if text_adapter:
                text_emb = text_adapter(text_emb)                  # [B, 128]

            # --- Subgraph pooling ---
            subgraph_emb_list = []
            for sg in subgraphs:
                sg = torch.tensor(sg, dtype=torch.long, device=device)  # <- chuyển list -> tensor
                subgraph_emb_list.append(node_emb[sg].mean(dim=0, keepdim=True))
            subgraph_emb = torch.cat(subgraph_emb_list, dim=0)     # [B, 128]

            # --- Local Loss ---
            L_local = contrastive_loss(subgraph_emb, text_emb, tau)
            L_local.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            if (batch_idx + 1) % log_interval == 0:
                log(f"[Local] Epoch {epoch+1} Step {batch_idx+1} Loss={L_local.item():.4f}")

        # ----------------- GLOBAL ALIGNMENT -----------------
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            heads = batch["head"].to(device)
            subgraphs = batch["subgraph"]

            # --- Graph embeddings ---
            node_emb = graph_encoder.embeddings.to(device)
            node_emb = adapter(node_emb)

            # --- Text embeddings ---
            names = [text_encoder.entity2text[id2entity[h.item()]] for h in heads]
            text_emb = text_encoder.encode(names, device=device)
            if text_adapter:
                text_emb = text_adapter(text_emb)

            # Document-level embedding (mean over batch)
            doc_emb = text_emb.mean(dim=0, keepdim=True).repeat(text_emb.size(0), 1)  # [B, 128]

            # --- Subgraph pooling ---
            subgraph_emb_list = []
            for sg in subgraphs:
                sg = torch.tensor(sg, dtype=torch.long, device=device)  # <- chuyển list -> tensor
                subgraph_emb_list.append(node_emb[sg].mean(dim=0, keepdim=True))
            subgraph_emb = torch.cat(subgraph_emb_list, dim=0)  # [B, 128]

            # --- Global Loss ---
            L_global = contrastive_loss(subgraph_emb, doc_emb, tau)
            L_global.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            if (batch_idx + 1) % log_interval == 0:
                log(f"[Global] Epoch {epoch+1} Step {batch_idx+1} Loss={L_global.item():.4f}")

    log("HKA training completed.")
