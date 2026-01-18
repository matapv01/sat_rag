# src/hka_trainer.py
import time
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def compute_similarity_matrix(A, B, tau=0.07):
    A = F.normalize(A, p=2, dim=1)
    B = F.normalize(B, p=2, dim=1)
    return A @ B.t() / tau

def contrastive_loss(A, B, tau=0.07):
    sim = compute_similarity_matrix(A, B, tau)
    N = A.size(0)
    labels = torch.arange(N, device=A.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))

def _pool_subgraph(node_emb, subgraphs, device):
    out = []
    for sg in subgraphs:
        sg = torch.tensor(sg, dtype=torch.long, device=device)
        out.append(node_emb[sg].mean(dim=0, keepdim=True))
    return torch.cat(out, dim=0)

def _get_openke_ent_weight(openke_model):
    """
    Trả về entity embedding weight Parameter của OpenKE model.
    Tên field có thể khác nhau giữa version → check dần.
    """
    for name in ["ent_embeddings", "ent_embedding", "ent_emb", "entity_embeddings"]:
        if hasattr(openke_model, name):
            obj = getattr(openke_model, name)
            if hasattr(obj, "weight"):
                return obj.weight
            if isinstance(obj, torch.nn.Parameter):
                return obj
    raise AttributeError("Không tìm thấy entity embedding weight trong OpenKE model. In model.__dict__ để biết tên field.")

def train_hka_joint(
    graph_encoder_openke,   # OpenKE RotatE model
    text_encoder,           # TextEncoderPretrained (HF)
    dataloader,
    optimizer,
    device="cuda",
    tau=0.07,
    epochs_local=2,
    epochs_global=1,
    warmup_ratio=0.03,
    log_interval=10000,
    grad_accum_steps=1,
    doc_max_length=96,
    train_te=True,          # mode: TE train?
    train_ge=True,          # mode: GE train?
):
    """
    Joint HKA (no adapter):
      - Local: subgraph(head) <-> TE(head_text)
      - Global: subgraph(head) <-> TE(subgraph_doc)

    Lưu ý: Nếu train_te=True thì KHÔNG dùng cache embedding (vì sẽ cắt grad).
    """

    # set train/eval theo mode
    graph_encoder_openke.to(device)
    text_encoder.to(device)

    if train_te:
        text_encoder.train()
    else:
        text_encoder.eval()
        for p in text_encoder.parameters():
            p.requires_grad = False

    if train_ge:
        graph_encoder_openke.train()
        for p in graph_encoder_openke.parameters():
            p.requires_grad = True
    else:
        graph_encoder_openke.eval()
        for p in graph_encoder_openke.parameters():
            p.requires_grad = False

    ent_weight = _get_openke_ent_weight(graph_encoder_openke)  # [N, D] Parameter
    D_ge = ent_weight.size(1)
    D_te = text_encoder.hidden_size
    assert D_ge == D_te, f"Dim mismatch: GE={D_ge} vs TE={D_te}. No-adapter yêu cầu bằng nhau."

    total_steps = len(dataloader) * (epochs_local + epochs_global)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = LambdaLR(optimizer, lr_lambda=lambda s: min((s + 1) / warmup_steps, 1.0))

    log(f"HKA start: local={epochs_local} global={epochs_global} train_TE={train_te} train_GE={train_ge}")
    optimizer.zero_grad(set_to_none=True)

    # -------- LOCAL --------
    if epochs_local > 0:
        log("=== STAGE 1: Local (subgraph <-> head_text) ===")
        for ep in range(epochs_local):
            log(f"--- Local Epoch {ep+1}/{epochs_local} ---")
            for it, batch in enumerate(dataloader):
                subgraphs = batch["subgraph"]
                head_texts = batch["head_text"]  # list[str]

                node_emb = ent_weight  # [N, D]
                subgraph_emb = _pool_subgraph(node_emb, subgraphs, device)  # [B, D]

                head_emb = text_encoder.encode(head_texts, device=device, max_length=doc_max_length)  # [B, D]

                loss = contrastive_loss(subgraph_emb, head_emb, tau) / grad_accum_steps
                loss.backward()

                if (it + 1) % grad_accum_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if (it + 1) % log_interval == 0:
                    log(f"[Local] Ep{ep+1} Step{it+1} Loss={(loss.item()*grad_accum_steps):.4f}")

    # -------- GLOBAL --------
    if epochs_global > 0:
        log("=== STAGE 2: Global (subgraph <-> subgraph_doc) ===")
        optimizer.zero_grad(set_to_none=True)

        for ep in range(epochs_global):
            log(f"--- Global Epoch {ep+1}/{epochs_global} ---")
            for it, batch in enumerate(dataloader):
                subgraphs = batch["subgraph"]
                docs = batch["subgraph_doc"]  # list[str]

                node_emb = ent_weight
                subgraph_emb = _pool_subgraph(node_emb, subgraphs, device)

                doc_emb = text_encoder.encode(docs, device=device, max_length=doc_max_length)

                loss = contrastive_loss(subgraph_emb, doc_emb, tau) / grad_accum_steps
                loss.backward()

                if (it + 1) % grad_accum_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if (it + 1) % log_interval == 0:
                    log(f"[Global] Ep{ep+1} Step{it+1} Loss={(loss.item()*grad_accum_steps):.4f}")

    log("HKA completed.")
