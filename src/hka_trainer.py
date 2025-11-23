import torch
import torch.nn.functional as F
from tqdm import tqdm   # ⭐ progress bar

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

def train_hka(graph_encoder, text_encoder, dataloader, optimizer, device,
              adapter, text_adapter=None, id2entity=None, tau=0.07):

    graph_encoder.train()
    adapter.train()
    if text_adapter:
        text_adapter.train()
    text_encoder.train()

    total_loss = 0.0

    # ⭐ thêm progress bar
    loop = tqdm(dataloader, desc="Training", ncols=100)

    for batch in loop:
        optimizer.zero_grad()

        # --- Graph embeddings ---
        node_feats = graph_encoder.node_feats.to(device)
        edge_index = graph_encoder.edge_index.to(device)
        node_emb = graph_encoder(node_feats, edge_index)
        node_emb = adapter(node_emb)

        # --- Text embeddings ---
        entity_names = [
            text_encoder.entity2text[id2entity[batch["head"][j].item()]]
            for j in range(len(batch["head"]))
        ]
        text_emb = text_encoder.encode(entity_names).to(device)
        if text_adapter:
            text_emb = text_adapter(text_emb)

        # --- Local loss ---
        L_local = local_alignment_loss(node_emb[batch["head"]], text_emb, tau)

        # --- Global loss ---
        subgraph_emb = node_emb[batch["head"]].mean(dim=0, keepdim=True)
        doc_emb = text_emb.mean(dim=0, keepdim=True)
        L_global = global_alignment_loss(subgraph_emb, doc_emb, tau)

        # --- Total loss ---
        L_HKA = L_local + L_global
        L_HKA.backward()
        optimizer.step()

        total_loss += L_HKA.item()

        # ⭐ cập nhật progress bar
        loop.set_postfix(loss=L_HKA.item())

    return total_loss / len(dataloader)
