# src/metrics.py
import torch
import torch.nn.functional as F

def compute_metrics_for_ground_truth(Lambda, true_ids):
    """
    Lambda: [B, N] similarity matrix
    true_ids: [B] indices of true tails
    """
    ranks = torch.argsort(-Lambda, dim=1)
    
    B = Lambda.size(0)
    rr = 0
    h1 = h3 = h10 = 0
    for i in range(B):
        gt = true_ids[i].item()
        rank = (ranks[i] == gt).nonzero(as_tuple=True)[0].item() + 1
        rr += 1.0 / rank
        if rank <= 1: h1 += 1
        if rank <= 3: h3 += 1
        if rank <= 10: h10 += 1

    return {
        "MRR": rr / B,
        "Hits@1": h1 / B,
        "Hits@3": h3 / B,
        "Hits@10": h10 / B,
    }


def evaluate_retrieval(
    graph_encoder,
    text_encoder,
    triples,
    id2entity,
    entity2id,
    adapter=None,
    text_adapter=None,
    device="cpu",
    batch_size=32,
    tau=0.07,
):
    """
    Evaluate retrieval:
    - head node embedding vs ALL tail text embeddings
    - Handles pretrained embeddings + adapter
    """
    graph_encoder.eval()
    text_encoder.eval()
    if adapter: adapter.eval()
    if text_adapter: text_adapter.eval()

    # ------------------- Full Node Embeddings -------------------
    with torch.no_grad():
        if hasattr(graph_encoder, "embeddings"):
            full_node_emb = graph_encoder.embeddings.to(device)  # [num_entities, hidden_dim]
        else:
            node_feats = graph_encoder.node_feats.to(device)
            edge_index = graph_encoder.edge_index.to(device)
            full_node_emb = graph_encoder(node_feats, edge_index)

        if adapter:
            full_node_emb = adapter(full_node_emb)  # -> [num_entities, adapter_dim]

        full_node_emb = F.normalize(full_node_emb, dim=1)

    # ------------------- Full Text Embeddings -------------------
    all_entities = [id2entity[i] for i in range(len(id2entity))]
    all_texts = [text_encoder.entity2text[e] for e in all_entities]

    with torch.no_grad():
        text_emb_all = text_encoder.encode(all_texts)  # [num_entities, text_dim]
        if text_adapter:
            text_emb_all = text_adapter(text_emb_all)
        text_emb_all = F.normalize(text_emb_all, dim=1)

    # ------------------- Batched retrieval -------------------
    all_metrics = {"MRR": 0.0, "Hits@1":0.0, "Hits@3":0.0, "Hits@10":0.0}
    n_batches = (len(triples) + batch_size - 1) // batch_size

    for bi in range(n_batches):
        batch = triples[bi*batch_size:(bi+1)*batch_size]

        head_ids = torch.tensor([entity2id[h] for h,_,_ in batch], device=device)
        tail_ids = torch.tensor([entity2id[t] for _,_,t in batch], device=device)

        # ------------------- Select head embeddings -------------------
        H = full_node_emb[head_ids]  # [B, dim]

        # ------------------- Similarity with all tail embeddings -------------------
        Lambda = H @ text_emb_all.t() / tau  # [B, num_entities]

        metrics = compute_metrics_for_ground_truth(Lambda, tail_ids)
        for k in all_metrics:
            all_metrics[k] += metrics[k]

    # Average over batches
    for k in all_metrics:
        all_metrics[k] /= n_batches

    return all_metrics
