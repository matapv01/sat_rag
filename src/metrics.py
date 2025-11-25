# src/metrics.py
import torch
import torch.nn.functional as F

def compute_similarity_matrix(H, D, tau=0.07):
    """
    H: [batch_size, dim] node embeddings
    D: [batch_size, dim] text embeddings
    """
    H = F.normalize(H, p=2, dim=1)
    D = F.normalize(D, p=2, dim=1)
    return torch.matmul(H, D.t()) / tau

def compute_metrics(Lambda):
    """
    Lambda: similarity matrix [batch_size, batch_size]
    Returns MRR, Hits@1, Hits@3, Hits@10
    """
    with torch.no_grad():
        ranks = torch.argsort(-Lambda, dim=1)  # descending
        N = Lambda.size(0)
        rr_sum = 0.0
        hits1 = hits3 = hits10 = 0
        for i in range(N):
            rank_i = (ranks[i] == i).nonzero(as_tuple=True)[0].item() + 1
            rr_sum += 1.0 / rank_i
            if rank_i <= 1: hits1 += 1
            if rank_i <= 3: hits3 += 1
            if rank_i <= 10: hits10 += 1
        return {
            "MRR": rr_sum / N,
            "Hits@1": hits1 / N,
            "Hits@3": hits3 / N,
            "Hits@10": hits10 / N
        }

def evaluate_retrieval(graph_encoder, text_encoder, triples, id2entity, entity2id,
                       adapter=None, text_adapter=None, device="cpu", batch_size=32, tau=0.07):
    """
    Evaluate retrieval metrics on given triples.
    - GCN receives full node_feats + edge_index
    - text embeddings are batched
    """
    graph_encoder.eval()
    text_encoder.eval()
    if adapter: adapter.eval()
    if text_adapter: text_adapter.eval()

    # --- compute full node embeddings once ---
    with torch.no_grad():
        node_feats = graph_encoder.node_feats.to(device)
        edge_index = graph_encoder.edge_index.to(device)
        full_node_emb = graph_encoder(node_feats, edge_index)
        if adapter:
            full_node_emb = adapter(full_node_emb)

    all_metrics = {"MRR": 0.0, "Hits@1":0.0, "Hits@3":0.0, "Hits@10":0.0}
    n_batches = (len(triples) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(n_batches):
            batch = triples[i*batch_size:(i+1)*batch_size]
            # --- convert string triples -> IDs ---
            head_ids = torch.tensor([entity2id[h] for h,_,_ in batch], dtype=torch.long, device=device)
            tail_ids = torch.tensor([entity2id[t] for _,_,t in batch], dtype=torch.long, device=device)

            # --- select node embeddings for head nodes ---
            node_emb = full_node_emb[head_ids]

            # --- text embeddings ---
            entity_names = [text_encoder.entity2text[id2entity[h.item()]] for h in head_ids]
            text_emb = text_encoder.encode(entity_names).to(device)
            if text_adapter:
                text_emb = text_adapter(text_emb)

            # --- similarity & metrics ---
            Lambda = compute_similarity_matrix(node_emb, text_emb, tau)
            metrics = compute_metrics(Lambda)
            for k in all_metrics:
                all_metrics[k] += metrics[k]

    # average over batches
    for k in all_metrics:
        all_metrics[k] /= n_batches
    return all_metrics