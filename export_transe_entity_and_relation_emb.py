import os
import torch


def find_embedding_tensor(state, num_rows, prefer_keywords):
    """
    Robustly find an embedding tensor in OpenKE ckpt.
    - prefer_keywords: list[str] used for scoring keys
    """
    candidates = []
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.dim() == 2 and v.size(0) == num_rows:
            score = 0
            lk = k.lower()
            for kw, w in prefer_keywords:
                if kw in lk:
                    score += w
            candidates.append((score, k, v))

    if not candidates:
        raise ValueError(f"Cannot find any 2D tensor with shape [num_rows={num_rows}, dim] in ckpt.")

    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1], candidates[0][2]


def load_openke_id_file(path):
    """
    OpenKE entity2id.txt / relation2id.txt format:
      first line: N
      then: token id
    """
    with open(path, "r") as f:
        n = int(f.readline().strip())
        id2token = [None] * n
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            tok = parts[0]
            idx = int(parts[1])
            id2token[idx] = tok
    for i in range(n):
        if id2token[i] is None:
            id2token[i] = f"__missing_{i}__"
    return n, id2token


def main():
    ckpt_path = "/home/naver/MinhPV/sat_rag/OpenKE/checkpoint/transe.ckpt"
    entity2id_path = "/home/naver/MinhPV/sat_rag/OpenKE/benchmarks/FB15K237/entity2id.txt"
    relation2id_path = "/home/naver/MinhPV/sat_rag/OpenKE/benchmarks/FB15K237/relation2id.txt"

    out_ent_path = "data/fb15k237/ge_transe_entity_emb.pt"
    out_rel_path = "data/fb15k237/ge_transe_relation_emb.pt"

    n_ent, _ = load_openke_id_file(entity2id_path)
    n_rel, _ = load_openke_id_file(relation2id_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # OpenKE ckpt can be:
    # - state_dict directly
    # - dict with 'state_dict'
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError(f"Unsupported ckpt type: {type(ckpt)}")

    # -------- entity embedding --------
    ent_key, ent_emb = find_embedding_tensor(
        state,
        num_rows=n_ent,
        prefer_keywords=[("ent", 10), ("entity", 10), ("embed", 3)]
    )
    print(f"[OK] Picked ENTITY key: {ent_key}, shape={tuple(ent_emb.shape)}")

    # -------- relation embedding --------
    rel_key, rel_emb = find_embedding_tensor(
        state,
        num_rows=n_rel,
        prefer_keywords=[("rel", 10), ("relation", 10), ("embed", 3)]
    )
    print(f"[OK] Picked RELATION key: {rel_key}, shape={tuple(rel_emb.shape)}")

    os.makedirs(os.path.dirname(out_ent_path), exist_ok=True)
    torch.save(ent_emb.contiguous(), out_ent_path)
    torch.save(rel_emb.contiguous(), out_rel_path)

    print(f"[OK] Saved entity emb:   {out_ent_path}")
    print(f"[OK] Saved relation emb: {out_rel_path}")


if __name__ == "__main__":
    main()
