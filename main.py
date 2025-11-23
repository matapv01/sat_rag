import torch
from torch.utils.data import DataLoader
from torch import optim, nn

from src.data_loader import FB15K237Graph, LocalAlignmentDataset
from src.graph_encoder import GraphEncoder
from src.text_encoder import TextEncoder
from src.hka_trainer import train_hka
from src.knowledge_adapter import KnowledgeAdapter


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load KG ---
    graph_obj = FB15K237Graph(data_dir="data/fb15k237")
    print(f"Loaded KG: {len(graph_obj.train_triples)} train triples, "
          f"{len(graph_obj.test_triples)} test triples, {len(graph_obj.entities)} entities")

    edges = [[graph_obj.entity2id[h], graph_obj.entity2id[t]] 
             for h, r, t in graph_obj.train_triples]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    node_feats = torch.randn(len(graph_obj.entities), 128)

    # --- Graph Encoder ---
    graph_encoder = GraphEncoder(in_dim=128, hidden_dim=256, out_dim=256).to(device)
    graph_encoder.node_feats = node_feats
    graph_encoder.edge_index = edge_index

    # --- Knowledge Adapter ---
    adapter = KnowledgeAdapter(in_dim=256, hidden_dim=128, out_dim=256).to(device)

    # --- Text Encoder ---
    text_encoder = TextEncoder(entity2text=graph_obj.entity2text)
    text_adapter = nn.Linear(768, 256).to(device)

    # --- Dataset ---
    dataset = LocalAlignmentDataset(graph_obj)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"Dataloader created: {len(dataloader)} batches")

    # --- Optimizer ---
    optimizer = optim.Adam(
        list(graph_encoder.parameters()) +
        list(adapter.parameters()) +
        list(text_encoder.encoder.parameters()) +
        list(text_adapter.parameters()),
        lr=1e-4
    )

    # --- Training ---
    for epoch in range(5):
        print(f"\n=== Epoch {epoch+1} ===")
        loss = train_hka(
            graph_encoder,
            text_encoder,
            dataloader,
            optimizer,
            device,
            adapter,
            text_adapter=text_adapter,
            id2entity=graph_obj.id2entity
        )
        print(f"Epoch {epoch+1} finished: loss={loss:.4f}")


if __name__ == "__main__":
    main()
