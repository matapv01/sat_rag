import networkx as nx

def get_k_hop_subgraph(G, anchors, k=2, max_nodes=200):
    nodes = set()
    for a in anchors:
        if a in G:
            nodes |= set(nx.single_source_shortest_path_length(G, a, cutoff=k).keys())
    subG = G.subgraph(list(nodes))
    if len(subG) > max_nodes:
        nodes_sorted = sorted(subG.degree, key=lambda x: x[1], reverse=True)
        top_nodes = [n for n,_ in nodes_sorted[:max_nodes]]
        subG = subG.subgraph(top_nodes)
    return subG
