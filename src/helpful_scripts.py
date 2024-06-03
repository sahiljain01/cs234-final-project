import networkx as nx
edges, _ = create_graph()
G = nx.Graph()

edges = edges.detach().numpy()
edge_to, edge_from = edges
for a, b in zip(edge_to, edge_from):
    G.add_edge(a, b)
    G.add_edge(a, 28)
    G.add_edge(b, 28)

mapping = {env.STOCKS.index(s): s for s in env.STOCKS}
mapping[28] = "DOW"
G = nx.relabel_nodes(G, mapping)
from pylab import rcParams
rcParams['figure.figsize'] = 14, 10
pos = nx.spring_layout(G, scale=20, k=3/np.sqrt(G.order()))
d = dict(G.degree)
nx.draw(G, pos, node_color='lightblue', 
        with_labels=True, 
        nodelist=d, 
        node_size=[d[k]*300 for k in d])

