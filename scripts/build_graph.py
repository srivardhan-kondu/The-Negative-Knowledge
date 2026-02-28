import sqlite3
import networkx as nx
import pickle

DB = "data/mindgap.db"

conn = sqlite3.connect(DB)
cur = conn.cursor()

# Load nodes (entities)
cur.execute("""
SELECT DISTINCT entity, category 
FROM entities 
WHERE category IS NOT NULL
""")
nodes = cur.fetchall()

# Load edges (relations) and count frequencies to use as weights
cur.execute("""
SELECT head, tail, COUNT(*) as weight
FROM relations
WHERE head != tail
GROUP BY head, tail
""")
edges = cur.fetchall()

conn.close()

# Build graph
G = nx.Graph()

# add nodes with type
for entity, category in nodes:
    G.add_node(entity, category=category)

# add edges with weight
for h, t, w in edges:
    if G.has_edge(h, t):
        G[h][t]['weight'] += w
    else:
        G.add_edge(h, t, relation="related_to", weight=w)

print("Graph created.")
print("Total nodes:", G.number_of_nodes())
print("Total edges:", G.number_of_edges())

# save graph for ML later
with open("data/mental_health_graph.pkl", "wb") as f:
    pickle.dump(G, f)

print("Graph saved to data/mental_health_graph.pkl")
