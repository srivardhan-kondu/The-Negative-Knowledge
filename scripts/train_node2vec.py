import pickle
from node2vec import Node2Vec

# load graph
with open("data/mental_health_graph.pkl", "rb") as f:
    G = pickle.load(f)

print("Graph loaded:")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# train Node2Vec model
node2vec = Node2Vec(
    G,
    dimensions=64,
    walk_length=20,
    num_walks=200,
    workers=2
)

model = node2vec.fit(window=10, min_count=1)

# save embeddings in gensim's native format (handles spaces in node names)
model.wv.save("data/node_embeddings.wv")

print("Node2Vec training complete. Embeddings saved.")
