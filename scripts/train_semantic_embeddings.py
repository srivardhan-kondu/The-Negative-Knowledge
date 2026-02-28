import sqlite3
import torch
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def main():
    print("Loading SciBERT model (this may download ~440MB if first time)...")
    # Using a model pre-trained on scientific literature
    model = SentenceTransformer('allenai/scibert_scivocab_uncased')
    
    conn = sqlite3.connect("data/mindgap.db")
    cur = conn.cursor()
    
    # Get all distinct entities and their categories from the database
    cur.execute("""
    SELECT DISTINCT entity, category 
    FROM entities 
    WHERE category IS NOT NULL
    """)
    rows = cur.fetchall()
    conn.close()
    
    print(f"Loaded {len(rows)} entities. Generating semantic embeddings...")
    
    # We embed the entity text combined with its category for extra context
    # e.g., "depression (disorder)"
    texts_to_embed = [f"{ent} ({cat})" for ent, cat in rows]
    entities = [ent for ent, cat in rows]
    
    # Generate embeddings (SciBERT outputs 768-dimensional vectors)
    embeddings = model.encode(texts_to_embed, show_progress_bar=True)
    
    # Create an embedding dictionary mapped by entity name
    emb_dict = {}
    for i, ent in enumerate(entities):
        emb_dict[ent] = embeddings[i]
        
    # Save the embeddings dictionary
    with open("data/semantic_embeddings.pkl", "wb") as f:
        pickle.dump(emb_dict, f)
        
    print(f"Successfully saved 768-dimensional SciBERT embeddings for {len(emb_dict)} entities.")

if __name__ == "__main__":
    main()
