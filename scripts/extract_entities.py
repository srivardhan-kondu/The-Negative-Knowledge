import spacy
import sqlite3
from tqdm import tqdm

# Load ML models
sci_nlp = spacy.load("en_core_sci_sm")
disease_nlp = spacy.load("en_ner_bc5cdr_md")

conn = sqlite3.connect("data/mindgap.db")
cur = conn.cursor()

cur.execute("SELECT paper_id, abstract FROM papers WHERE abstract IS NOT NULL")
papers = cur.fetchall()

# Deduplication map for synonyms
synonyms = {
    "major depressive disorder": "depression",
    "depressive symptoms": "depression",
    "ptsd": "post-traumatic stress disorder",
    "cbt": "cognitive behavioral therapy",
    "dbt": "dialectical behavior therapy",
    "severe anxiety": "anxiety",
    "anxiety disorders": "anxiety"
}

def save_entity(pid, text, etype, source):
    # Normalize entity text (lowercase, strip, lemmatize map)
    norm_text = text.lower().strip()
    norm_text = synonyms.get(norm_text, norm_text)

    cur.execute("""
        INSERT OR IGNORE INTO entities (paper_id, entity, type, source)
        VALUES (?, ?, ?, ?)
    """, (pid, norm_text, etype, source))

for pid, abstract in tqdm(papers):

    doc = sci_nlp(abstract)
    for ent in doc.ents:
        save_entity(pid, ent.text.strip(), "biomedical_term", "sci_sm")

    ddoc = disease_nlp(abstract)
    for ent in ddoc.ents:
        save_entity(pid, ent.text.strip(), "disease_or_drug", "bc5cdr")

conn.commit()
conn.close()

print("Entity extraction complete.")
