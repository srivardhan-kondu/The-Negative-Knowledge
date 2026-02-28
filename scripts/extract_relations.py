import sqlite3
import spacy
from tqdm import tqdm

# Import negspacy first so its factory is registered BEFORE spacy.load()
try:
    import negspacy
    from negspacy.termsets import termset
    NEGSPACY_AVAILABLE = True
except ImportError:
    NEGSPACY_AVAILABLE = False

# load English model for sentence splitting
nlp = spacy.load("en_core_web_sm")
if NEGSPACY_AVAILABLE:
    try:
        ts = termset("en_clinical")
        nlp.add_pipe("negex", config={"ent_types": ["ENTITY", "biomedical_term", "disease_or_drug"]})
    except Exception as e:
        print(f"Warning: Could not add negex pipe: {e}. Negation detection disabled.")
else:
    print("Warning: negspacy not installed. Negation detection is disabled.")

conn = sqlite3.connect("data/mindgap.db")
cur = conn.cursor()

# Get all papers
cur.execute("SELECT paper_id, abstract FROM papers WHERE abstract IS NOT NULL")
papers = cur.fetchall()

# Get categorized entities
cur.execute("""
SELECT paper_id, entity, category 
FROM entities 
WHERE category IS NOT NULL
""")
rows = cur.fetchall()

# Build lookup by paper
paper_entities = {}
for pid, ent, cat in rows:
    paper_entities.setdefault(pid, []).append((ent.lower(), cat))


def save_relation(pid, head, rel, tail):
    cur.execute("""
        INSERT OR IGNORE INTO relations (paper_id, head, relation, tail)
        VALUES (?, ?, ?, ?)
    """, (pid, head, rel, tail))


total_edges = 0

for pid, abstract in tqdm(papers):

    ents = paper_entities.get(pid, [])
    if not ents:
        continue

    doc = nlp(abstract)

    for sent in doc.sents:
        s = sent.text.lower()

        doc_sent = nlp(s)
        
        # Check which entities are present and NOT negated
        valid_entities = []
        
        # For each entity belonging to the paper
        for e, c in ents:
            if e in s:
                is_negated = False
                # If negspacy is active, check if the entity exists in doc_sent.ents and is negated
                if nlp.has_pipe("negex"):
                    for ent in doc_sent.ents:
                        # simple substring check within the entity text
                        if e in ent.text.lower() and getattr(ent._, "negex", False):
                            is_negated = True
                            break
                if not is_negated:
                    valid_entities.append((e, c))

        # build pairwise relations from valid non-negated entities
        for i in range(len(valid_entities)):
            for j in range(i + 1, len(valid_entities)):
                h = valid_entities[i][0].strip()
                t = valid_entities[j][0].strip()

                if h != t:
                    save_relation(pid, h, "related_to", t)
                    total_edges += 1

conn.commit()
conn.close()

print("Co-occurrence relation extraction complete.")
print("Edges created:", total_edges)
