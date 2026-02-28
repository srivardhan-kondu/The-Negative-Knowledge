import sqlite3

conn = sqlite3.connect("data/mindgap.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT,
    head TEXT,
    relation TEXT,
    tail TEXT,
    UNIQUE(paper_id, head, relation, tail)
)
""")

conn.commit()
conn.close()

print("Relations table created.")
