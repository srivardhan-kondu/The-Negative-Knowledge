import sqlite3

conn = sqlite3.connect("data/mindgap.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT,
    entity TEXT,
    type TEXT,
    source TEXT,
    category TEXT,
    UNIQUE(paper_id, entity, type)
)
""")

conn.commit()
conn.close()

print("Entities table created.")
