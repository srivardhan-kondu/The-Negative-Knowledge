import sqlite3

conn = sqlite3.connect("data/mindgap.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT,
    abstract TEXT,
    year INTEGER,
    authors TEXT,
    venue TEXT,
    source TEXT
)
""")

conn.commit()
conn.close()

print("Database created successfully.")
