import sqlite3

conn = sqlite3.connect("data/mindgap.db")
cur = conn.cursor()

cur.execute("SELECT paper_id, entity, type FROM entities LIMIT 50")

rows = cur.fetchall()
for r in rows:
    print(r)

conn.close()
