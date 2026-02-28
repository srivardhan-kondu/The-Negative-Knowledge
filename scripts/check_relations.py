import sqlite3

conn = sqlite3.connect("data/mindgap.db")
cur = conn.cursor()

print("Total relations:", cur.execute("SELECT COUNT(*) FROM relations").fetchone()[0])

for row in cur.execute("SELECT head, relation, tail FROM relations LIMIT 20"):
    print(row)

conn.close()
