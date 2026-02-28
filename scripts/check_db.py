import sqlite3

conn = sqlite3.connect("data/mindgap.db")
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM papers")
count = cur.fetchone()[0]

print("Total papers stored:", count)

print("\nSample rows:\n")
for row in cur.execute("SELECT title, year FROM papers LIMIT 10"):
    print(row)

conn.close()
