# import sqlite3

# DISORDER = ["depression","anxiety","ptsd","bipolar",
#             "panic","ocd","schizophrenia","phobia"]

# THERAPY = ["therapy","cbt","dbt","treatment",
#            "counseling","psychotherapy","mindfulness",
#            "ssri","antidepressant","medication"]

# RISK = ["trauma","abuse","stress","insomnia",
#         "sleep","loneliness","poverty","bullying"]

# OUTCOME = ["suicide","relapse","recovery",
#            "self harm","quality of life","mortality","ideation"]

# POP = ["adolescent","child","teen","student",
#        "women","men","adult","elderly","veteran"]


# def classify(text):
#     t = text.lower()

#     if any(k in t for k in DISORDER):
#         return "disorder"
#     if any(k in t for k in THERAPY):
#         return "therapy"
#     if any(k in t for k in RISK):
#         return "risk_factor"
#     if any(k in t for k in OUTCOME):
#         return "outcome"
#     if any(k in t for k in POP):
#         return "population"

#     return None


# conn = sqlite3.connect("data/mindgap.db")
# cur = conn.cursor()

# cur.execute("SELECT id, entity FROM entities")
# rows = cur.fetchall()

# for _id, ent in rows:
#     cat = classify(ent)
#     if cat:
#         cur.execute(
#             "UPDATE entities SET category=? WHERE id=?",
#             (cat, _id)
#         )

# conn.commit()
# conn.close()

# print("Classification complete.")
import sqlite3

conn = sqlite3.connect("data/mindgap.db")
cur = conn.cursor()

cur.execute("SELECT entity, category FROM entities WHERE category IS NOT NULL LIMIT 20")
for row in cur.fetchall():
    print(row)

conn.close()
