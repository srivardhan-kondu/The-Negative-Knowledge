"""
Bulk Paper Fetcher — Targets 5,000+ papers for dense graph learning.
Uses Semantic Scholar (20 terms × 25 pages × 100/page = 50,000 candidates)
and PubMed (20 terms × 200 results) for maximum coverage.
"""
import requests
import sqlite3
from tqdm import tqdm
import time
from xml.etree import ElementTree as ET

DB_PATH = "data/mindgap.db"

# Expanded search terms: 20 domain-specific queries
SEARCH_TERMS = [
    "depression mental health treatment",
    "anxiety disorder therapy outcomes",
    "PTSD post traumatic stress disorder",
    "suicidal ideation prevention risk",
    "cognitive behavioral therapy CBT",
    "bipolar disorder lithium treatment",
    "schizophrenia antipsychotic medication",
    "obsessive compulsive disorder OCD treatment",
    "mindfulness meditation anxiety",
    "social anxiety disorder phobia",
    "major depressive disorder antidepressant",
    "insomnia sleep disorder mental health",
    "eating disorder anorexia bulimia",
    "attention deficit hyperactivity disorder ADHD",
    "substance abuse addiction mental health",
    "borderline personality disorder DBT",
    "trauma childhood adversity mental health",
    "psychotherapy intervention mental illness",
    "stress cortisol mental health",
    "emotional regulation emotion dysregulation",
]

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

def fetch_semantic_scholar_bulk():
    """Fetch ~25,000 papers from Semantic Scholar at 100/page × 25 pages × 10 terms."""
    print("\n" + "="*70)
    print("📚 Semantic Scholar — Bulk Fetch (25 pages × 20 terms)")
    print("="*70)
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    FIELDS = "title,abstract,year,authors,venue"
    total = 0
    
    for term in SEARCH_TERMS:
        print(f"\n🔍 '{term}'")
        for page in tqdm(range(25), desc="Pages"):
            try:
                r = requests.get(BASE_URL, params={
                    "query": term,
                    "limit": 100,
                    "offset": page * 100,
                    "fields": FIELDS
                }, timeout=15)
                
                if r.status_code == 429:
                    print("  Rate limited. Waiting 60s...")
                    time.sleep(60)
                    continue
                    
                data = r.json()
                for paper in data.get("data", []):
                    pid = paper.get("paperId")
                    title = paper.get("title")
                    abstract = paper.get("abstract")
                    if pid and title and abstract and len(abstract) > 100:
                        cur.execute("""
                            INSERT OR IGNORE INTO papers
                            (paper_id, title, abstract, year, authors, venue, source)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            pid, title, abstract,
                            paper.get("year"),
                            ", ".join([a.get("name","") for a in paper.get("authors",[])[:5]]),
                            paper.get("venue",""),
                            "SemanticScholar"
                        ))
                        total += 1
                
                conn.commit()
                time.sleep(1.2)  # Respect rate limits
                
            except Exception as e:
                print(f"  ⚠ {e}")
                time.sleep(3)
                continue
    
    print(f"\n✅ Semantic Scholar total: {total} new papers")
    return total


def fetch_pubmed_bulk():
    """Fetch ~4,000 papers from PubMed (200 per term × 20 terms)."""
    print("\n" + "="*70)
    print("📚 PubMed — Bulk Fetch (200 per term × 20 terms)")
    print("="*70)
    
    BASE_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    BASE_FETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    total = 0

    for term in SEARCH_TERMS:
        print(f"\n🔍 '{term}'")
        try:
            search_r = requests.get(BASE_SEARCH, params={
                "db": "pubmed",
                "term": f"{term} AND (abstract[sb])",
                "retmax": 200,
                "retmode": "json"
            }, timeout=15)
            ids = search_r.json().get("esearchresult", {}).get("idlist", [])
            
            for pmid in tqdm(ids, desc="Fetching"):
                try:
                    fr = requests.get(BASE_FETCH, params={
                        "db": "pubmed", "id": pmid, "retmode": "xml"
                    }, timeout=10)
                    root = ET.fromstring(fr.content)
                    article = root.find(".//Article")
                    if article is None:
                        continue
                    
                    title_e = article.find(".//ArticleTitle")
                    abstract_e = article.find(".//AbstractText")
                    year_e = article.find(".//PubDate/Year")
                    journal_e = article.find(".//Journal/Title")
                    
                    title = title_e.text if title_e is not None else None
                    abstract = abstract_e.text if abstract_e is not None else None
                    
                    if title and abstract and len(abstract) > 100:
                        authors = []
                        for a in article.findall(".//Author")[:5]:
                            ln = a.find("LastName")
                            fn = a.find("ForeName")
                            if ln is not None:
                                authors.append(f"{fn.text if fn is not None else ''} {ln.text}".strip())
                        
                        cur.execute("""
                            INSERT OR IGNORE INTO papers
                            (paper_id, title, abstract, year, authors, venue, source)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            f"PMID{pmid}", title, abstract,
                            year_e.text if year_e is not None else None,
                            ", ".join(authors),
                            journal_e.text if journal_e is not None else "PubMed",
                            "PubMed"
                        ))
                        total += 1
                    
                    time.sleep(0.4)
                except:
                    continue
            
            conn.commit()
            time.sleep(1)
            
        except Exception as e:
            print(f"  ⚠ {e}")
            continue
    
    print(f"\n✅ PubMed total: {total} new papers")
    return total


if __name__ == "__main__":
    # Show current count
    cur.execute("SELECT COUNT(*) FROM papers")
    before = cur.fetchone()[0]
    print(f"📊 Papers before fetch: {before}")
    print("\n🚀 Starting bulk paper fetch — this will take 10-20 minutes...")
    
    t1 = fetch_semantic_scholar_bulk()
    t2 = fetch_pubmed_bulk()
    
    cur.execute("SELECT COUNT(*) FROM papers")
    after = cur.fetchone()[0]
    
    conn.close()
    print(f"\n{'='*70}")
    print(f"✅ BULK FETCH COMPLETE")
    print(f"   Before: {before} papers")
    print(f"   After:  {after} papers (+{after - before} new)")
    print(f"{'='*70}")
