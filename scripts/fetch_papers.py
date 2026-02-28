"""
Multi-Source Paper Fetcher for MindGap Explorer
Fetches mental health research papers from:
1. Semantic Scholar
2. PubMed (via Entrez API)
3. arXiv (mental health/psychology papers)
4. CORE (open access research)
"""

import requests
import sqlite3
from tqdm import tqdm
import time
from xml.etree import ElementTree as ET

DB_PATH = "data/mindgap.db"

SEARCH_TERMS = [
    "depression mental health",
    "anxiety disorder treatment",
    "PTSD therapy",
    "suicidal ideation risk",
    "cognitive behavioral therapy",
    "dialectical behavior therapy",
    "bipolar disorder treatment",
    "mindfulness therapy",
    "trauma mental health",
    "loneliness depression"
]

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# ============================================================
# SOURCE 1: Semantic Scholar
# ============================================================

def fetch_semantic_scholar():
    """Fetch papers from Semantic Scholar API"""
    print("\n" + "="*70)
    print("📚 SOURCE 1: Semantic Scholar")
    print("="*70)
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    FIELDS = "title,abstract,year,authors,venue"
    total_papers = 0
    
    for term in SEARCH_TERMS:
        print(f"\n🔍 Searching: {term}")
        offset = 0
        
        for page in tqdm(range(5), desc="Pages"):  # 5 pages = 500 papers per term
            try:
                params = {
                    "query": term,
                    "limit": 100,
                    "offset": offset,
                    "fields": FIELDS
                }
                
                response = requests.get(BASE_URL, params=params, timeout=10)
                data = response.json()
                
                for paper in data.get("data", []):
                    pid = paper.get("paperId")
                    title = paper.get("title")
                    abstract = paper.get("abstract")
                    year = paper.get("year")
                    venue = paper.get("venue")
                    authors = ", ".join([a.get("name", "") for a in paper.get("authors", [])])
                    
                    if pid and title and abstract:
                        cur.execute("""
                            INSERT OR REPLACE INTO papers
                            (paper_id, title, abstract, year, authors, venue, source)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (pid, title, abstract, year, authors, venue, "Semantic Scholar"))
                        total_papers += 1
                
                conn.commit()
                offset += 100
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"  ⚠️  Error: {e}")
                continue
    
    print(f"\n✅ Semantic Scholar: {total_papers} papers fetched")
    return total_papers

# ============================================================
# SOURCE 2: PubMed
# ============================================================

def fetch_pubmed():
    """Fetch papers from PubMed via Entrez API"""
    print("\n" + "="*70)
    print("📚 SOURCE 2: PubMed (NIH)")
    print("="*70)
    
    BASE_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    BASE_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    total_papers = 0
    
    for term in SEARCH_TERMS:
        print(f"\n🔍 Searching: {term}")
        
        try:
            # Search for paper IDs
            search_params = {
                "db": "pubmed",
                "term": f"{term} AND (abstract[sb])",
                "retmax": 50,  # 50 papers per term
                "retmode": "json"
            }
            
            search_response = requests.get(BASE_SEARCH, params=search_params, timeout=10)
            search_data = search_response.json()
            
            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            
            if not id_list:
                print("  No results found")
                continue
            
            # Fetch details for each paper
            for pmid in tqdm(id_list, desc="Fetching"):
                try:
                    fetch_params = {
                        "db": "pubmed",
                        "id": pmid,
                        "retmode": "xml"
                    }
                    
                    fetch_response = requests.get(BASE_FETCH, params=fetch_params, timeout=10)
                    root = ET.fromstring(fetch_response.content)
                    
                    # Extract fields
                    article = root.find(".//Article")
                    if article is None:
                        continue
                    
                    title_elem = article.find(".//ArticleTitle")
                    abstract_elem = article.find(".//AbstractText")
                    year_elem = article.find(".//PubDate/Year")
                    journal_elem = article.find(".//Journal/Title")
                    
                    title = title_elem.text if title_elem is not None else None
                    abstract = abstract_elem.text if abstract_elem is not None else None
                    year = year_elem.text if year_elem is not None else None
                    venue = journal_elem.text if journal_elem is not None else "PubMed"
                    
                    # Extract authors
                    authors_list = []
                    for author in article.findall(".//Author"):
                        lastname = author.find("LastName")
                        forename = author.find("ForeName")
                        if lastname is not None:
                            name = lastname.text
                            if forename is not None:
                                name = f"{forename.text} {name}"
                            authors_list.append(name)
                    authors = ", ".join(authors_list[:5])  # Limit to first 5 authors
                    
                    if title and abstract:
                        cur.execute("""
                            INSERT OR REPLACE INTO papers
                            (paper_id, title, abstract, year, authors, venue, source)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (f"PMID{pmid}", title, abstract, year, authors, venue, "PubMed"))
                        total_papers += 1
                    
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    continue
            
            conn.commit()
            
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            continue
    
    print(f"\n✅ PubMed: {total_papers} papers fetched")
    return total_papers

# ============================================================
# SOURCE 3: arXiv (Psychology papers)
# ============================================================

def fetch_arxiv():
    """Fetch psychology/mental health papers from arXiv"""
    print("\n" + "="*70)
    print("📚 SOURCE 3: arXiv (Psychology)")
    print("="*70)
    
    BASE_URL = "http://export.arxiv.org/api/query"
    total_papers = 0
    
    # arXiv categories related to psychology/mental health
    categories = ["q-bio.NC", "cs.CY", "stat.AP"]  # Neuroscience, Computers & Society, Applications
    
    for term in SEARCH_TERMS[:5]:  # Limit to first 5 terms
        print(f"\n🔍 Searching: {term}")
        
        try:
            params = {
                "search_query": f"all:{term}",
                "start": 0,
                "max_results": 30,  # 30 papers per term
                "sortBy": "relevance"
            }
            
            response = requests.get(BASE_URL, params=params, timeout=10)
            root = ET.fromstring(response.content)
            
            # Namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in tqdm(root.findall('atom:entry', ns), desc="Processing"):
                try:
                    arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
                    title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                    summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                    published = entry.find('atom:published', ns).text[:4]  # Year
                    
                    # Authors
                    authors_list = []
                    for author in entry.findall('atom:author', ns):
                        name = author.find('atom:name', ns).text
                        authors_list.append(name)
                    authors = ", ".join(authors_list[:5])
                    
                    if title and summary and len(summary) > 100:  # Ensure decent abstract
                        cur.execute("""
                            INSERT OR REPLACE INTO papers
                            (paper_id, title, abstract, year, authors, venue, source)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (f"arXiv{arxiv_id}", title, summary, published, authors, "arXiv", "arXiv"))
                        total_papers += 1
                        
                except Exception:
                    continue
            
            conn.commit()
            time.sleep(3)  # arXiv rate limiting
            
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            continue
    
    print(f"\n✅ arXiv: {total_papers} papers fetched")
    return total_papers

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n🚀 Multi-Source Paper Fetcher")
    print("="*70)
    print("Fetching mental health research papers from 3 sources...")
    print("="*70)
    
    total_all = 0
    
    # Fetch from all sources
    total_all += fetch_semantic_scholar()
    total_all += fetch_pubmed()
    total_all += fetch_arxiv()
    
    conn.close()
    
    print("\n" + "="*70)
    print("✅ MULTI-SOURCE FETCH COMPLETE!")
    print("="*70)
    print(f"\n📊 Total papers collected: {total_all}")
    print(f"📁 Saved to: {DB_PATH}")
    print("\n💡 Next steps:")
    print("   1. Run: python scripts/extract_entities.py")
    print("   2. Run: python scripts/classify_entities.py")
    print("   3. Continue with the full pipeline...")
    print("="*70)
