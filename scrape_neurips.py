# scrape_neurips.py
import os
import time
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient, errors
from urllib.parse import urljoin

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "neurips")
COLL_NAME = os.getenv("COLL_NAME", "papers")

# === SET THIS ===
NEURIPS_URL = "https://papers.neurips.cc/paper/2024" 
# <-- Replace with the real NeurIPS 2024 proceedings URL (an index page listing all papers)

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
coll = db[COLL_NAME]

def ensure_indexes():
    # Unique index to avoid duplicates based on title (or link)
    try:
        coll.create_index("link", unique=True)
        # For keyword search, create a text index on title and authors.
        coll.create_index([("title", "text"), ("authors", "text")], default_language="english")
    except errors.PyMongoError as e:
        print("Index error:", e)

def parse_listing_page(url):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    papers = []

    # The HTML structure varies by site. Inspect the proceedings page and adjust selectors.
    # Below are a few approaches; you probably need to tweak based on actual page.

    # ATTEMPT 1: If the site lists papers in <li> or <div class="paper"> blocks:
    for item in soup.select("li.paper, div.paper, div.media"):  # try a few common selectors
        # Try to get title & link:
        title_tag = item.find("a")
        if not title_tag:
            continue
        title = title_tag.get_text(strip=True)
        link = urljoin(url, title_tag.get("href"))
        # authors may be in a <span class="authors"> or small tag
        authors_tag = item.find(class_="authors") or item.find("span", {"class": "author"}) or item.find("p", {"class": "authors"})
        if authors_tag:
            authors_text = authors_tag.get_text(" ", strip=True)
            authors = [a.strip() for a in authors_text.split(",") if a.strip()]
        else:
            # fallback: sometimes authors are in the next <p> or small tag
            next_text = item.get_text(" ", strip=True)
            authors = []
        papers.append({"title": title, "link": link, "authors": authors})

    # If previous selector didn't return items, try scanning all anchors that look like paper titles:
    if not papers:
        for a in soup.select("a"):
            href = a.get("href") or ""
            text = a.get_text(strip=True)
            if not text:
                continue
            # heuristic: title length > 10 and href contains 'paper' or endswith '.pdf'
            if len(text) > 10 and ("paper" in href.lower() or href.lower().endswith(".pdf")):
                link = urljoin(url, href)
                # try to find sibling containing authors
                parent = a.parent
                authors = []
                if parent:
                    # look for immediate following sibling text nodes
                    sib = parent.find_next_sibling()
                    if sib:
                        authors_text = sib.get_text(" ", strip=True)
                        if authors_text and len(authors_text) < 200:
                            authors = [x.strip() for x in authors_text.split(",") if x.strip()]
                papers.append({"title": text, "link": link, "authors": authors})

    return papers

def save_papers(papers):
    inserted = 0
    for p in papers:
        doc = {
            "title": p["title"],
            "authors": p["authors"],
            "link": p["link"],
            "scraped_at": time.time()
        }
        try:
            coll.insert_one(doc)
            inserted += 1
        except errors.DuplicateKeyError:
            # already exists
            continue
        except Exception as e:
            print("Insert error:", e)
    return inserted

def main():
    ensure_indexes()
    print("Scraping:", NEURIPS_URL)
    papers = parse_listing_page(NEURIPS_URL)
    print(f"Found {len(papers)} candidates. Inserting to MongoDB...")
    n = save_papers(papers)
    print(f"Inserted {n} new documents.")

if __name__ == "__main__":
    main()
