"""Extract article text and source attribution from official USGS science pages."""
import hashlib

from claim_checker.data.general_science import source_sentences
from claim_checker.schemas import CorpusDocument


def parse_usgs_article(html):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("h1")
    title = " ".join(title.get_text(" ", strip=True).split()) if title else "USGS water science"
    bodies = soup.select(".field--name-body")
    if not bodies:
        bodies = soup.select("article")
    if not bodies:
        raise ValueError("USGS article body not found; refuse to ingest navigation as evidence")
    paragraphs, seen = [], set()
    for body in bodies:
        for element in body.select("script,style,nav,footer,header,aside,figure,figcaption,form,table,button"):
            element.decompose()
        for paragraph in body.find_all(["p", "li"]):
            if paragraph.find(["p", "li"]):
                continue  # Read the innermost paragraph once.
            text = " ".join(paragraph.get_text(" ", strip=True).split())
            if len(text) >= 35 and text not in seen:
                seen.add(text)
                paragraphs.append(text)
    if not paragraphs:
        raise ValueError("USGS page has no usable article paragraphs")
    return title, paragraphs


def usgs_documents(html, url):
    title, paragraphs = parse_usgs_article(html)
    return usgs_text_documents(title, paragraphs, url, hashlib.sha256(html).hexdigest(), "Cached USGS article HTML")


def usgs_text_documents(title, paragraphs, url, checksum, acquisition):
    chunks, current = [], []
    for paragraph in paragraphs:
        for sentence in source_sentences(paragraph):
            if current and (len(current) >= 8 or sum(map(len, current)) + len(sentence) > 2200):
                chunks.append(current)
                current = []
            current.append(sentence)
    if current:
        chunks.append(current)
    source = {"publisher": "U.S. Geological Survey", "collection": "general_science", "subject": "water_science",
              "url": url, "book_title": "Water Science School", "book_url": "https://www.usgs.gov/water-science-school",
              "attribution": f"{title}. U.S. Geological Survey, Water Science School. {url}",
              "license": "See USGS original source terms and credits; images and other media are excluded",
              "source_sha256": checksum, "acquisition": acquisition}
    return [CorpusDocument(doc_id="usgs-" + hashlib.sha256(f"{url}/{i}".encode()).hexdigest()[:20],
                           title=f"USGS Water Science School — {title}", sentences=sentences,
                           text=f"USGS Water Science School — {title}\n{' '.join(sentences)}", source={**source, "chunk": i})
            for i, sentences in enumerate(chunks)]
