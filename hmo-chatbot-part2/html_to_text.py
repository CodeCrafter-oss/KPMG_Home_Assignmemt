# html_to_text.py
from __future__ import annotations
from bs4 import BeautifulSoup
from typing import List
import re

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    for table in soup.find_all("table"):
        rows: List[str] = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with("\n".join(rows) + "\n")

    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{2,}", "\n", text)
    return text
