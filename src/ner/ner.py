from __future__ import annotations
import re
from typing import Dict, List
from langchain.schema import Document
import spacy


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
  
    nlp = spacy.blank("en")


_money_re = re.compile(r"(USD|EUR|\$|£)\s?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?", re.I)


def extract_entities_from_text(text: str) -> Dict[str, List[str]]:
    doc = nlp(text)
    orgs = [ent.text for ent in doc.ents if ent.label_ in ("ORG",)]
    persons = [ent.text for ent in doc.ents if ent.label_ in ("PERSON",)]
    parties = list(dict.fromkeys(orgs + persons))

    dates = [ent.text for ent in doc.ents if ent.label_ in ("DATE",)]
    money = [ent.text for ent in doc.ents if ent.label_ == "MONEY"]
    money += _money_re.findall(text)

 
    law = []
    law_re = re.compile(r"governing law|jurisdiction|venue", re.I)
    for sent in re.split(r"(?<=[\.\!\?])\s+", text):
        if law_re.search(sent):
            law.append(sent.strip())

    return {
        "parties": list(dict.fromkeys(parties))[:10],
        "dates": list(dict.fromkeys(dates))[:10],
        "money": list(dict.fromkeys(money))[:10],
        "governing_law_mentions": law[:5],
    }


def extract_entities_from_docs(docs: List[Document]) -> Dict[str, List[str]]:
    merged = "\n\n".join(d.page_content for d in docs)
    return extract_entities_from_text(merged)