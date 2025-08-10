from src.ner.ner import extract_entities_from_text

def test_ner_basic():
    text = "This Agreement is effective as of January 1, 2024 between Acme Corp and John Doe. The fee is $10,000."
    ents = extract_entities_from_text(text)
    assert "Acme Corp" in " ".join(ents["parties"]) or "John Doe" in " ".join(ents["parties"])
    assert any("$10,000" in m or "10,000" in m for m in ents["money"])