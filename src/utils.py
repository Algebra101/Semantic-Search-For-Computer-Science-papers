import re


def clean_basic_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\\S+|www\\.\\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def merge_text_fields(title: str, abstract: str, keywords: str, category: str) -> str:
    parts = [
        str(title).strip(),
        str(abstract).strip(),
        str(keywords).strip(),
        str(category).strip(),
    ]
    return " ".join([p for p in parts if p and p.lower() != "nan"])