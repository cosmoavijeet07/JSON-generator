import re

def extract_headings(text):
    return [(m.start(), m.group().strip()) for m in re.finditer(r"\n\s*(\d{0,2}\.?\s?[A-Z][\w \-\(\)]{3,})\n", text)]

def extract_toc(text):
    toc_match = re.search(r'Table of Contents(.*?)(\n[A-Z][^\n]+\n){3,}', text, flags=re.IGNORECASE|re.DOTALL)
    if not toc_match: return None
    items = re.findall(r'([A-Z][\w \-]+)\.{0,}\s?\d+', toc_match.group(1))
    return items

def clean_ocr(text):
    text = re.sub(r'\n\d{1,3}\n', '\n', text)   # remove page numbers
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove non-ASCII
    return text.strip()
