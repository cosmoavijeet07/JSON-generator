import re
try:
    import spacy
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

def split_semantic(text, max_len=3000, overlap=200):
    if HAS_SPACY:
        doc = nlp(text)
        sents = [sent.text for sent in doc.sents]
    else:
        sents = re.split(r'(?<=[.?!])\s+(?=[A-Z])', text)
    chunks, cur, tokens = [], [], 0
    def tokens_count(txt): return len(txt.split())
    for sent in sents:
        stokens = tokens_count(sent)
        if tokens + stokens > max_len:
            chunks.append(" ".join(cur))
            cur = cur[-(overlap//(tokens//len(cur) or 1)):] if cur else []
            tokens = tokens_count(" ".join(cur))
        cur.append(sent)
        tokens += stokens
    if cur: chunks.append(" ".join(cur))
    return [ch.strip() for ch in chunks if ch.strip()]
