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
    """
    Split text into semantic chunks with strict word/token limits
    """
    if HAS_SPACY:
        doc = nlp(text)
        sents = [sent.text.strip() for sent in doc.sents]
    else:
        sents = re.split(r'(?<=[.?!])\s+(?=[A-Z])', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    def word_count(txt):
        return len(txt.split())
    
    for sentence in sents:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_length = word_count(sentence)
        
        # If adding this sentence would exceed max_len, finalize current chunk
        if current_length + sentence_length > max_len and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            
            # Apply overlap by keeping last 'overlap' words
            if overlap > 0 and chunk_text:
                words = chunk_text.split()
                if len(words) > overlap:
                    overlap_words = words[-overlap:]
                    current_chunk = [" ".join(overlap_words)]
                    current_length = len(overlap_words)
                else:
                    current_chunk = [chunk_text]
                    current_length = len(words)
            else:
                current_chunk = []
                current_length = 0
        
        # If single sentence exceeds max_len, split it at word level
        if sentence_length > max_len:
            words = sentence.split()
            for i in range(0, len(words), max_len):
                word_chunk = " ".join(words[i:i + max_len])
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                chunks.append(word_chunk)
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Add final chunk if exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def validate_chunks(chunks, max_len):
    """
    Validate that all chunks respect the max_len constraint
    """
    for i, chunk in enumerate(chunks):
        word_count = len(chunk.split())
        if word_count > max_len:
            raise ValueError(f"Chunk {i+1} has {word_count} words, exceeds max_len of {max_len}")
    return True
