from sentence_transformers import SentenceTransformer
import numpy as np

class FieldChunkRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunk_embeds = self.encoder.encode(chunks, convert_to_numpy=True)

    def retrieve(self, field_query, top_n=2):
        fq_embed = self.encoder.encode([field_query], convert_to_numpy=True)[0]
        sims = np.dot(self.chunk_embeds, fq_embed)
        ixs = np.argsort(-sims)[:top_n]
        return [self.chunks[i] for i in ixs]
