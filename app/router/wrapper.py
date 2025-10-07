import numpy as np

class BGEM3Wrapper:
    """
    Wrapper để BGEM3FlagModel tương thích với SemanticRouter
    """
    def __init__(self, model, normalize=True):
        self.model = model
        self.normalize = normalize

    def embed_documents(self, texts):
        """Embedding nhiều documents"""
        embeddings = self.model.encode(texts)['dense_vecs']
        embeddings = np.array(embeddings).astype('float32')
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-10)
        return embeddings

    def embed_query(self, text):
        """Embedding một query"""
        return self.embed_documents([text])[0]
