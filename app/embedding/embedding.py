# embedding_store.py
import os
import json
import numpy as np
import faiss
from FlagEmbedding import BGEM3FlagModel

class EmbeddingStore:
    def __init__(self, data_path, index_path, model_name="BAAI/bge-m3", k=3, use_fp16=True):
        self.data_path = data_path
        self.index_path = index_path
        self.model_name = model_name
        self.k = k

        # Khởi tạo model
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

        # Load documents
        self.documents = self.load_documents()

        # Load hoặc tạo FAISS index
        self.index = self.load_or_create_index()

    def load_documents(self):
        docs = []
        if os.path.exists(self.data_path):
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        docs.append(item["text"])
                    except json.JSONDecodeError:
                        continue
        else:
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {self.data_path}")
        return docs

    def encode(self, texts, batch_size=12, max_length=8192):
        """Encode list of texts thành dense vectors"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length
        )['dense_vecs']
        embeddings = np.array(embeddings).astype('float32')
        # normalize để dùng inner product ≈ cosine similarity
        faiss.normalize_L2(embeddings)
        return embeddings

    def load_or_create_index(self):
        if os.path.exists(self.index_path):
            print("🔹 Loading existing FAISS index...")
            index = faiss.read_index(self.index_path)
        else:
            print("🔹 Creating new FAISS index...")
            doc_embeddings = self.encode(self.documents)
            dim = doc_embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)  # Inner Product
            index.add(doc_embeddings)
            faiss.write_index(index, self.index_path)
        return index

    def retrieve(self, query, top_k=None):
        """Trả về top k documents gần query"""
        if top_k is None:
            top_k = self.k
        query_emb = self.encode([query])
        distances, indices = self.index.search(query_emb, top_k)
        results = [self.documents[i] for i in indices[0]]
        return results
