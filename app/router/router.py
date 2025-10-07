import numpy as np
from typing import List

class Route:
    def __init__(self, name: str = None, samples: List[str] = None):
        self.name = name
        self.samples = samples or []

class SemanticRouter:
    def __init__(self, embedding, routes: List[Route]):
        self.routes = routes
        self.embedding = embedding
        self.routesEmbedding = {}

        # Tạo embedding cho từng route
        for route in self.routes:
            self.routesEmbedding[route.name] = self.embedding.embed_documents(route.samples)

    def guide(self, query):
        queryEmbedding = self.embedding.embed_query(query)
        # normalize query embedding
        queryEmbedding = queryEmbedding / (np.linalg.norm(queryEmbedding) + 1e-10)

        scores = []

        for route in self.routes:
            routeEmb = self.routesEmbedding[route.name]
            # normalize route embeddings
            routeEmb = routeEmb / (np.linalg.norm(routeEmb, axis=1, keepdims=True) + 1e-10)
            # cosine similarity
            score = np.mean(np.dot(routeEmb, queryEmbedding.T).flatten())
            scores.append((score, route.name))

        # sort descending
        scores.sort(reverse=True)
        return scores[0]
