import numpy as np


from typing import List
class Route():
    def __init__(
        self,
        name: str = None,
        samples:List = []
    ):

        self.name = name
        self.samples = samples


class SemanticRouter():
    def __init__(self, embedding, routes):
        # [productRoute, chitchatRoute]
        self.routes = routes
        self.embedding = embedding
        self.routesEmbedding = {

        }

        for route in self.routes:
            self.routesEmbedding[
                route.name
            ] = self.embedding.embed_documents(route.samples)
    
    def guide(self, query):
        queryEmbedding = np.array(self.embedding.embed_query(query))
        queryEmbedding = queryEmbedding / np.linalg.norm(queryEmbedding)
        scores = []

        # Calculate the cosine similarity of the query embedding with the sample embeddings of the router.

        for route in self.routes:
            routesEmbedding = self.routesEmbedding[route.name] / np.linalg.norm(self.routesEmbedding[route.name])
            score = np.mean(np.dot(routesEmbedding, queryEmbedding.T).flatten())
            scores.append((score, route.name))

        scores.sort(reverse=True)
        return scores[0]