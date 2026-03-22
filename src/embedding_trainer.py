import numpy as np
from gensim.models import Word2Vec


class EmbeddingTrainer:
    def __init__(self, vector_size=100, window=5, min_count=1, epochs=20):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs

    def train(self, corpus: list[list[str]]) -> Word2Vec:
        model = Word2Vec(
            sentences=corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            sg=1,   # Skip-gram
            epochs=self.epochs,
            seed=42
        )
        return model


class VectorBuilder:
    def __init__(self, model: Word2Vec):
        self.model = model
        self.dim = model.vector_size

    def document_vector(self, tokens: list[str]) -> np.ndarray:
        vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]

        if not vectors:
            return np.zeros(self.dim, dtype=np.float32)

        return np.mean(vectors, axis=0).astype(np.float32)

    def build_matrix(self, corpus: list[list[str]]) -> np.ndarray:
        return np.vstack([self.document_vector(tokens) for tokens in corpus])