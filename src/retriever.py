import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.embedding_trainer import VectorBuilder


def build_tfidf(clean_texts: list[str]):
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(clean_texts)
    return vectorizer, matrix


class HybridRetriever:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        word2vec_model,
        tfidf_vectorizer,
        tfidf_matrix,
        semantic_weight=0.6,
        lexical_weight=0.4,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.vector_builder = VectorBuilder(word2vec_model)
        self.tfidf_vectorizer = tfidf_vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight

        self.doc_vectors = self.vector_builder.build_matrix(
            self.dataframe["tokens"].tolist()
        )

    def search(self, clean_query: str, query_tokens: list[str], top_k: int = 10) -> pd.DataFrame:
        query_vector = self.vector_builder.document_vector(query_tokens)
        semantic_scores = cosine_similarity([query_vector], self.doc_vectors)[0]

        query_tfidf = self.tfidf_vectorizer.transform([clean_query])
        lexical_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]

        final_scores = (
            self.semantic_weight * semantic_scores +
            self.lexical_weight * lexical_scores
        )

        results = self.dataframe.copy()
        results["semantic_score"] = semantic_scores
        results["lexical_score"] = lexical_scores
        results["final_score"] = final_scores

        results = results.sort_values("final_score", ascending=False).head(top_k)

        return results[
            [
                "doc_id",
                "title",
                "abstract",
                "keywords",
                "category",
                "semantic_score",
                "lexical_score",
                "final_score",
            ]
        ].reset_index(drop=True)