import joblib
import numpy as np
import pandas as pd

from config.config import (
    DATA_FILE,
    PROCESSED_DATA_PATH,
    W2V_MODEL_PATH,
    TFIDF_VECTORIZER_PATH,
    #DOC_VECTORS_PATH,
    TOP_K,
    SEMANTIC_WEIGHT,
    LEXICAL_WEIGHT,
    VECTOR_SIZE,
    WINDOW,
    MIN_COUNT,
    EPOCHS,
)
from src.preprocessor import TextPreprocessor
from src.ontology_engine import OntologyExpander
from src.embedding_trainer import EmbeddingTrainer, VectorBuilder
from src.retriever import HybridRetriever, build_tfidf


def train_pipeline():
    print("Loading and preprocessing dataset...")
    preprocessor = TextPreprocessor()
    df = preprocessor.load_and_process(DATA_FILE)

    print("Applying ontology expansion to documents...")
    ontology = OntologyExpander()
    df["expanded_text"] = df["clean_text"].apply(ontology.expand_query_text)

    print("Training Word2Vec model...")
    trainer = EmbeddingTrainer(
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        epochs=EPOCHS
    )
    w2v_model = trainer.train(df["tokens"].tolist())
    w2v_model.save(str(W2V_MODEL_PATH))

    print("Building TF-IDF matrix...")
    tfidf_vectorizer, tfidf_matrix = build_tfidf(df["expanded_text"].tolist())
    joblib.dump(tfidf_vectorizer, TFIDF_VECTORIZER_PATH)

    print("Saving processed dataset...")
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    vector_builder = VectorBuilder(w2v_model)
    doc_vectors = vector_builder.build_matrix(df["tokens"].tolist())
    np.save(DOC_VECTORS_PATH, doc_vectors)

    print("Training complete.")
    print(f"Processed data saved to: {PROCESSED_DATA_PATH}")
    print(f"Word2Vec model saved to: {W2V_MODEL_PATH}")
    print(f"TF-IDF vectorizer saved to: {TFIDF_VECTORIZER_PATH}")


if __name__ == "__main__":
    train_pipeline()