import ast
import joblib
import pandas as pd
import streamlit as st
from gensim.models import Word2Vec

from config.config import (
    PROCESSED_DATA_PATH,
    W2V_MODEL_PATH,
    TFIDF_VECTORIZER_PATH,
    TOP_K,
    SEMANTIC_WEIGHT,
    LEXICAL_WEIGHT,
)
from src.preprocessor import TextPreprocessor
from src.ontology_engine import OntologyExpander
from src.retriever import HybridRetriever


@st.cache_resource
def load_components():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["tokens"] = df["tokens"].apply(ast.literal_eval)

    w2v_model = Word2Vec.load(str(W2V_MODEL_PATH))
    tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    tfidf_matrix = tfidf_vectorizer.transform(df["expanded_text"].tolist())

    retriever = HybridRetriever(
        dataframe=df,
        word2vec_model=w2v_model,
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix,
        semantic_weight=SEMANTIC_WEIGHT,
        lexical_weight=LEXICAL_WEIGHT,
    )

    return retriever


st.set_page_config(page_title="ArXiv Semantic Search", layout="wide")
st.title("Semantic-Based Information Retrieval System")
st.caption("ArXiv Computer Science Abstract Search using Word2Vec + Ontology Expansion + TF-IDF")

preprocessor = TextPreprocessor()
ontology = OntologyExpander()
retriever = load_components()

query = st.text_input(
    "Enter a Computer Science topic",
    placeholder="e.g. intrusion detection, deep learning, semantic search"
)

top_k = st.slider("Top K Results", min_value=3, max_value=20, value=TOP_K)

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        expanded_query = ontology.expand_query_text(query)
        clean_query = preprocessor.clean_text(expanded_query)
        query_tokens = preprocessor.tokenize(expanded_query)

        results = retriever.search(clean_query, query_tokens, top_k=top_k)

        st.subheader("Expanded Query")
        st.write(expanded_query)

        st.subheader("Results")
        for i, row in results.iterrows():
            with st.container(border=True):
                st.markdown(f"### {i + 1}. {row['title']}")
                st.write(row["abstract"])
                st.caption(f"Category: {row['category']}")
                st.caption(f"Keywords: {row['keywords']}")
                st.write(
                    f"Semantic Score: {row['semantic_score']:.4f} | "
                    f"Lexical Score: {row['lexical_score']:.4f} | "
                    f"Final Score: {row['final_score']:.4f}"
                )