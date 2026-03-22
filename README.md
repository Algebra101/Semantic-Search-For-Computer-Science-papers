# Semantic-Based Information Retrieval System for ArXiv Computer Science Abstracts

## Overview

This project is an undergraduate semantic information retrieval system built on the ArXiv Computer Science dataset.

The system improves traditional keyword search by combining:

- Word2Vec semantic similarity
- ontology-style Computer Science topic expansion
- TF-IDF lexical retrieval
- cosine similarity ranking
- Streamlit web interface

## Dataset

This project uses an ArXiv Computer Science dataset containing:

- id
- title
- abstract
- categories

The dataset is converted into the internal format:

- doc_id
- title
- abstract
- keywords
- category

## Project Structure

```text
semantic_ir_arxiv/
├── app/
│   └── streamlit_app.py
├── config/
│   └── config.py
├── data/
│   ├── raw/
│   │   └── arxiv_raw.csv
│   └── processed/
├── models/
├── src/
│   ├── __init__.py
│   ├── prepare_arxiv_dataset.py
│   ├── preprocessor.py
│   ├── ontology_engine.py
│   ├── embedding_trainer.py
│   ├── retriever.py
│   ├── evaluator.py
│   └── utils.py
├── .env
├── README.md
├── requirements.txt
└── main.py
```
