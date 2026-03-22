import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # adds project root

from config.config import RAW_ARXIV_FILE, DATA_FILE, MAX_RECORDS

def prepare_arxiv_dataset():
    required_cols = ["id", "title", "abstract", "categories"]
    first_chunk = True
    total_records = 0

    for chunk in pd.read_json(RAW_ARXIV_FILE, lines=True, chunksize=1000):
        # Validate columns on first chunk
        if first_chunk:
            for col in required_cols:
                if col not in chunk.columns:
                    raise ValueError(f"Missing required column: {col}")

        df = chunk[required_cols].copy()

        # Keep only CS records
        df = df[df["categories"].astype(str).str.contains(r'\bcs\.', regex=True, na=False)]

        # Remove missing values
        df = df.dropna(subset=["title", "abstract", "categories"])

        if df.empty:
            continue

        # Rename and restructure
        df = df.rename(columns={"id": "doc_id"})
        df["keywords"] = df["categories"]
        df["category"] = df["categories"]
        df = df[["doc_id", "title", "abstract", "keywords", "category"]]
        df = df.drop_duplicates(subset=["doc_id"]).reset_index(drop=True)

        # Write chunk to disk
        df.to_csv(DATA_FILE, mode='w' if first_chunk else 'a', header=first_chunk, index=False)
        first_chunk = False
        total_records += len(df)

        print(f"Records so far: {total_records}", end="\r")

        # Stop early if MAX_RECORDS reached
        if MAX_RECORDS > 0 and total_records >= MAX_RECORDS:
            break

    print(f"\nPrepared ArXiv dataset saved successfully.")
    print(f"Output file: {DATA_FILE}")
    print(f"Total records: {total_records}")


if __name__ == "__main__":
    prepare_arxiv_dataset()