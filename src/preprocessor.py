import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords

from src.utils import clean_basic_text, merge_text_fields


class TextPreprocessor:
    def __init__(self):
        self._setup_nltk()
        self.stop_words = set(stopwords.words("english"))

        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError as exc:
            raise OSError(
                "spaCy model not found. Run: python -m spacy download en_core_web_sm"
            ) from exc

    def _setup_nltk(self):
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")

    def clean_text(self, text: str) -> str:
        return clean_basic_text(text)

    def tokenize(self, text: str) -> list[str]:
        cleaned = self.clean_text(text)
        doc = self.nlp(cleaned)

        tokens = []
        for token in doc:
            lemma = token.lemma_.strip().lower()

            if not lemma:
                continue
            if lemma in self.stop_words:
                continue
            if token.is_space or token.is_punct:
                continue
            if len(lemma) < 2:
                continue

            tokens.append(lemma)

        return tokens

    def load_and_process(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path, low_memory=False)

        required_cols = ["doc_id", "title", "abstract", "keywords", "category"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df["full_text"] = df.apply(
            lambda row: merge_text_fields(
                row["title"], row["abstract"], row["keywords"], row["category"]
            ),
            axis=1
        )

        df["clean_text"] = df["full_text"].apply(self.clean_text)
        df["tokens"] = df["full_text"].apply(self.tokenize)

        df = df[df["tokens"].map(len) > 0].reset_index(drop=True)
        return df