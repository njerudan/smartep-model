"""Text features.

Encoders:
- ``none``: no text features.
- ``tfidf_svd``: TF-IDF + truncated SVD fitted on training texts. No download, fast on CPU. Default.
- ``minilm``: all-MiniLM-L6-v2 sentence embeddings reduced with PCA fitted on training texts. Needs the
  ``text`` extra. Replaces the notebook's DistilBERT (768 dims on 2,000 rows over-fits; slow on CPU).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def aggregate_texts(df: pd.DataFrame, educator_ids, text_col: str, empty: str = "") -> list[str]:
    joined = df.groupby("educator_id")[text_col].agg(" ".join)
    return joined.reindex(educator_ids).fillna(empty).tolist()


class TextEncoder:
    def __init__(self, encoder: str = "tfidf_svd", dim: int = 32, seed: int = 0):
        if encoder not in {"none", "tfidf_svd", "minilm"}:
            raise ValueError(f"unknown text encoder {encoder!r}")
        self.encoder, self.dim, self.seed = encoder, dim, seed
        self._vectorizer = self._reducer = None
        self.output_dim = 0

    @staticmethod
    def _embed_minilm(texts: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MINILM, device="cpu")
        return model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)

    def fit(self, texts: list[str]) -> TextEncoder:
        if self.encoder == "none":
            return self
        if self.encoder == "tfidf_svd":
            self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
            matrix = self._vectorizer.fit_transform(texts)
            n = max(1, min(self.dim, matrix.shape[1] - 1))
            self._reducer = TruncatedSVD(n_components=n, random_state=self.seed).fit(matrix)
        else:
            emb = self._embed_minilm(texts)
            self._reducer = PCA(n_components=min(self.dim, emb.shape[0], emb.shape[1]), random_state=self.seed).fit(emb)
        self.output_dim = self._reducer.n_components
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.encoder == "none":
            return np.zeros((len(texts), 0), dtype=np.float32)
        if self.encoder == "tfidf_svd":
            return self._reducer.transform(self._vectorizer.transform(texts)).astype(np.float32)
        return self._reducer.transform(self._embed_minilm(texts)).astype(np.float32)

    def feature_names(self, prefix: str = "text") -> list[str]:
        return [f"{prefix}_{i}" for i in range(self.output_dim)]
