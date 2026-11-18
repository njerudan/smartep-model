"""Text features from evidence packages.

Three text groups per package: ``comments`` (student comments), ``notes`` (notes, slides and document
text) and ``transcripts`` (video transcript segments). Each group is turned into a few numbers:

- ``tfidf_svd``: the group's units joined into one document, TF-IDF, truncated SVD. No download.
- ``minilm``: each unit embedded with all-MiniLM-L6-v2 (unique texts embedded once), averaged per
  package, then PCA. Needs the ``text`` extra; the model is downloaded once and cached by Hugging Face.

Every learned step (vocabulary, SVD, PCA) is fitted on the training packages only.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
GROUPS = ("comments", "notes", "transcripts")
_MODEL = None
_CACHE: dict[str, np.ndarray] = {}


def text_units(pkg: dict) -> dict[str, list[str]]:
    units = {g: [] for g in GROUPS}
    for r in pkg["evaluations"]:
        for q in r.get("questions") or []:
            if q["kind"] == "text":
                units["comments"].extend(q.get("comments") or [])
    for les in pkg["lessons"]:
        if les.get("notes"):
            units["notes"].append(les["notes"]["text"])
        units["notes"].extend(s["text"] for s in les.get("slides") or [])
        units["notes"].extend(d["extractedText"] for d in les["documents"] if d.get("extractedText"))
        for v in les["videos"]:
            units["transcripts"].extend(s["text"] for s in (v.get("transcript") or {}).get("segments", []))
    return units


def _minilm():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(MINILM, device="cpu")
    return _MODEL


def embed_units(texts: list[str]) -> np.ndarray:
    """MiniLM embeddings with a process-wide cache keyed by text."""
    new = sorted({t for t in texts if t not in _CACHE})
    if new:
        for t, e in zip(new, _minilm().encode(new, batch_size=64, show_progress_bar=False,
                                               convert_to_numpy=True, normalize_embeddings=True), strict=True):
            _CACHE[t] = e
    return np.stack([_CACHE[t] for t in texts]) if texts else np.zeros((0, 384), dtype=np.float32)


def load_embedding_cache(path) -> int:
    """Load MiniLM embeddings saved by ``save_embedding_cache``; returns the number of texts."""
    import pickle
    from pathlib import Path

    path = Path(path)
    if path.exists():
        with open(path, "rb") as f:
            _CACHE.update(pickle.load(f))
    return len(_CACHE)


def save_embedding_cache(path) -> None:
    import pickle
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({t: e.astype(np.float32) for t, e in _CACHE.items()}, f, protocol=pickle.HIGHEST_PROTOCOL)


class PackageTextFeaturizer:
    def __init__(self, encoder: str = "tfidf_svd", dim: int = 16, seed: int = 0):
        if encoder not in {"none", "tfidf_svd", "minilm"}:
            raise ValueError(f"unknown text encoder {encoder!r}")
        self.encoder, self.dim, self.seed = encoder, dim, seed
        self.models: dict[str, tuple] = {}
        self.feature_names: list[str] = []

    def _group_matrix(self, units: list[list[str]]):
        if self.encoder == "minilm":
            return np.stack([embed_units(u).mean(axis=0) if u else np.zeros(384) for u in units])
        return [" ".join(u) for u in units]

    def fit(self, packages: list[dict]) -> PackageTextFeaturizer:
        self.models, self.feature_names = {}, []
        if self.encoder == "none":
            return self
        all_units = [text_units(p) for p in packages]
        for g in GROUPS:
            data = self._group_matrix([u[g] for u in all_units])
            if self.encoder == "tfidf_svd":
                vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)
                matrix = vec.fit_transform(data)
                if matrix.shape[1] < 2:
                    continue
                red = TruncatedSVD(n_components=min(self.dim, matrix.shape[1] - 1), random_state=self.seed).fit(matrix)
                self.models[g] = (vec, red)
            else:
                red = PCA(n_components=min(self.dim, data.shape[0], data.shape[1]), random_state=self.seed).fit(data)
                self.models[g] = (None, red)
            self.feature_names += [f"text_{g}_{i}" for i in range(red.n_components)]
        return self

    def transform(self, packages: list[dict]) -> np.ndarray:
        if not self.models:
            return np.zeros((len(packages), 0), dtype=np.float32)
        all_units = [text_units(p) for p in packages]
        parts = []
        for g in GROUPS:
            if g not in self.models:
                continue
            vec, red = self.models[g]
            data = self._group_matrix([u[g] for u in all_units])
            parts.append(red.transform(vec.transform(data) if vec is not None else data))
        return np.hstack(parts).astype(np.float32)
