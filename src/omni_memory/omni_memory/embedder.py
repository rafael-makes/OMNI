"""Embedding interface + Gemini implementation.

Kept behind an `Embedder` ABC so the model can be swapped later (SPEC decision).
Pure Python, no ROS. google-genai is imported lazily so this module imports even
where the SDK / API key aren't present (e.g. offline unit runs).
"""
from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from typing import Optional, Sequence

# Verified live 2026-07-12: gemini-embedding-001 with output_dimensionality=768
# returns exactly 768 dims and is NOT L2-normalized (norm ~0.59), so we normalize
# here. The 3072 default is normalized; keep that in mind if the dim is ever raised.
DEFAULT_MODEL = "gemini-embedding-001"
DEFAULT_DIM = 768


class Embedder(ABC):
    """Turns text into a fixed-length vector. `dim` is the output dimension."""

    dim: int

    @abstractmethod
    def embed_document(self, text: str) -> list[float]:
        """Embed text that will be stored/retrieved against."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_document(t) for t in texts]


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


class GeminiEmbedder(Embedder):
    """Embedder backed by the Gemini embedding API.

    Uses distinct task types for documents vs queries (asymmetric retrieval),
    which the Gemini embedding models are trained for.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        dim: int = DEFAULT_DIM,
        normalize: bool = True,
        client=None,
    ) -> None:
        self.model = model
        self.dim = dim
        self._normalize = normalize
        if client is not None:
            self._client = client
            return
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set (export it or add it to .env)."
            )
        from google import genai  # lazy

        self._client = genai.Client(api_key=api_key)

    def _embed(self, text: str, task_type: str) -> list[float]:
        from google.genai import types  # lazy

        resp = self._client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.dim,
            ),
        )
        vec = list(resp.embeddings[0].values)
        if len(vec) != self.dim:
            raise RuntimeError(
                f"embedding dim mismatch: expected {self.dim}, got {len(vec)}"
            )
        return _l2_normalize(vec) if self._normalize else vec

    def embed_document(self, text: str) -> list[float]:
        return self._embed(text, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, "RETRIEVAL_QUERY")
