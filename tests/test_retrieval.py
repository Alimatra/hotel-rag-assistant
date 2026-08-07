import pandas as pd
import pytest

from hotel_rag.retrieval import VectorStore


def test_vector_store_requires_section_column(fake_embedder):
    pages_without_section = pd.DataFrame({"title": ["A"], "text": ["texte"]})

    with pytest.raises(ValueError, match="section"):
        VectorStore(pages_without_section, fake_embedder)


def test_search_returns_top_k_rows_with_score_column(sample_pages, fake_embedder):
    store = VectorStore(sample_pages, fake_embedder)

    results = store.search("Est-ce qu'il y a une piscine ?", top_k=2)

    assert len(results) == 2
    assert "score" in results.columns
    # Les scores doivent être triés du plus haut au plus bas
    assert results["score"].is_monotonic_decreasing


def test_search_rejects_top_k_larger_than_index(sample_pages, fake_embedder):
    store = VectorStore(sample_pages, fake_embedder)

    with pytest.raises(ValueError, match="top_k"):
        store.search("une question", top_k=100)


def test_vector_store_size_matches_number_of_pages(sample_pages, fake_embedder):
    store = VectorStore(sample_pages, fake_embedder)
    assert store.size == len(sample_pages)
