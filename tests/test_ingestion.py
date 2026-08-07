from pathlib import Path

import pytest

from hotel_rag.ingestion import Chunk, build_full_context, load_documentation

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_chunk_section_formats_as_markdown_h2():
    chunk = Chunk(source="a.pdf", title="Piscine", text="Ouverte de 8h à 20h.")
    assert chunk.section == "## Piscine\n\nOuverte de 8h à 20h."


def test_load_documentation_finds_all_pdfs_and_expected_columns():
    pages = load_documentation(DATA_DIR)

    assert len(pages) == 15, "5 PDF, 3 pages chacun d'après la doc d'origine"
    assert list(pages.columns) == ["source", "title", "text", "section"]
    assert pages["source"].nunique() == 5


def test_load_documentation_raises_on_missing_directory(tmp_path):
    empty_dir = tmp_path / "no_pdfs_here"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        load_documentation(empty_dir)


def test_build_full_context_contains_every_section_title():
    pages = load_documentation(DATA_DIR)
    context = build_full_context(pages)

    for title in pages["title"]:
        assert f"## {title}" in context
