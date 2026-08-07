"""Extraction et structuration de la documentation source (PDF -> chunks).

Chaque page d'un PDF correspond à une rubrique de la documentation.
On extrait, pour chacune : le fichier source, le titre (première ligne)
et le texte (tout sauf le titre et le pied de page).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Chunk:
    """Une rubrique de documentation, prête à être indexée."""

    source: str
    title: str
    text: str

    @property
    def section(self) -> str:
        """Représentation Markdown de la rubrique : titre en h2 + texte."""
        return f"## {self.title}\n\n{self.text}"


def extract_page_text(page) -> list[str]:
    """Extrait le texte brut d'une page PDF et le découpe en lignes.

    Lève une erreur explicite si la page ne contient aucun texte
    exploitable (PDF scanné/image, par exemple), plutôt que de
    propager un IndexError obscur plus loin dans le pipeline.
    """
    lines = page.extract_text().split("\n")
    if len(lines) < 2:
        raise ValueError(
            "Page sans contenu exploitable (moins de 2 lignes) : "
            "vérifie qu'il ne s'agit pas d'un PDF scanné sans OCR."
        )
    return lines


def load_pdf(path: Path) -> list[Chunk]:
    """Charge un PDF et retourne une rubrique par page."""
    reader = PdfReader(path)
    chunks = []
    for page in reader.pages:
        lines = extract_page_text(page)
        chunks.append(
            Chunk(
                source=path.name,
                title=lines[0],
                text="\n".join(lines[1:-1]),
            )
        )
    logger.info("PDF chargé : %s (%d rubriques)", path.name, len(chunks))
    return chunks


def load_documentation(data_dir: Path) -> pd.DataFrame:
    """Charge tous les PDF d'un dossier en un DataFrame de rubriques.

    Colonnes : source, title, text, section.
    """
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"Aucun PDF trouvé dans {data_dir}")

    chunks: list[Chunk] = []
    for path in pdf_paths:
        chunks.extend(load_pdf(path))

    df = pd.DataFrame(
        {
            "source": [c.source for c in chunks],
            "title": [c.title for c in chunks],
            "text": [c.text for c in chunks],
            "section": [c.section for c in chunks],
        }
    )
    logger.info("Documentation chargée : %d rubriques depuis %d PDF", len(df), len(pdf_paths))
    return df


def build_full_context(pages: pd.DataFrame) -> str:
    """Assemble toutes les rubriques en un seul contexte Markdown.

    Utile pour comparer l'approche "tout dans le prompt" au RAG,
    mais ne devrait pas servir en production (cf. README).
    """
    return "\n\n".join(pages["section"])
