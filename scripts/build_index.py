"""CLI simple pour interroger l'assistant sans lancer l'API.

Usage :
    python scripts/build_index.py "A quelle heure commence le check-in ?"
"""

from __future__ import annotations

import argparse
import logging
import sys

from hotel_rag.pipeline import load_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interroge l'assistant RAG de l'hôtel en ligne de commande.")
    parser.add_argument("question", help="La question à poser à l'assistant")
    parser.add_argument("--top-k", type=int, default=2, help="Nombre de rubriques à récupérer (défaut : 2)")
    args = parser.parse_args()

    pipeline = load_pipeline()
    result = pipeline.answer(args.question, top_k=args.top_k)

    print(f"\nQuestion : {result.question}")
    print(f"Réponse  : {result.answer}\n")
    print("Sources :")
    for row in result.sources_summary():
        print(f"  - {row['title']} ({row['source']}), score={row['score']:.3f}")


if __name__ == "__main__":
    sys.exit(main())
