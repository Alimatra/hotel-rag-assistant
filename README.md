# Hotel RAG Assistant

Assistant virtuel documentaire pour un hôtel, construit sur un pipeline **RAG (Retrieval-Augmented Generation)** : les réponses sont générées à partir de la documentation officielle de l'établissement, jamais inventées.

## Le problème

Un modèle de langage généraliste interrogé sur un hôtel spécifique invente des informations plausibles mais fausses (adresse, étoiles, tarifs...) — un risque réputationnel direct pour l'établissement. Deux approches ont été comparées avant de converger vers un RAG :

| Approche | Fiabilité | Latence | Passage à l'échelle |
|---|---|---|---|
| LLM seul, sans contexte | ❌ Hallucine | Rapide | — |
| Toute la documentation dans le prompt | ⚠️ Le modèle se perd dans un contexte trop long | Lente (relit tout à chaque question) | ❌ Ne tient pas au-delà de quelques dizaines de documents |
| **RAG (ce projet)** | ✅ Réponses ancrées dans la documentation, sources citées | Rapide (contexte réduit) | ✅ Le temps de réponse ne dépend pas du volume de documentation |

## Architecture

```
Question client
      │
      ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Embedder   │────▶│ VectorStore  │────▶│  Generator  │──▶ Réponse + sources
│ (question)  │     │  (top-k)     │     │   (LLM)     │
└─────────────┘     └──────────────┘     └─────────────┘
                            ▲
                     ┌──────────────┐
                     │  Ingestion   │
                     │ (PDF → chunks)│
                     └──────────────┘
```

```
src/hotel_rag/
├── config.py       # Paramètres centralisés (modèles, prompts, chemins)
├── ingestion.py     # Extraction PDF → DataFrame de rubriques (chunks)
├── embeddings.py    # Encodage texte → vecteurs (sentence-transformers)
├── retrieval.py     # Index vectoriel en mémoire + recherche par similarité cosinus
├── generation.py    # Appel au LLM + construction du prompt
├── pipeline.py       # Orchestration : ingestion + retrieval + generation
└── api.py           # API REST (FastAPI)
```

Chaque étage a une responsabilité unique et une interface stable. Concrètement, ça veut dire :
- remplacer `numpy` par une vraie base vectorielle (FAISS, Chroma, Pinecone) = ne toucher qu'à `retrieval.py`
- changer de modèle de génération (un modèle plus gros, une API payante) = ne toucher qu'à `generation.py`
- passer d'un PDF à une base SQL ou une API = ne toucher qu'à `ingestion.py`

## Stack technique

- **Extraction** : `pypdf`
- **Embeddings** : `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`, multilingue, 384 dimensions)
- **Recherche vectorielle** : `numpy` (similarité cosinus par produit matriciel — voir *Limites et évolutions* ci-dessous)
- **Génération** : `transformers` (`Qwen/Qwen2.5-0.5B-Instruct`)
- **API** : `FastAPI` + `uvicorn`
- **Tests** : `pytest`, avec un embedder factice pour tester la logique de recherche sans dépendance réseau
- **CI** : GitHub Actions (lint + tests à chaque push)
- **Containerisation** : Docker

## Installation

```bash
git clone <url-du-repo>
cd hotel-rag-assistant
pip install -e ".[dev]"
```

## Utilisation

**En ligne de commande :**
```bash
python scripts/build_index.py "A quelle heure commence le check-in ?"
```

**Via l'API :**
```bash
uvicorn hotel_rag.api:app --reload
```
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Le wifi est-il gratuit ?", "top_k": 2}'
```

**Avec Docker :**
```bash
docker build -t hotel-rag-assistant .
docker run -p 8000:8000 hotel-rag-assistant
```

## Tests

```bash
pytest --cov=hotel_rag --cov-report=term-missing
```

Les tests d'ingestion et de recherche n'ont pas de dépendance réseau : un `FakeEmbedder` déterministe (voir `tests/conftest.py`) simule l'interface de l'embedder réel, ce qui permet à la CI de tourner en quelques secondes sans télécharger de modèle.

## Limites et évolutions possibles

Ce projet assume des choix volontairement simples, adaptés à une documentation de quelques dizaines de pages :

- **Recherche vectorielle en mémoire (`numpy`)** plutôt qu'une vraie base vectorielle : suffisant ici, mais à remplacer par FAISS ou Chroma au-delà de quelques milliers de chunks.
- **Chunking par page** : chaque page de PDF devient un chunk. Un vrai projet découperait des documents plus longs avec une stratégie de chunking dédiée (taille fixe, découpage sémantique...).
- **Modèle de génération volontairement petit** (0,5 milliard de paramètres) pour tourner sans GPU : un modèle plus gros ou une API (Claude, GPT) améliorerait la qualité rédactionnelle, sans changer l'architecture.
- **Pas d'évaluation automatisée du pipeline** (jeu de questions/réponses de référence, métriques de retrieval) : à ajouter avant tout usage en production réelle.

