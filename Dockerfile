# Image légère : les modèles (torch + transformers) sont volumineux,
# on part donc d'une base slim et on installe uniquement le nécessaire.
FROM python:3.11-slim

WORKDIR /app

# Dépendances système minimales requises par torch/sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# Installation du package en mode non-éditable (image de prod)
RUN pip install --no-cache-dir .

COPY data ./data
COPY scripts ./scripts

# Les modèles Hugging Face sont téléchargés au premier démarrage et mis
# en cache dans ce volume, pour ne pas re-télécharger à chaque redéploiement.
ENV HF_HOME=/app/.cache/huggingface
VOLUME ["/app/.cache/huggingface"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "hotel_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
