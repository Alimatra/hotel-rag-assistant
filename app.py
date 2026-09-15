"""Interface Streamlit pour l'assistant RAG de l'hôtel.

Appelle directement `hotel_rag.pipeline.RAGPipeline` (pas besoin de l'API
FastAPI ni d'un serveur uvicorn séparé) : la démo tourne en un seul process.
"""

from __future__ import annotations

import streamlit as st

from hotel_rag.pipeline import load_pipeline

st.set_page_config(page_title="Hotel RAG Assistant", page_icon="🏨", layout="centered")


@st.cache_resource(show_spinner="Chargement du pipeline RAG (modèles + index)...")
def get_pipeline():
    """Charge le pipeline une seule fois par session serveur (coûteux : embedder + LLM)."""
    return load_pipeline()


st.title("🏨 Hotel RAG Assistant")
st.caption(
    "Assistant virtuel documentaire : les réponses sont générées à partir de la "
    "documentation officielle de l'hôtel, jamais inventées."
)

pipeline = get_pipeline()
st.sidebar.success(f"{pipeline.num_documents} rubriques indexées")
top_k = st.sidebar.slider("Nombre de sources à consulter (top_k)", min_value=1, max_value=5, value=2)

question = st.text_input(
    "Posez votre question",
    placeholder="Ex : À quelle heure commence le check-in ?",
)

if st.button("Envoyer", type="primary") and question.strip():
    with st.spinner("Recherche dans la documentation puis génération de la réponse..."):
        result = pipeline.answer(question, top_k=top_k)

    st.markdown("### Réponse")
    st.write(result.answer)

    st.markdown("### Sources")
    for src in result.sources_summary():
        st.markdown(f"- **{src['title']}** *({src['source']})* — score={src['score']:.3f}")

elif question.strip() == "" and st.session_state.get("_asked"):
    st.info("Tapez une question puis cliquez sur Envoyer.")

st.session_state["_asked"] = True
