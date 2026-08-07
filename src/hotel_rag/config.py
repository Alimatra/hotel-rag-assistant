"""Configuration centralisée du projet.

Toutes les constantes (modèles, chemins, hyperparamètres) vivent ici,
pour éviter les valeurs codées en dur éparpillées dans le code.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Paramètres de l'application, surchargeables via variables d'environnement.

    Exemple : `EMBEDDING_MODEL=... GENERATION_MODEL=... uvicorn hotel_rag.api:app`
    """

    # Chemins
    data_dir: Path = Path("data")

    # Modèles Hugging Face
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    generation_model: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # Paramètres de génération
    max_new_tokens: int = 80
    do_sample: bool = False

    # Paramètres de recherche
    default_top_k: int = 2

    # Prompt système
    hotel_name: str = "Hôtel Le Belvédère"
    hotel_location: str = "au bord du lac d'Annecy"
    role_prompt: str = (
        "Tu es l'assistant virtuel de l'{name}, {location}.\n"
        "Voici la documentation officielle de l'hôtel :"
    )
    consigne_prompt: str = (
        "Réponds en une ou deux phrases, uniquement à partir des informations "
        "de la documentation ci-dessus. Si l'information ne s'y trouve pas, réponds exactement : "
        '"Je ne sais pas, je vous invite à contacter la réception."'
    )

    model_config = SettingsConfigDict(env_prefix="HOTEL_RAG_")

    @property
    def role(self) -> str:
        return self.role_prompt.format(name=self.hotel_name, location=self.hotel_location)


settings = Settings()
