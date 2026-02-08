from pathlib import Path
from functools import lru_cache

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from pydantic import SecretStr, Field
from typing import List



class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API Keys
    openai_api_key: SecretStr = SecretStr("")
    anthropic_api_key: SecretStr = SecretStr("")
    google_api_key: SecretStr = SecretStr("")
    perplexity_api_key: SecretStr = SecretStr("")

    # Database
    database_url: str = "./data/perplexity.db"

    # ChromaDB
    chroma_persist_dir: str = "./data/vectorstore"

    # Server
    backend_port: int = 8000
    log_level: str = "INFO"

    # Safety
    max_daily_spend_usd: float = 10.0
    access_code: str = ""
    admin_password: str = ""

    # Deployment
    allowed_origins: str = ""

    # Loaded from YAML
    yaml_config: dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        yaml_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                self.yaml_config = yaml.safe_load(f) or {}

    @property
    def models_config(self) -> list[dict]:
        return self.yaml_config.get("models", {}).get("available", [])

    @property
    def default_model(self) -> str:
        return self.yaml_config.get("models", {}).get("default", "gpt-4o")

    @property
    def rag_config(self) -> dict:
        return self.yaml_config.get("rag", {})

    @property
    def voice_config(self) -> dict:
        return self.yaml_config.get("voice", {})

    @property
    def pricing_config(self) -> dict:
        return self.yaml_config.get("pricing", {})

    @property
    def embedding_config(self) -> dict:
        return self.yaml_config.get("embedding", {})


@lru_cache
def get_settings() -> Settings:
    return Settings()
