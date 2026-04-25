from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

_TOML_FILENAME = "pythia.toml"
_DOTENV_FILENAME = ".env"


def _find_file(name: str) -> Path | None:
    for directory in [Path.cwd(), *Path.cwd().parents]:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


class EmbeddingSettings(BaseModel):
    model: str = "jinaai/jina-embeddings-v5-text-small"
    dim: int = 1024


class QdrantSettings(BaseModel):
    path: str = "data/qdrant_store"
    collection: str = "datasheets"


class RetrieveSettings(BaseModel):
    top_k: int = 5


class AssistantSettings(BaseModel):
    ollama_model: str = "qwen3.5:4b"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PYTHIA_",
        env_nested_delimiter="__",
    )

    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    embedding: EmbeddingSettings = EmbeddingSettings()
    qdrant: QdrantSettings = QdrantSettings()
    retrieve: RetrieveSettings = RetrieveSettings()
    assistant: AssistantSettings = AssistantSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [env_settings]
        if dotenv_path := _find_file(_DOTENV_FILENAME):
            sources.append(DotEnvSettingsSource(settings_cls, env_file=dotenv_path))
        if toml_path := _find_file(_TOML_FILENAME):
            sources.append(TomlConfigSettingsSource(settings_cls, toml_file=toml_path))
        return tuple(sources)
