from pathlib import Path

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, TomlConfigSettingsSource

_TOML_FILENAME = "pythia.toml"


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
    model_config = SettingsConfigDict(env_file=".env")

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
        sources: list[PydanticBaseSettingsSource] = [env_settings, dotenv_settings]
        if toml_path := _find_file(_TOML_FILENAME):
            sources.append(TomlConfigSettingsSource(settings_cls, toml_file=toml_path))
        return tuple(sources)
