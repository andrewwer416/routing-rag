from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    data_dir: Path = Path("data")
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    llm_base_url: str = ""
    llm_model: str = "llama3.2:3b"
    llm_api_key: str = "local"
    answer_max_tokens: int = Field(default=512, ge=1, le=8192)
    routing_provider: Literal["chat", "kev"] = "chat"
    routing_model: str = ""
    kev_base_url: str = "http://kev:8009"
    kev_api_key: str = "local"
    kev_model: str = "kev-latest"
    routing_timeout: float = 60
    api_key: str = ""

    @property
    def routing_identity(self) -> str:
        return self.routing_model or (
            "jaredpalmer/kev-4b@v1.0" if self.routing_provider == "kev" else self.llm_model
        )
