"""Global settings for the application."""

from typing import List

from pydantic import AnyHttpUrl, field_validator
from pydantic_core import Url
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "human-demo-py"
    LOG_LEVEL: str = "DEBUG"
    PORT: int = 8000

    BASE_URL: AnyHttpUrl = Url("http://localhost:8000")
    ROOT_PATH: str = "/"
    API_V1_STR: str = "/v1"
    SERVER_NAME: str = ""
    SERVER_HOST: AnyHttpUrl = Url("http://localhost:8000")
    # BACKEND_CORS_ORIGINS is a JSON-formatted list of origins
    # e.g: '["http://localhost", "http://localhost:4200", "http://localhost:3000", \
    # "http://localhost:8080", "http://local.dockertoolbox.tiangolo.com"]'
    BACKEND_CORS_ORIGINS: List[str] = []

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: str | List[str]) -> List[str] | str:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        if isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    PROJECT_NAME: str = "Human Demo"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
# print(settings)
