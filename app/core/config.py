from typing import List, Optional, Any, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator, field_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application
    APP_NAME: str = "Investment Assistant"
    APP_ENV: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    DEBUG: bool = False

    # Database Backend ("json" or "firestore")
    DB_BACKEND: str = "json"
    JSON_DATA_PATH: str = "data/assets.json"
    GCP_PROJECT_ID: Optional[str] = None
    FIRESTORE_DATABASE: str = "(default)"
    FIRESTORE_COLLECTION_CONFIG: str = "system_config"
    FIRESTORE_COLLECTION_PORTFOLIO: str = "portfolio"

    # LINE Bot
    LINE_CHANNEL_SECRET: str = ""
    LINE_CHANNEL_ACCESS_TOKEN: str = ""
    LINE_USER_ID: Optional[str] = None  # Target user/group ID for scheduled push notifications

    # Gemini AI
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"

    # SLM (Small Language Model - OpenAI Compatible Endpoint, e.g. Colab / vLLM / Ollama)
    SLM_ENABLED: bool = False
    SLM_ENDPOINT_URL: str = "http://localhost:8000/v1"
    SLM_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct"
    SLM_API_KEY: str = "EMPTY"
    SLM_TIMEOUT_SECONDS: float = 5.0
    SLM_FALLBACK_TO_REGEX: bool = True

    # Network & Retry settings
    HTTP_TIMEOUT_SECONDS: float = 15.0
    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_INITIAL_DELAY: float = 1.0
    RETRY_BACKOFF_FACTOR: float = 2.0

    # Security & Administration
    ADMIN_API_KEY: str = ""
    CORS_ALLOWED_ORIGINS: Union[List[str], str] = Field(default_factory=list)
    ALLOWED_LINE_USER_IDS: Union[List[str], str] = Field(default_factory=list)

    # Stock Watchlist & FX for Buy Strategy Scanning
    BUY_CANDIDATES: Union[List[str], str] = Field(
        default_factory=lambda: ["2330", "6442", "2313", "2454", "2382", "SPCX"]
    )
    USD_TWD_RATE: float = 32.0

    # Scheduler settings
    ENABLE_SCHEDULER: bool = True
    SELL_CHECK_INTERVAL_MINUTES: int = 30
    BUY_CHECK_CRON_HOUR: int = 13
    BUY_CHECK_CRON_MINUTE: int = 0
    MARKET_REPORT_CRON_HOUR: int = 8
    MARKET_REPORT_CRON_MINUTE: int = 0
    MARKET_REPORT_CRON_DAYS: str = "mon-sun"

    @field_validator("ALLOWED_LINE_USER_IDS", "CORS_ALLOWED_ORIGINS", "BUY_CANDIDATES", mode="after")
    @classmethod
    def parse_str_or_list(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            if v.startswith("[") and v.endswith("]"):
                import json
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [x.strip() for x in v.split(",") if x.strip()]
        return list(v) if v else []

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        if self.APP_ENV.lower() == "production":
            missing = []
            if not self.LINE_CHANNEL_SECRET:
                missing.append("LINE_CHANNEL_SECRET")
            if not self.LINE_CHANNEL_ACCESS_TOKEN:
                missing.append("LINE_CHANNEL_ACCESS_TOKEN")
            if not self.GEMINI_API_KEY:
                missing.append("GEMINI_API_KEY")
            if not self.ADMIN_API_KEY:
                missing.append("ADMIN_API_KEY")
            if missing:
                raise ValueError(
                    f"Production startup failed: Required environment variable(s) missing: {', '.join(missing)}"
                )
        return self


settings = Settings()
