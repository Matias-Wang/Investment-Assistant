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

    # Gemini 輸出成本控制 (output token 單價為 input 的 8.3 倍，是主要成本來源)
    # 上限需留足夠餘裕：response_schema 為 JSON，截斷會導致反序列化失敗而整筆作廢
    BUY_ANALYSIS_MAX_OUTPUT_TOKENS: int = 800
    MARKET_REPORT_MAX_OUTPUT_TOKENS: int = 1024
    # 選用的 thinking 上限 (thinking token 併入 output 計價)。
    # 空字串 = 不傳送此參數，沿用模型預設 (gemini-3.5-flash-lite 預設為 Off)。
    # 若改用 thinking 預設開啟的模型 (如 3.x Flash)，設為 "MINIMAL" 可避免費用暴增。
    GEMINI_THINKING_LEVEL: str = ""

    # 買入掃描效能與成本
    # 並行度：掃描為 I/O bound (yfinance + 新聞 + Gemini)，並行可大幅縮短牆鐘時間；
    # 上限刻意保守以避免觸發 Gemini / yfinance 的 rate limit (429)。
    BUY_SCAN_CONCURRENCY: int = 3
    # 新聞快取存活秒數：同一標的同一日重複掃描時不重抓 (0 = 停用快取)
    NEWS_CACHE_TTL_SECONDS: int = 3600
    # 新聞關鍵字預篩模式：
    #   off     = 完全不執行
    #   shadow  = 執行並記錄判定，但不淘汰任何標的 (預設，用於量測 recall/precision)
    #   enforce = 實際淘汰未通過預篩的標的，不送 Gemini
    NEWS_FILTER_MODE: str = "shadow"

    # --- SLM 幻覺驗證層 (NLI entailment) ---------------------------------
    # 專案宣稱「嚴格禁止臆測」，但該限制原本只存在於提示詞中、無任何驗證。
    # 此層以小型 NLI 模型逐條檢查 Gemini 的 reasons 是否真被新聞文本支持。
    #   off      = 不執行
    #   shadow   = 執行並記錄，不影響結果 (預設，用於量測攔截率與誤殺率)
    #   annotate = 於推播卡片標示各理由的驗證狀態（預設，2026-09-21 量測後改採）
    #   enforce  = 接地比例未達門檻者不推播買入建議
    # 為何預設 annotate 而非 enforce：t=0.25 時誤殺率 0.111，約每 9 筆合法建議會被
    # 擋掉 1 筆。annotate 的誤判只是標示不準，使用者仍看得到原始理由可自行判斷。
    GROUNDING_MODE: str = "annotate"
    # ONNX 模型與 tokenizer 路徑 (由 slm/colab_nli_export.ipynb 產出並隨映像打包)
    GROUNDING_MODEL_PATH: str = "models/nli/model_quantized.onnx"
    GROUNDING_TOKENIZER_PATH: str = "models/nli/tokenizer.json"
    # 模型輸出的標籤順序，必須與所選模型的 config.id2label 一致，
    # 由匯出腳本印出後填入；填錯會讓 entailment 與 contradiction 對調。
    GROUNDING_LABEL_ORDER: str = "entailment,neutral,contradiction"
    # 單一理由被視為「有被新聞支持」所需的 entailment 機率。
    # 2026-09-21 以 36 真 + 36 假理由（真實台股新聞，Gemini 生成）實測 mDeBERTa int8：
    #   AUC = 0.910（強鑑別力）；真理由平均 0.421、假理由平均 0.219
    #   門檻掃描  攔截率 / 誤殺率：
    #     0.22 -> 0.556 / 0.083    0.25 -> 0.750 / 0.111
    #     0.30 -> 0.944 / 0.250    0.50 -> 1.000 / 0.722
    # 此模型在中文財經語域的 entailment 分數整體偏低，0.5 會落在真理由分佈的上緣
    # （P75 = 0.520），導致大量誤殺。0.25 是 annotate 模式下誤殺與攔截的較佳平衡。
    GROUNDING_ENTAILMENT_THRESHOLD: float = 0.25
    # enforce 模式下，通過的理由需佔全部理由的比例
    GROUNDING_MIN_GROUNDED_RATIO: float = 0.5
    GROUNDING_MAX_SEQ_LENGTH: int = 512
    # onnxruntime 的 intra-op 執行緒數。本機 8 核實測（mDeBERTa int8，單次句對推論）：
    #   1 執行緒 121.6ms / 2 執行緒 89.8ms / 4 執行緒 62.0ms / 預設(全部 8) 315.9ms
    # 開滿反而最慢（執行緒互搶）。預設 1 是為 Cloud Run 常見的 1 vCPU 而設，
    # 在多核機器上可調為 2~4。0 = 交給 onnxruntime 自行決定（不建議）。
    GROUNDING_INTRA_OP_THREADS: int = 1

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

    # --- 耗時工作派送 (Cloud Tasks) ---------------------------------------
    # LINE 要求 webhook 5 秒內回應，但買入掃描約 90 秒，故必須非同步執行。
    # Cloud Run 在「僅於處理請求時分配 CPU」模式下，回應後的背景工作會被節流而
    # 可能跑不完；改由 Cloud Tasks 發新請求回到 /api/trigger/*，即可重新取得 CPU。
    # 三項皆留空時退回 asyncio 背景執行（本機開發用，本機無節流問題）。
    SERVICE_BASE_URL: str = ""          # 例：https://investment-assistant-xxxx.a.run.app
    TASK_QUEUE_NAME: str = ""           # 例：investment-jobs
    TASK_QUEUE_LOCATION: str = "asia-east1"

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
