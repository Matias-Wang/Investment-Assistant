import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import logger
from app.repositories.factory import (
    get_config_repository,
    get_portfolio_repository,
    get_session_repository,
)
from app.services.market_service import MarketService
from app.services.news_service import NewsService
from app.services.llm_service import LLMService
from app.services.slm_service import SLMService
from app.services.sell_engine import SellMonitoringEngine
from app.services.buy_engine import BuyStrategyEngine
from app.services.line_bot_service import LineBotService
from app.services.scheduler_service import SchedulerService
from app.services.market_report_service import MarketReportService
from app.controllers.line_controller import router as line_router
from app.controllers.api_controller import router as api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    應用程式生命週期管理
    初始化各層服務、資料庫連線與定時排程引擎
    """
    logger.info("Initializing application services and repositories...")

    # 1. 初始化資料庫倉儲
    config_repo = get_config_repository()
    portfolio_repo = get_portfolio_repository()
    session_repo = get_session_repository()

    # 2. 初始化核心領域服務
    market_service = MarketService()
    news_service = NewsService(timeout_seconds=settings.HTTP_TIMEOUT_SECONDS)
    llm_service = LLMService()
    slm_service = SLMService()

    sell_engine = SellMonitoringEngine(
        portfolio_repo=portfolio_repo,
        config_repo=config_repo,
        market_service=market_service,
    )

    buy_engine = BuyStrategyEngine(
        config_repo=config_repo,
        portfolio_repo=portfolio_repo,
        market_service=market_service,
        news_service=news_service,
        llm_service=llm_service,
    )

    market_report_service = MarketReportService(
        portfolio_repo=portfolio_repo,
        news_service=news_service,
    )

    line_service = LineBotService(
        config_repo=config_repo,
        portfolio_repo=portfolio_repo,
        session_repo=session_repo,
        slm_service=slm_service,
    )

    scheduler_service = SchedulerService(
        sell_engine=sell_engine,
        buy_engine=buy_engine,
        line_service=line_service,
        market_report_service=market_report_service,
    )

    # 掛載至 app.state
    app.state.config_repo = config_repo
    app.state.portfolio_repo = portfolio_repo
    app.state.session_repo = session_repo
    app.state.market_service = market_service
    app.state.news_service = news_service
    app.state.llm_service = llm_service
    app.state.slm_service = slm_service
    app.state.sell_engine = sell_engine
    app.state.buy_engine = buy_engine
    app.state.market_report_service = market_report_service
    app.state.line_service = line_service
    app.state.scheduler_service = scheduler_service

    # 3. 啟動排程
    scheduler_service.start()

    # 4. 提早喚醒幻覺驗證模型（不阻塞啟動）
    #    min-instances 0 下每次排程掃描都是冷啟動。若等到 13:00 掃描時才載入模型，
    #    併發分析的前幾檔會在模型就緒前就跑完（fail-open 跳過驗證）。此處改為啟動後
    #    立即在背景執行緒載入，讓模型在掃描真正開始前就備妥。
    #    刻意不 await：載入約 7 秒，掛在啟動路徑上會拖慢冷啟動與 LINE 首則回覆。
    if (settings.GROUNDING_MODE or "").strip().lower() != "off":
        asyncio.create_task(asyncio.to_thread(buy_engine.grounding_verifier.warmup))

    logger.info("Investment Assistant application started successfully.")

    yield

    # 應用關閉時清理
    logger.info("Shutting down application...")
    scheduler_service.shutdown()


def create_app() -> FastAPI:
    is_prod = settings.APP_ENV.lower() == "production"

    app = FastAPI(
        title="LLM 投資策略與自動追蹤系統 (Investment Assistant)",
        description=(
            "整合 Python 量化篩選、Gemini 3.5 文本分析、LINE Bot 互動介面，"
            "以及 GCP Firestore / JSON 資料庫的自動監控與投資助理系統。"
        ),
        version="1.0.0",
        docs_url=None if is_prod else "/docs",
        redoc_url=None if is_prod else "/redoc",
        openapi_url=None if is_prod else "/openapi.json",
        lifespan=lifespan,
    )

    # 僅在有設定白名單時啟用 CORS
    if settings.CORS_ALLOWED_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.CORS_ALLOWED_ORIGINS,
            allow_credentials=False if "*" in settings.CORS_ALLOWED_ORIGINS else True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # 註冊路由
    app.include_router(line_router)
    app.include_router(api_router)

    @app.get("/")
    async def root():
        return {
            "app": settings.APP_NAME,
            "version": "1.0.0",
            "status": "running",
            "docs": None if is_prod else "/docs",
            "webhook_endpoint": "/callback",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
