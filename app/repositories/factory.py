from app.core.config import settings
from app.core.logging import logger
from app.repositories.base import ConfigRepository, PortfolioRepository, SessionRepository
from app.repositories.json_repo import JsonConfigRepository, JsonPortfolioRepository, MemorySessionRepository

_config_repo: ConfigRepository = None
_portfolio_repo: PortfolioRepository = None
_session_repo: SessionRepository = None


def get_config_repository() -> ConfigRepository:
    global _config_repo
    if _config_repo is None:
        if settings.DB_BACKEND.lower() == "firestore":
            from app.repositories.firestore_repo import FirestoreConfigRepository
            _config_repo = FirestoreConfigRepository()
            logger.info("[RepositoryFactory] Initialized FirestoreConfigRepository")
        else:
            if settings.APP_ENV.lower() == "production":
                raise RuntimeError(
                    "[RepositoryFactory] Running JSON backend in production is prohibited to prevent data loss on Cloud Run. Set DB_BACKEND=firestore."
                )
            _config_repo = JsonConfigRepository(file_path=settings.JSON_DATA_PATH)
            logger.info(f"[RepositoryFactory] Initialized JsonConfigRepository with {settings.JSON_DATA_PATH}")
    return _config_repo


def get_portfolio_repository() -> PortfolioRepository:
    global _portfolio_repo
    if _portfolio_repo is None:
        if settings.DB_BACKEND.lower() == "firestore":
            from app.repositories.firestore_repo import FirestorePortfolioRepository
            _portfolio_repo = FirestorePortfolioRepository()
            logger.info("[RepositoryFactory] Initialized FirestorePortfolioRepository")
        else:
            if settings.APP_ENV.lower() == "production":
                raise RuntimeError(
                    "[RepositoryFactory] Running JSON backend in production is prohibited to prevent data loss on Cloud Run. Set DB_BACKEND=firestore."
                )
            _portfolio_repo = JsonPortfolioRepository(file_path=settings.JSON_DATA_PATH)
            logger.info(f"[RepositoryFactory] Initialized JsonPortfolioRepository with {settings.JSON_DATA_PATH}")
    return _portfolio_repo


def get_session_repository() -> SessionRepository:
    global _session_repo
    if _session_repo is None:
        if settings.DB_BACKEND.lower() == "firestore":
            from app.repositories.firestore_repo import FirestoreSessionRepository
            _session_repo = FirestoreSessionRepository()
            logger.info("[RepositoryFactory] Initialized FirestoreSessionRepository")
        else:
            _session_repo = MemorySessionRepository()
            logger.info("[RepositoryFactory] Initialized MemorySessionRepository")
    return _session_repo
