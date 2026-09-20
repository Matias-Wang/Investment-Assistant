from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional, Any
from google.cloud import firestore
from google.cloud.firestore import async_transactional
from app.models.domain import SystemConfig, PortfolioItem
from app.models.line import UserSession
from app.repositories.base import ConfigRepository, PortfolioRepository, SessionRepository, ConflictError
from app.core.config import settings
from app.core.logging import logger
from app.core.retry import with_retry


class FirestoreBase:
    def __init__(self):
        try:
            if settings.GCP_PROJECT_ID:
                self.db = firestore.AsyncClient(project=settings.GCP_PROJECT_ID, database=settings.FIRESTORE_DATABASE)
            else:
                self.db = firestore.AsyncClient(database=settings.FIRESTORE_DATABASE)
        except Exception as e:
            logger.error(f"[FirestoreBase] Failed to initialize Firestore client: {e}")
            raise


class FirestoreConfigRepository(FirestoreBase, ConfigRepository):
    def __init__(self):
        super().__init__()
        self.collection = settings.FIRESTORE_COLLECTION_CONFIG
        self.doc_id = "default"

    async def get_config(self) -> SystemConfig:
        try:
            doc_ref = self.db.collection(self.collection).document(self.doc_id)
            doc = await doc_ref.get()
            if doc.exists:
                return SystemConfig(**doc.to_dict())
            # Initialize default
            default_config = SystemConfig()
            await doc_ref.set(default_config.model_dump(mode="json"))
            return default_config
        except Exception as e:
            logger.error(f"[FirestoreConfigRepository] get_config error: {e}")
            raise

    async def update_config(self, config: SystemConfig) -> SystemConfig:
        try:
            doc_ref = self.db.collection(self.collection).document(self.doc_id)
            await doc_ref.set(config.model_dump(mode="json"))
            logger.info(f"[FirestoreConfigRepository] Updated config: {config.model_dump(mode='json')}")
            return config
        except Exception as e:
            logger.error(f"[FirestoreConfigRepository] update_config error: {e}")
            raise

    async def update_field(self, field_name: str, value: Any) -> SystemConfig:
        try:
            doc_ref = self.db.collection(self.collection).document(self.doc_id)
            await doc_ref.update({field_name: value})
            return await self.get_config()
        except Exception as e:
            logger.error(f"[FirestoreConfigRepository] update_field error: {e}")
            raise


class FirestorePortfolioRepository(FirestoreBase, PortfolioRepository):
    def __init__(self):
        super().__init__()
        self.collection = settings.FIRESTORE_COLLECTION_PORTFOLIO

    async def get_all(self, status: Optional[str] = "holding") -> List[PortfolioItem]:
        try:
            col_ref = self.db.collection(self.collection)
            if status:
                from google.cloud.firestore_v1.base_query import FieldFilter
                query = col_ref.where(filter=FieldFilter("status", "==", status))
            else:
                query = col_ref
            docs = query.stream()
            items = []
            async for doc in docs:
                items.append(PortfolioItem(**doc.to_dict()))
            return items
        except Exception as e:
            logger.error(f"[FirestorePortfolioRepository] get_all error: {e}")
            raise

    async def get_by_ticker(self, ticker: str) -> Optional[PortfolioItem]:
        try:
            doc_ref = self.db.collection(self.collection).document(ticker.strip().upper())
            doc = await doc_ref.get()
            if doc.exists:
                return PortfolioItem(**doc.to_dict())
            return None
        except Exception as e:
            logger.error(f"[FirestorePortfolioRepository] get_by_ticker error: {e}")
            raise

    async def upsert(self, item: PortfolioItem) -> PortfolioItem:
        try:
            ticker_key = item.ticker.strip().upper()
            doc_ref = self.db.collection(self.collection).document(ticker_key)
            doc = await doc_ref.get()
            if doc.exists:
                current_data = doc.to_dict()
                curr_version = int(current_data.get("version", 1))
                if item.version is not None and item.version < curr_version:
                    raise ConflictError(f"Version conflict for {ticker_key}: current version is {curr_version}, got {item.version}")
                item.version = curr_version + 1
            else:
                item.version = 1
            await doc_ref.set(item.model_dump(mode="json"))
            return item
        except ConflictError:
            raise
        except Exception as e:
            logger.error(f"[FirestorePortfolioRepository] upsert error: {e}")
            raise

    async def record_buy(
        self,
        ticker: str,
        shares: float,
        price: float,
        name: Optional[str] = None,
        asset_type: str = "tw_stock",
        currency: str = "TWD"
    ) -> PortfolioItem:
        if shares <= 0 or price <= 0:
            raise ValueError("shares and price must be positive")

        shares_dec = Decimal(str(shares))
        price_dec = Decimal(str(price))

        ticker_key = ticker.strip().upper()
        doc_ref = self.db.collection(self.collection).document(ticker_key)
        transaction = self.db.transaction()

        @async_transactional
        async def _buy_in_tx(tx) -> PortfolioItem:
            doc = await doc_ref.get(transaction=tx)
            now_iso = datetime.now(timezone.utc).isoformat()

            if doc.exists:
                data = doc.to_dict()
                old_shares = Decimal(str(data.get("shares", 0.0)))
                old_cost = Decimal(str(data.get("average_cost", 0.0)))
                was_holding = data.get("status") == "holding"
                curr_version = int(data.get("version", 1))

                if was_holding and old_shares > 0:
                    total_shares = old_shares + shares_dec
                    if total_shares <= 0:
                        raise ValueError("invalid resulting share count")
                    new_avg_cost = ((old_shares * old_cost) + (shares_dec * price_dec)) / total_shares
                else:
                    total_shares = shares_dec
                    new_avg_cost = price_dec

                new_avg_cost = new_avg_cost.quantize(Decimal("0.0001"), ROUND_HALF_UP)
                data["shares"] = total_shares
                data["average_cost"] = new_avg_cost
                data["last_updated"] = now_iso
                data["status"] = "holding"
                data["version"] = curr_version + 1
                if name:
                    data["name"] = name
                item = PortfolioItem(**data)
            else:
                item = PortfolioItem(
                    ticker=ticker_key,
                    name=name or ticker_key,
                    asset_type=asset_type,
                    shares=shares_dec,
                    average_cost=price_dec.quantize(Decimal("0.0001"), ROUND_HALF_UP),
                    currency=currency,
                    purchase_date=datetime.now().strftime("%Y-%m-%d"),
                    last_updated=now_iso,
                    status="holding",
                    version=1,
                )

            tx.set(doc_ref, item.model_dump(mode="json"))
            return item

        try:
            item = await _buy_in_tx(transaction)
            logger.info(f"[FirestorePortfolioRepository] Recorded buy {ticker_key}: {shares} @ {price} (v{item.version})")
            return item
        except Exception as e:
            logger.error(f"[FirestorePortfolioRepository] record_buy error: {e}")
            raise

    async def record_sell(self, ticker: str, shares: float) -> Optional[PortfolioItem]:
        if shares <= 0:
            raise ValueError("shares to sell must be positive")

        shares_dec = Decimal(str(shares))

        ticker_key = ticker.strip().upper()
        doc_ref = self.db.collection(self.collection).document(ticker_key)
        transaction = self.db.transaction()

        @async_transactional
        async def _sell_in_tx(tx) -> Optional[PortfolioItem]:
            doc = await doc_ref.get(transaction=tx)
            if not doc.exists:
                logger.warning(f"[FirestorePortfolioRepository] Stock {ticker_key} not found for sell.")
                return None

            data = doc.to_dict()
            current_shares = Decimal(str(data.get("shares", 0.0)))
            curr_version = int(data.get("version", 1))
            now_iso = datetime.now(timezone.utc).isoformat()

            if shares_dec >= current_shares:
                data["shares"] = Decimal("0.0")
                data["status"] = "sold"
            else:
                data["shares"] = current_shares - shares_dec
                data["status"] = "holding"

            data["last_updated"] = now_iso
            data["version"] = curr_version + 1
            item = PortfolioItem(**data)
            tx.set(doc_ref, item.model_dump(mode="json"))
            return item

        try:
            item = await _sell_in_tx(transaction)
            if item:
                logger.info(f"[FirestorePortfolioRepository] Recorded sell {ticker_key}: {shares} shares (v{item.version}).")
            return item
        except Exception as e:
            logger.error(f"[FirestorePortfolioRepository] record_sell error: {e}")
            raise


class FirestoreSessionRepository(FirestoreBase, SessionRepository):
    """
    Google Cloud Firestore 使用者對話狀態倉儲實作
    解決 Cloud Run 無狀態多實例環境下的 session 遺失問題
    """

    def __init__(self):
        super().__init__()
        self.collection = "user_sessions"

    async def get_session(self, user_id: str) -> Optional[UserSession]:
        try:
            doc_ref = self.db.collection(self.collection).document(user_id)
            doc = await doc_ref.get()
            if doc.exists:
                return UserSession(**doc.to_dict())
            return None
        except Exception as e:
            logger.error(f"[FirestoreSessionRepository] get_session error: {e}")
            return None

    async def save_session(self, session: UserSession) -> None:
        try:
            doc_ref = self.db.collection(self.collection).document(session.user_id)
            await doc_ref.set(session.model_dump(mode="json"))
        except Exception as e:
            logger.error(f"[FirestoreSessionRepository] save_session error: {e}")
            raise

    async def clear_session(self, user_id: str) -> None:
        try:
            doc_ref = self.db.collection(self.collection).document(user_id)
            await doc_ref.delete()
        except Exception as e:
            logger.error(f"[FirestoreSessionRepository] clear_session error: {e}")
            raise
