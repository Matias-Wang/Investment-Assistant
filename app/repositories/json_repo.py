import json
import os
import asyncio
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional, Any, Dict
from app.models.domain import SystemConfig, PortfolioItem
from app.models.line import UserSession
from app.repositories.base import ConfigRepository, PortfolioRepository, SessionRepository, ConflictError
from app.core.logging import logger


class JsonRepositoryBase:
    """僅供本機開發，Cloud Run 上會因容器暫時性儲存而遺失資料。"""
    _lock = asyncio.Lock()

    def __init__(self, file_path: str = "data/assets.json"):
        self.file_path = file_path
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        if not os.path.exists(self.file_path):
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            default_data = {
                "system_config": SystemConfig().model_dump(mode="json"),
                "portfolio": []
            }
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(default_data, f, ensure_ascii=False, indent=2, default=str)

    def _read_data(self) -> Dict[str, Any]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_data(self, data: Dict[str, Any]) -> None:
        temp_file = f"{self.file_path}.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        os.replace(temp_file, self.file_path)


class JsonConfigRepository(JsonRepositoryBase, ConfigRepository):
    async def get_config(self) -> SystemConfig:
        async with self._lock:
            data = self._read_data()
            cfg_data = data.get("system_config", {})
            return SystemConfig(**cfg_data)

    async def update_config(self, config: SystemConfig) -> SystemConfig:
        async with self._lock:
            data = self._read_data()
            data["system_config"] = config.model_dump(mode="json")
            self._write_data(data)
            logger.info(f"[JsonConfigRepository] Updated system config: {config.model_dump(mode='json')}")
            return config

    async def update_field(self, field_name: str, value: Any) -> SystemConfig:
        async with self._lock:
            data = self._read_data()
            cfg_dict = data.get("system_config", {})
            cfg_dict[field_name] = value
            new_cfg = SystemConfig(**cfg_dict)
            data["system_config"] = new_cfg.model_dump(mode="json")
            self._write_data(data)
            logger.info(f"[JsonConfigRepository] Updated config field {field_name}={value}")
            return new_cfg


class JsonPortfolioRepository(JsonRepositoryBase, PortfolioRepository):
    async def get_all(self, status: Optional[str] = "holding") -> List[PortfolioItem]:
        async with self._lock:
            data = self._read_data()
            portfolio_raw = data.get("portfolio", [])
            items = [PortfolioItem(**item) for item in portfolio_raw]
            if status:
                items = [item for item in items if item.status == status]
            return items

    async def get_by_ticker(self, ticker: str) -> Optional[PortfolioItem]:
        async with self._lock:
            data = self._read_data()
            portfolio_raw = data.get("portfolio", [])
            for item in portfolio_raw:
                if item.get("ticker").strip().upper() == ticker.strip().upper():
                    return PortfolioItem(**item)
            return None

    async def upsert(self, item: PortfolioItem) -> PortfolioItem:
        async with self._lock:
            data = self._read_data()
            portfolio_raw = data.get("portfolio", [])
            found = False
            for idx, raw in enumerate(portfolio_raw):
                if raw.get("ticker", "").strip().upper() == item.ticker.strip().upper():
                    curr_ver = int(raw.get("version", 1))
                    if item.version is not None and item.version < curr_ver:
                        raise ConflictError(f"Version conflict for {item.ticker}: current is {curr_ver}, got {item.version}")
                    item.version = curr_ver + 1
                    portfolio_raw[idx] = item.model_dump(mode="json")
                    found = True
                    break
            if not found:
                item.version = 1
                portfolio_raw.append(item.model_dump(mode="json"))
            data["portfolio"] = portfolio_raw
            self._write_data(data)
            return item

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

        async with self._lock:
            data = self._read_data()
            portfolio_raw = data.get("portfolio", [])
            now_iso = datetime.now(timezone.utc).isoformat()

            existing_idx = None
            for idx, raw in enumerate(portfolio_raw):
                if raw.get("ticker").strip().upper() == ticker.strip().upper():
                    existing_idx = idx
                    break

            if existing_idx is not None:
                item_data = portfolio_raw[existing_idx]
                old_shares = Decimal(str(item_data.get("shares", 0.0)))
                old_cost = Decimal(str(item_data.get("average_cost", 0.0)))
                was_holding = item_data.get("status") == "holding"
                curr_ver = int(item_data.get("version", 1))

                if was_holding and old_shares > 0:
                    total_shares = old_shares + shares_dec
                    if total_shares <= 0:
                        raise ValueError("invalid resulting share count")
                    new_avg_cost = ((old_shares * old_cost) + (shares_dec * price_dec)) / total_shares
                else:
                    total_shares = shares_dec
                    new_avg_cost = price_dec

                new_avg_cost = new_avg_cost.quantize(Decimal("0.0001"), ROUND_HALF_UP)
                item_data["shares"] = total_shares
                item_data["average_cost"] = new_avg_cost
                item_data["last_updated"] = now_iso
                item_data["status"] = "holding"
                item_data["version"] = curr_ver + 1
                if name:
                    item_data["name"] = name
                updated_item = PortfolioItem(**item_data)
                portfolio_raw[existing_idx] = updated_item.model_dump(mode="json")
            else:
                updated_item = PortfolioItem(
                    ticker=ticker.strip().upper(),
                    name=name or ticker,
                    asset_type=asset_type,
                    shares=shares_dec,
                    average_cost=price_dec.quantize(Decimal("0.0001"), ROUND_HALF_UP),
                    currency=currency,
                    purchase_date=datetime.now().strftime("%Y-%m-%d"),
                    last_updated=now_iso,
                    status="holding",
                    version=1,
                )
                portfolio_raw.append(updated_item.model_dump(mode="json"))

            data["portfolio"] = portfolio_raw
            self._write_data(data)
            logger.info(f"[JsonPortfolioRepository] Recorded buy for {ticker}: {shares} shares @ {price} (v{updated_item.version})")
            return updated_item

    async def record_sell(self, ticker: str, shares: float) -> Optional[PortfolioItem]:
        if shares <= 0:
            raise ValueError("shares to sell must be positive")

        shares_dec = Decimal(str(shares))

        async with self._lock:
            data = self._read_data()
            portfolio_raw = data.get("portfolio", [])
            now_iso = datetime.now(timezone.utc).isoformat()

            existing_idx = None
            for idx, raw in enumerate(portfolio_raw):
                if raw.get("ticker").strip().upper() == ticker.strip().upper():
                    existing_idx = idx
                    break

            if existing_idx is None:
                logger.warning(f"[JsonPortfolioRepository] Attempted to sell {ticker} which does not exist in portfolio.")
                return None

            item_data = portfolio_raw[existing_idx]
            current_shares = Decimal(str(item_data.get("shares", 0.0)))
            curr_ver = int(item_data.get("version", 1))

            if shares_dec >= current_shares:
                item_data["shares"] = Decimal("0.0")
                item_data["status"] = "sold"
            else:
                item_data["shares"] = current_shares - shares_dec
                item_data["status"] = "holding"

            item_data["last_updated"] = now_iso
            item_data["version"] = curr_ver + 1
            updated_item = PortfolioItem(**item_data)
            portfolio_raw[existing_idx] = updated_item.model_dump(mode="json")
            data["portfolio"] = portfolio_raw
            self._write_data(data)
            logger.info(f"[JsonPortfolioRepository] Recorded sell for {ticker}: {shares} shares (v{updated_item.version})")
            return updated_item


class MemorySessionRepository(SessionRepository):
    """記憶體內的會話狀態管理 (具備 TTL)"""

    def __init__(self, ttl_seconds: float = 600.0):
        self._sessions: Dict[str, UserSession] = {}
        self._ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()

    async def get_session(self, user_id: str) -> Optional[UserSession]:
        async with self._lock:
            session = self._sessions.get(user_id)
            if not session:
                return None
            import time
            if time.time() - session.updated_at > self._ttl_seconds:
                del self._sessions[user_id]
                return None
            return session

    async def save_session(self, session: UserSession) -> None:
        async with self._lock:
            import time
            session.updated_at = time.time()
            self._sessions[session.user_id] = session

    async def clear_session(self, user_id: str) -> None:
        async with self._lock:
            if user_id in self._sessions:
                del self._sessions[user_id]
