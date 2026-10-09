from abc import ABC, abstractmethod
from typing import List, Optional, Any
from app.models.domain import SystemConfig, PortfolioItem
from app.models.line import UserSession


class ConflictError(Exception):
    """樂觀鎖版本衝突例外"""
    pass


class ConfigRepository(ABC):
    """系統設定資料庫抽象介面"""

    @abstractmethod
    async def get_config(self) -> SystemConfig:
        """取得目前系統設定"""
        pass

    @abstractmethod
    async def update_config(self, config: SystemConfig) -> SystemConfig:
        """更新整份系統設定"""
        pass

    @abstractmethod
    async def update_field(self, field_name: str, value: Any) -> SystemConfig:
        """動態更新特定參數 (如 budget, take_profit, stop_loss)"""
        pass


class PortfolioRepository(ABC):
    """持股庫存資料庫抽象介面"""

    @abstractmethod
    async def get_all(self, status: Optional[str] = "holding") -> List[PortfolioItem]:
        """取得所有持股 (預設只取 holding 狀態)"""
        pass

    @abstractmethod
    async def get_by_ticker(self, ticker: str) -> Optional[PortfolioItem]:
        """依代號取得持股"""
        pass

    @abstractmethod
    async def upsert(self, item: PortfolioItem) -> PortfolioItem:
        """新增或更新持股項目"""
        pass

    @abstractmethod
    async def record_buy(
        self,
        ticker: str,
        shares: float,
        price: float,
        name: Optional[str] = None,
        asset_type: str = "tw_stock",
        currency: str = "TWD"
    ) -> PortfolioItem:
        """
        記錄買入：若已有部位則依加權平均計算新成本與累加股數；若無則新增部位
        """
        pass

    @abstractmethod
    async def record_sell(self, ticker: str, shares: float) -> Optional[PortfolioItem]:
        """
        記錄賣出：扣減股數；若股數歸零或不足則更新 status 為 'sold'。
        標的不存在或目前未持有（已賣光）時回傳 None，且不寫入任何變更。
        """
        pass


class SessionRepository(ABC):
    """使用者對話狀態快取介面"""

    @abstractmethod
    async def get_session(self, user_id: str) -> Optional[UserSession]:
        pass

    @abstractmethod
    async def save_session(self, session: UserSession) -> None:
        pass

    @abstractmethod
    async def clear_session(self, user_id: str) -> None:
        pass
