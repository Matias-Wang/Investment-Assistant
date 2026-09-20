import pytest
import os
import json
import tempfile
from app.models.domain import SystemConfig, PortfolioItem
from app.repositories.json_repo import JsonConfigRepository, JsonPortfolioRepository


@pytest.mark.asyncio
async def test_json_repo_load_and_update():
    # 建立臨時 JSON 檔案以隔離測試
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json", encoding="utf-8") as tmp:
        initial_data = {
            "system_config": {
                "trade_budget_per_order": 100000,
                "take_profit_min": 0.10,
                "take_profit_max": 0.20,
                "stop_loss": -0.50,
                "currency": "TWD"
            },
            "portfolio": [
                {
                    "ticker": "2330",
                    "name": "台積電",
                    "asset_type": "tw_stock",
                    "shares": 10,
                    "average_cost": 1000.0,
                    "currency": "TWD",
                    "purchase_date": "2026-09-01",
                    "last_updated": "2026-09-07T00:00:00Z",
                    "status": "holding"
                }
            ]
        }
        json.dump(initial_data, tmp)
        tmp_path = tmp.name

    try:
        cfg_repo = JsonConfigRepository(file_path=tmp_path)
        port_repo = JsonPortfolioRepository(file_path=tmp_path)

        # 1. 讀取設定
        cfg = await cfg_repo.get_config()
        assert cfg.trade_budget_per_order == 100000.0
        assert cfg.take_profit_min == 0.10

        # 2. 更新設定
        await cfg_repo.update_field("trade_budget_per_order", 150000.0)
        updated_cfg = await cfg_repo.get_config()
        assert updated_cfg.trade_budget_per_order == 150000.0

        # 3. 讀取持股
        items = await port_repo.get_all(status="holding")
        assert len(items) == 1
        assert items[0].ticker == "2330"
        assert items[0].shares == 10.0

        # 4. 加碼買入 (10 股 @ 1200 -> 總股數 20，加權平均成本 (10*1000 + 10*1200)/20 = 1100)
        updated_item = await port_repo.record_buy(ticker="2330", shares=10.0, price=1200.0)
        assert updated_item.shares == 20.0
        assert updated_item.average_cost == 1100.0

        # 5. 部分賣出 (賣出 5 股 -> 剩 15 股，成本仍為 1100)
        after_sell = await port_repo.record_sell(ticker="2330", shares=5.0)
        assert after_sell.shares == 15.0
        assert after_sell.status == "holding"

        # 6. 全數賣出 (賣出 15 股 -> 剩 0 股，status = "sold")
        sold_out = await port_repo.record_sell(ticker="2330", shares=15.0)
        assert sold_out.shares == 0.0
        assert sold_out.status == "sold"

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
