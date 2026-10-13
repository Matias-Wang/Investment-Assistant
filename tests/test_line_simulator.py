"""本地 LINE 模擬器 (scripts/test_ui.py) 的核心行為與隔離保證。

只測 `LineSimulator`（不需 streamlit）：訊息確實經簽章送進真實 `/callback`、
回覆被攔截而非送往 LINE，且隔離設定生效。
"""

import importlib.util
from pathlib import Path

import pytest
from starlette.datastructures import State

import app.main as app_main
from app.core.config import settings
from app.repositories import factory
from app.repositories.json_repo import JsonConfigRepository, JsonPortfolioRepository, MemorySessionRepository
from app.services import line_bot_service as line_bot_module

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "test_ui.py"
_SPEC = importlib.util.spec_from_file_location("line_simulator_ui", SCRIPT_PATH)
sim_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sim_module)

# 模擬「.env 指向正式環境」的最壞情況：隔離覆寫必須全部蓋掉
PRODUCTION_LIKE = {
    "DB_BACKEND": "firestore",
    "SERVICE_BASE_URL": "https://investment-assistant.example.run.app",
    "TASK_QUEUE_NAME": "investment-jobs",
    "GCP_PROJECT_ID": "prod-project",
    "LINE_USER_ID": "U_real_owner",
    "ALLOWED_LINE_USER_IDS": ["U_real_owner"],
    "LINE_CHANNEL_ACCESS_TOKEN": "real-token",
    "ENABLE_SCHEDULER": True,
}


@pytest.fixture
def isolated_globals(monkeypatch):
    """讓模擬器改動的全域狀態（settings、倉儲單例、MessagingApi、app.state）於測試後還原。"""
    for key in set(PRODUCTION_LIKE) | set(sim_module.isolation_overrides(Path("x"), "s", [])):
        monkeypatch.setattr(settings, key, getattr(settings, key))
    for name in ("_config_repo", "_portfolio_repo", "_session_repo"):
        monkeypatch.setattr(factory, name, None)
    monkeypatch.setattr(line_bot_module, "MessagingApi", line_bot_module.MessagingApi)
    monkeypatch.setattr(app_main.app, "state", State())
    return monkeypatch


@pytest.fixture
def simulator(tmp_path, isolated_globals):
    secret = "unit-test-secret"
    for key, value in PRODUCTION_LIKE.items():
        isolated_globals.setattr(settings, key, value)
    for key, value in sim_module.isolation_overrides(
        tmp_path / "assets.json", secret, [sim_module.DEFAULT_USER_ID]
    ).items():
        isolated_globals.setattr(settings, key, value)

    sim = sim_module.LineSimulator(channel_secret=secret, log_dir=tmp_path / "logs")
    yield sim
    sim.close()


def test_isolation_overrides_production_like_settings(simulator, tmp_path):
    state = simulator.app.state
    assert isinstance(state.config_repo, JsonConfigRepository)
    assert isinstance(state.portfolio_repo, JsonPortfolioRepository)
    assert isinstance(state.session_repo, MemorySessionRepository)
    assert Path(state.portfolio_repo.file_path) == tmp_path / "assets.json"
    assert simulator.line_service.task_dispatcher.enabled is False
    assert simulator.line_service.channel_token == sim_module.SIMULATOR_TOKEN
    assert settings.LINE_USER_ID is None
    assert "U_real_owner" not in settings.ALLOWED_LINE_USER_IDS
    assert state.scheduler_service.scheduler.running is False


def test_recreating_simulator_closes_previous_instance(simulator, tmp_path):
    from app.core.logging import logger as app_logger

    handlers_before = len(app_logger.handlers)
    second = sim_module.LineSimulator(channel_secret=settings.LINE_CHANNEL_SECRET, log_dir=tmp_path / "logs2")
    try:
        assert simulator._loop.is_closed()
        assert len(app_logger.handlers) == handlers_before
        assert second.send_text(sim_module.DEFAULT_USER_ID, "查詢設定")["replied"] is True
    finally:
        second.close()


def _reply_texts(sim, user_id):
    return [
        m.get("text", m.get("altText", ""))
        for r in sim.records_for(user_id) if r["kind"] == "reply"
        for m in r["messages"]
    ]


def test_text_message_goes_through_real_webhook_and_reply_is_captured(simulator):
    result = simulator.send_text(sim_module.DEFAULT_USER_ID, "查詢設定")

    assert result["http_status"] == 200
    assert result["replied"] is True
    assert "目前系統策略參數" in _reply_texts(simulator, sim_module.DEFAULT_USER_ID)[0]
    reply = next(r for r in simulator.records_for(sim_module.DEFAULT_USER_ID) if r["kind"] == "reply")
    assert reply["messages"][-1]["quickReply"]["items"], "應掛上預設 Quick Reply"


def test_isolation_blocks_real_line_api_and_cloud_tasks(simulator):
    assert line_bot_module.MessagingApi is sim_module._BlockedMessagingApi
    assert simulator.line_service.task_dispatcher.enabled is False
    with pytest.raises(RuntimeError):
        line_bot_module.MessagingApi(None)


def test_unauthorized_user_is_rejected(simulator):
    simulator.send_text("U_intruder", "查詢設定")
    assert "未獲授權" in _reply_texts(simulator, "U_intruder")[0]

    simulator.set_authorized("U_intruder", True)
    simulator.send_text("U_intruder", "查詢設定")
    assert "目前系統策略參數" in _reply_texts(simulator, "U_intruder")[1]


def test_postback_then_confirm_records_buy(simulator):
    uid = sim_module.DEFAULT_USER_ID
    simulator.send_postback(uid, "action=buy_click&ticker=2330&shares=8&price=1050", "手動已買入")
    assert simulator.get_session(uid)["awaiting_action"] == "CONFIRM_BUY"

    simulator.send_text(uid, "確認")
    assert simulator.get_session(uid) is None
    holdings = {p["ticker"]: p for p in simulator.get_portfolio()}
    assert float(holdings["2330"]["shares"]) == 8


@pytest.mark.parametrize("report", [
    "賣出2313 成交價254.5 500股",
    "賣出 2313 500",
    "賣出 2313 254.5 500",
])
def test_sell_report_updates_persisted_portfolio(simulator, report):
    import json

    uid = sim_module.DEFAULT_USER_ID
    simulator.send_text(uid, "買入 2313 200 1000")
    result = simulator.send_text(uid, report)
    assert result["http_status"] == 200
    assert "成功" in _reply_texts(simulator, uid)[-1]
    holdings = {p["ticker"]: p for p in simulator.get_portfolio()}
    assert float(holdings["2313"]["shares"]) == 500
    data = json.loads(Path(simulator.app.state.portfolio_repo.file_path).read_text(encoding="utf-8"))
    item = next(p for p in data["portfolio"] if p["ticker"] == "2313")
    assert float(item["shares"]) == 500
    assert float(item["average_cost"]) == 200
    assert item["status"] == "holding"


@pytest.mark.parametrize("confirmation", ["確認", "254.5 500"])
def test_postback_sell_updates_portfolio(simulator, confirmation):
    uid = sim_module.DEFAULT_USER_ID
    simulator.send_text(uid, "買入 2313 200 1000")
    simulator.send_postback(uid, "action=sell_click&ticker=2313&shares=500&price=254.5", "手動已賣出")
    simulator.send_text(uid, confirmation)
    assert simulator.get_session(uid) is None
    holdings = {p["ticker"]: p for p in simulator.get_portfolio()}
    assert float(holdings["2313"]["shares"]) == 500
    simulator.send_text(uid, "賣出2313 成交價254.5 500股")
    assert "2313" not in {p["ticker"] for p in simulator.get_portfolio()}


SEED_CONTENT = {
    "system_config": {
        "trade_budget_per_order": "88000.0", "profit_alert_threshold": 0.05,
        "take_profit_min": 0.1, "take_profit_max": 0.2, "stop_loss": -0.5, "currency": "TWD",
    },
    "portfolio": [{
        "ticker": "2330", "name": "台積電", "shares": "10", "average_cost": "900",
        "currency": "TWD", "status": "holding", "purchase_date": "2026-09-01",
    }],
}


def test_ui_fixture_matches_repository_schema():
    """模擬器預設的測試資料必須能被正式的資料模型解析，且不是真實資料檔。"""
    import json
    from app.models.domain import PortfolioItem, SystemConfig

    assert sim_module.SEED_DATA_PATH.name == "ui_test_assets.json"
    data = json.loads(sim_module.SEED_DATA_PATH.read_text(encoding="utf-8"))
    SystemConfig(**data["system_config"])
    items = [PortfolioItem(**raw) for raw in data["portfolio"]]

    holding = [i.ticker for i in items if i.status == "holding"]
    assert holding == ["2330", "0050", "2317", "2454", "2603", "AAPL"]
    assert [i.ticker for i in items if i.status == "sold"] == ["2882"]
    assert all(i.last_alert_at is None for i in items), "預設不可帶警示冷卻，否則賣出檢查不會推播"
    assert len({i.ticker for i in items}) == len(items)


def test_seed_copies_source_only_when_target_missing(tmp_path):
    import json
    source, target = tmp_path / "real.json", tmp_path / "sim" / "sim.json"
    source.write_text(json.dumps(SEED_CONTENT), encoding="utf-8")

    assert sim_module.seed_data_file(target, source) is True
    assert json.loads(target.read_text(encoding="utf-8")) == SEED_CONTENT

    target.write_text("{}", encoding="utf-8")
    assert sim_module.seed_data_file(target, source) is False, "已存在時不可覆蓋測試中的資料"
    assert target.read_text(encoding="utf-8") == "{}"

    assert sim_module.seed_data_file(tmp_path / "x.json", tmp_path / "missing.json") is False
    with pytest.raises(ValueError):
        sim_module.seed_data_file(source, source)


def test_seeded_holdings_are_visible_and_source_is_never_written(tmp_path, isolated_globals):
    import json
    source = tmp_path / "real.json"
    source.write_text(json.dumps(SEED_CONTENT), encoding="utf-8")
    original = source.read_bytes()
    data_path = tmp_path / "sim.json"
    sim_module.seed_data_file(data_path, source)

    secret = "seed-secret"
    for key, value in sim_module.isolation_overrides(data_path, secret, [sim_module.DEFAULT_USER_ID]).items():
        isolated_globals.setattr(settings, key, value)
    sim = sim_module.LineSimulator(channel_secret=secret, log_dir=tmp_path / "logs")
    try:
        uid = sim_module.DEFAULT_USER_ID
        assert [p["ticker"] for p in sim.get_portfolio()] == ["2330"]
        sim.send_text(uid, "查詢設定")
        assert "88,000" in _reply_texts(sim, uid)[-1]

        sim.send_text(uid, "買入 2454 1000 5")
        sim.send_text(uid, "賣出 2330 950 10")
        assert [p["ticker"] for p in sim.get_portfolio()] == ["2454"]
        assert source.read_bytes() == original, "模擬器的買賣不可寫回來源檔"

        assert sim.reset_data(source) is True
        assert [p["ticker"] for p in sim.get_portfolio()] == ["2330"]
        assert source.read_bytes() == original
    finally:
        sim.close()


def test_records_are_written_to_log_dir(simulator, tmp_path):
    simulator.send_text(sim_module.DEFAULT_USER_ID, "說明")
    jsonl = list((tmp_path / "logs").glob("simulator-*.jsonl"))
    assert jsonl and sim_module.DEFAULT_USER_ID in jsonl[0].read_text(encoding="utf-8")


def test_iter_flex_extracts_text_and_postback_buttons():
    bubble = {
        "type": "bubble",
        "body": {"type": "box", "contents": [
            {"type": "text", "text": "2330 台積電", "weight": "bold"},
            {"type": "separator"},
        ]},
        "footer": {"type": "box", "contents": [
            {"type": "button", "action": {"type": "postback", "label": "手動已買入", "data": "action=buy_click"}},
        ]},
    }
    items = list(sim_module.iter_flex(bubble))
    assert items[0] == ("text", ("2330 台積電", True))
    assert items[1] == ("separator", None)
    assert items[2][1]["label"] == "手動已買入"


def test_streamlit_entrypoint_applies_isolation(tmp_path, isolated_globals):
    """走真正的 `streamlit run` 路徑（main() → get_simulator()），而非直接建構 LineSimulator。"""
    st = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    import os
    monkeypatch = isolated_globals
    monkeypatch.chdir(os.getcwd())  # main() 會 chdir 至專案根目錄
    # 先登記再覆寫，測試結束時 monkeypatch 會把 main() 寫入的環境變數一併還原
    for key in sim_module.isolation_overrides(Path("x"), "s", []):
        monkeypatch.setenv(key, "placeholder")
    for key, value in PRODUCTION_LIKE.items():
        env_value = ",".join(value) if isinstance(value, list) else str(value)
        monkeypatch.setenv(key, env_value)
        monkeypatch.setattr(settings, key, value)
    monkeypatch.setenv("LINE_SIMULATOR_LOG_DIR", str(tmp_path / "logs"))

    at = AppTest.from_file(str(SCRIPT_PATH), default_timeout=120)
    try:
        at.run()
        assert not at.exception, at.exception
        assert settings.DB_BACKEND == "json"
        assert Path(settings.JSON_DATA_PATH) == tmp_path / "logs" / "simulator_assets.json"
        assert line_bot_module.MessagingApi.__name__ == "_BlockedMessagingApi"
        sim = app_main.app.state.line_simulator
        assert sim.line_service.task_dispatcher.enabled is False

        at.chat_input[0].set_value("查詢設定").run()
        assert not at.exception, at.exception
        assert any("目前系統策略參數" in m.value for m in at.markdown)
        assert list((tmp_path / "logs").glob("simulator-*.jsonl"))
    finally:
        sim = getattr(app_main.app.state, "line_simulator", None)
        if sim is not None:
            sim.close()
        st.cache_resource.clear()
