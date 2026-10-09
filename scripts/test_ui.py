"""本地端到端 LINE 互動模擬器（Streamlit，僅供開發測試）。

啟動方式（需先安裝 requirements-dev.txt，於專案根目錄執行）::

    streamlit run scripts/test_ui.py

`.streamlit/config.toml` 已將監聽位址限定為 127.0.0.1；若不在專案根目錄啟動，
該設定不會套用，請改加 `--server.address 127.0.0.1`。

運作方式
--------
不是另外寫一套「模擬邏輯」，而是把真實的 FastAPI app 跑在本程序內：

1. 依 LINE Messaging API 的格式組出 webhook JSON（文字 / postback 事件），
   以本程序專用的 channel secret 算出 `X-Line-Signature`，
   經 httpx ASGITransport 打到真正的 `/callback` 端點。
   因此簽章驗證、事件去重、白名單、多輪對話、規則 NLU、Flex 組裝全都照正式路徑執行。
2. 攔截 `LineBotService._send_reply_sync` / `_send_push_sync`，把 Bot 原本要送給
   LINE 的訊息改存進對話紀錄，畫面上以聊天泡泡與 Flex 卡片呈現。
3. 所有往來寫入 `data_logs/simulator-YYYYMMDD.jsonl`，應用程式 Log 寫入
   `data_logs/app-YYYYMMDD.log`，方便依 user_id 事後比對。

環境隔離（啟動時強制覆寫 .env，見 `isolation_overrides`）
--------------------------------------------------------
* `DB_BACKEND=json`，資料檔改為 `data_logs/simulator_assets.json`：
  不會碰到 Firestore 正式資料，也不會讀寫本機的 `data/assets.json`。
  首次啟動（或按下重設）時，以測試資料 `tests/fixtures/ui_test_assets.json` 的**副本**
  作為初始持股，之後的買賣只寫入副本。
* 清空 `SERVICE_BASE_URL` / `TASK_QUEUE_NAME`：耗時工作一律本地執行，
  不會派 Cloud Tasks 去打線上服務。
* `ENABLE_SCHEDULER=False`：不在背景自動跑排程掃描。
* LINE token 換成假值，且 `MessagingApi` 被替換成一呼叫就拋錯的守門類別：
  即使攔截失效，也只會報錯而不會推播給真實使用者。
* 白名單只包含畫面上勾選授權的 user_id，與 `.env` 的真實帳號無關。

Gemini、yfinance、新聞來源等外部查詢仍使用 `.env` 設定真實呼叫，
這是刻意的：否則就不算端到端。觸發「買入建議」會產生 Gemini 費用。

注意：本檔名以 `test_` 開頭，會被 pytest 收集匯入，因此 import 時不得有任何副作用；
環境覆寫與 Streamlit 匯入都只在 `main()` 內進行。
"""

from __future__ import annotations

import asyncio
import base64
import collections
import hashlib
import hmac
import json
import logging
import os
import random
import secrets
import shutil
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 可由環境變數改到別處（測試用），預設為已被 .gitignore / .dockerignore / .gcloudignore 排除的 data_logs/
LOG_DIR = Path(os.environ.get("LINE_SIMULATOR_LOG_DIR") or PROJECT_ROOT / "data_logs")
DEFAULT_USER_ID = "U_test_simulator"
SIMULATOR_TOKEN = "simulator-token-never-sent"
LINE_REPLY_DEADLINE_SECONDS = 5.0

# 與 LineBotService._get_default_quick_reply 相同；在 Bot 尚未回覆任何訊息前先顯示
FALLBACK_QUICK_REPLIES = [
    ("📊 查詢庫存", "查詢庫存"),
    ("📑 市場報告", "市場報告"),
    ("💡 買入建議", "買入建議"),
    ("🚨 賣出檢查", "賣出檢查"),
    ("⚙️ 查詢設定", "查詢設定"),
    ("🎛️ 功能選單", "選單"),
]

# 模擬器資料檔的初始來源：專為 UI 測試設計的虛構持股（各標的對應不同的警示情境，
# 說明見 UI_TESTING.md）。只會被「複製」，模擬器絕不寫回此檔；也刻意不讀真實的 data/assets.json。
SEED_DATA_PATH = PROJECT_ROOT / "tests" / "fixtures" / "ui_test_assets.json"


def seed_data_file(target: Path, source: Path = SEED_DATA_PATH, overwrite: bool = False) -> bool:
    """以 source 的副本建立模擬器資料檔。回傳是否有複製。

    預設只在 target 不存在時複製，避免重啟就把測試中寫入的資料洗掉。
    """
    if Path(source).resolve() == Path(target).resolve():
        raise ValueError("模擬器資料檔不可與來源為同一檔案，否則測試會改到真實資料")
    if not Path(source).is_file() or (Path(target).exists() and not overwrite):
        return False
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return True


def isolation_overrides(data_path: Path, channel_secret: str, allowed_user_ids: List[str]) -> Dict[str, Any]:
    """模擬器必須覆寫的設定。任何一項缺漏都可能讓測試流量碰到正式環境。"""
    return {
        "APP_ENV": "development",
        "DB_BACKEND": "json",
        "JSON_DATA_PATH": str(data_path),
        "ENABLE_SCHEDULER": False,
        "SERVICE_BASE_URL": "",
        "TASK_QUEUE_NAME": "",
        "LINE_CHANNEL_SECRET": channel_secret,
        "LINE_CHANNEL_ACCESS_TOKEN": SIMULATOR_TOKEN,
        "LINE_USER_ID": None,
        "ALLOWED_LINE_USER_IDS": list(allowed_user_ids),
    }


def sign_body(channel_secret: str, body: bytes) -> str:
    """依 LINE 規格計算 X-Line-Signature（HMAC-SHA256 後 Base64）。"""
    digest = hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


class _BlockedMessagingApi:
    """替換 line_bot_service 內的 MessagingApi：模擬器內任何真實送出都必須失敗。"""

    def __init__(self, *args: Any, **kwargs: Any):
        raise RuntimeError("[Simulator] 偵測到真實 LINE API 呼叫，已阻擋。請檢查攔截是否失效。")


class _RingLogHandler(logging.Handler):
    """保留最近的應用程式 Log 供畫面顯示。"""

    def __init__(self, maxlen: int = 500):
        super().__init__()
        self.lines: collections.deque = collections.deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(self.format(record))
        except Exception:
            self.handleError(record)


def _message_to_dict(message: Any) -> Dict[str, Any]:
    if isinstance(message, dict):
        return message
    if hasattr(message, "to_dict"):
        return message.to_dict()
    return message.model_dump(by_alias=True, exclude_none=True)


class LineSimulator:
    """在背景執行緒的事件迴圈上跑真實 app，並扮演 LINE 平台收發訊息。

    Streamlit 每次互動都會重跑腳本，但 app 內的背景工作（買入掃描等）需要
    一個長期存活的事件迴圈，因此迴圈放在獨立執行緒，UI 以
    `run_coroutine_threadsafe` 與之溝通。
    """

    def __init__(self, channel_secret: str, log_dir: Path = LOG_DIR):
        self.channel_secret = channel_secret
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._records: List[Dict[str, Any]] = []
        self._token_owner: Dict[str, str] = {}
        self._replied_tokens: set = set()
        self.log_handler = _RingLogHandler()

        # Streamlit 快取失效（改原始碼、Clear cache）時會重建實例。舊實例若不先關閉，
        # 會留下第二條事件迴圈、重複的 log handler，以及兩組倉儲同時寫同一個 JSON 檔。
        # 腳本每次重跑都是新的 __main__ 模組，模組全域變數留不住舊實例，
        # 因此登記在跨重跑不變的 app.state 上。
        from app.main import app
        previous = getattr(app.state, "line_simulator", None)
        if previous is not None:
            previous.close()

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="line-simulator-loop", daemon=True)
        self._thread.start()
        self._run(self._startup(), timeout=180)
        app.state.line_simulator = self

    # ------------------------------------------------------------------
    # 啟動與攔截
    # ------------------------------------------------------------------
    def _run(self, coro: Any, timeout: float) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    async def _startup(self) -> None:
        import httpx
        from app.core.logging import logger as app_logger
        from app.repositories import factory
        from app.services import line_bot_service as line_bot_module
        from app.main import app

        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        self.log_handler.setFormatter(fmt)
        file_handler = logging.FileHandler(
            self.log_dir / f"app-{datetime.now():%Y%m%d}.log", encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        app_logger.addHandler(self.log_handler)
        app_logger.addHandler(file_handler)
        self._log_handlers = [(app_logger, self.log_handler), (app_logger, file_handler)]

        # 倉儲為模組層級單例；清空才會依覆寫後的 DB_BACKEND / JSON_DATA_PATH 重建
        factory._config_repo = None
        factory._portfolio_repo = None
        factory._session_repo = None

        line_bot_module.MessagingApi = _BlockedMessagingApi

        self._lifespan = app.router.lifespan_context(app)
        await self._lifespan.__aenter__()
        self.app = app

        line_service = app.state.line_service
        line_service.channel_token = SIMULATOR_TOKEN
        line_service._send_reply_sync = self._capture_reply
        line_service._send_push_sync = self._capture_push
        self.line_service = line_service

        self._client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://line-simulator", timeout=120.0
        )

    def close(self) -> None:
        """結束 app lifespan 並停止事件迴圈（Streamlit 下隨程序結束，主要供測試使用）。"""
        if self._loop.is_closed():
            return

        async def _shutdown() -> None:
            await self._client.aclose()
            await self._lifespan.__aexit__(None, None, None)
            # 仍在跑的背景工作（如買入掃描）一併取消，避免關閉後還寫入資料檔
            pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        try:
            self._run(_shutdown(), timeout=30)
        finally:
            for lg, handler in self._log_handlers:
                lg.removeHandler(handler)
                handler.close()
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            if not self._thread.is_alive():
                self._loop.close()
            if getattr(self.app.state, "line_simulator", None) is self:
                self.app.state.line_simulator = None

    def _capture_reply(self, reply_token: str, messages: List[Any]) -> None:
        with self._lock:
            owner = self._token_owner.get(reply_token, "?")
            self._replied_tokens.add(reply_token)
        self._append({
            "kind": "reply",
            "user_id": owner,
            "reply_token": reply_token,
            "messages": [_message_to_dict(m) for m in messages],
        })

    def _capture_push(self, to_user_id: str, messages: List[Any]) -> None:
        self._append({
            "kind": "push",
            "user_id": to_user_id,
            "messages": [_message_to_dict(m) for m in messages],
        })

    def _append(self, record: Dict[str, Any]) -> Dict[str, Any]:
        record = {"id": uuid.uuid4().hex, "ts": datetime.now().isoformat(timespec="milliseconds"), **record}
        with self._lock:
            self._records.append(record)
            path = self.log_dir / f"simulator-{datetime.now():%Y%m%d}.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return record

    # ------------------------------------------------------------------
    # 模擬 LINE 平台送出 webhook
    # ------------------------------------------------------------------
    def _base_event(self, event_type: str, user_id: str) -> Dict[str, Any]:
        return {
            "type": event_type,
            "mode": "active",
            "timestamp": int(time.time() * 1000),
            "source": {"type": "user", "userId": user_id},
            "webhookEventId": f"01SIM{uuid.uuid4().hex[:21].upper()}",
            "deliveryContext": {"isRedelivery": False},
            "replyToken": uuid.uuid4().hex,
        }

    def send_text(self, user_id: str, text: str) -> Dict[str, Any]:
        event = self._base_event("message", user_id)
        event["message"] = {
            "type": "text",
            "id": str(random.randint(10**17, 10**18 - 1)),
            "quoteToken": uuid.uuid4().hex,
            "text": text,
        }
        return self._deliver(event, user_id, summary=text)

    def send_postback(self, user_id: str, data: str, label: str = "") -> Dict[str, Any]:
        event = self._base_event("postback", user_id)
        event["postback"] = {"data": data}
        return self._deliver(event, user_id, summary=f"🔘 {label or '按鈕'}  ({data})")

    def _deliver(self, event: Dict[str, Any], user_id: str, summary: str) -> Dict[str, Any]:
        token = event["replyToken"]
        with self._lock:
            self._token_owner[token] = user_id
        record = self._append({
            "kind": "user",
            "user_id": user_id,
            "event_type": event["type"] if event["type"] != "message" else event["message"]["type"],
            "summary": summary,
            "reply_token": token,
            "event": event,
        })
        status, elapsed, replied = self._run(self._post_and_wait(event), timeout=150)
        self._append({
            "kind": "system",
            "user_id": user_id,
            "reply_token": token,
            "http_status": status,
            "elapsed_ms": round(elapsed * 1000),
            "replied": replied,
            "for_record": record["id"],
        })
        return {"http_status": status, "elapsed_ms": round(elapsed * 1000), "replied": replied}

    async def _post_and_wait(self, event: Dict[str, Any], reply_wait: float = 2.0) -> Tuple[int, float, bool]:
        body = json.dumps(
            {"destination": "Usimulatorbot", "events": [event]}, ensure_ascii=False
        ).encode("utf-8")
        headers = {"Content-Type": "application/json", "X-Line-Signature": sign_body(self.channel_secret, body)}

        start = time.perf_counter()
        resp = await self._client.post("/callback", content=body, headers=headers)
        # reply 是以 create_task(to_thread(...)) 射後不理送出，回應返回時可能尚未送達
        token = event["replyToken"]
        deadline = time.perf_counter() + reply_wait
        while time.perf_counter() < deadline:
            with self._lock:
                if token in self._replied_tokens:
                    break
            await asyncio.sleep(0.05)
        with self._lock:
            replied = token in self._replied_tokens
        return resp.status_code, time.perf_counter() - start, replied

    # ------------------------------------------------------------------
    # 狀態查詢與操作
    # ------------------------------------------------------------------
    def records_for(self, user_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [r for r in self._records if r.get("user_id") == user_id]

    def clear_records(self) -> None:
        """只清畫面；JSONL 紀錄檔保留。"""
        with self._lock:
            self._records.clear()

    def is_authorized(self, user_id: str) -> bool:
        from app.core.config import settings
        return user_id in settings.ALLOWED_LINE_USER_IDS

    def set_authorized(self, user_id: str, allowed: bool) -> None:
        from app.core.config import settings
        ids = [u for u in settings.ALLOWED_LINE_USER_IDS if u != user_id]
        if allowed:
            ids.append(user_id)
        settings.ALLOWED_LINE_USER_IDS = ids

    def get_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        session = self._run(self.line_service.session_repo.get_session(user_id), timeout=10)
        return session.model_dump(mode="json") if session else None

    def clear_session(self, user_id: str) -> None:
        self._run(self.line_service.session_repo.clear_session(user_id), timeout=10)

    def get_portfolio(self) -> List[Dict[str, Any]]:
        items = self._run(self.line_service.portfolio_repo.get_all(status="holding"), timeout=10)
        return [i.model_dump(mode="json") for i in items]

    def reset_data(self, source: Path = SEED_DATA_PATH) -> bool:
        """以 source 的副本覆蓋模擬器資料檔（設定與持股一併重設）。"""
        from app.repositories.json_repo import JsonRepositoryBase

        async def _reset() -> bool:
            # 取得倉儲共用的鎖，避免與進行中的讀寫（例如背景賣出檢查）交錯
            async with JsonRepositoryBase._lock:
                return seed_data_file(Path(self.line_service.portfolio_repo.file_path), source, overwrite=True)

        return self._run(_reset(), timeout=10)


# ======================================================================
# Flex Message 解析（轉為可在 Streamlit 呈現的文字與按鈕）
# ======================================================================
def iter_flex(node: Any) -> Iterator[Tuple[str, Any]]:
    """深度優先走訪 Flex 結構，依序產出 ("text", (文字, 是否粗體)) 與 ("button", action)。"""
    if isinstance(node, list):
        for child in node:
            yield from iter_flex(child)
        return
    if not isinstance(node, dict):
        return

    node_type = node.get("type")
    if node_type == "text":
        text = node.get("text")
        if not text and node.get("contents"):
            text = "".join(span.get("text", "") for span in node["contents"])
        if text:
            yield "text", (text, node.get("weight") == "bold")
        return
    if node_type == "button" and node.get("action"):
        yield "button", node["action"]
        return
    if node_type == "separator":
        yield "separator", None
        return

    for key in ("header", "hero", "body", "footer", "contents"):
        if key in node:
            yield from iter_flex(node[key])
    if node.get("action") and node_type in ("box", "image"):
        yield "button", node["action"]


def latest_quick_replies(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for rec in reversed(records):
        if rec["kind"] not in ("reply", "push"):
            continue
        for msg in reversed(rec["messages"]):
            items = (msg.get("quickReply") or {}).get("items")
            if items:
                return [item["action"] for item in items if item.get("action")]
    return [{"type": "message", "label": label, "text": text} for label, text in FALLBACK_QUICK_REPLIES]


# ======================================================================
# Streamlit UI
# ======================================================================
def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="LINE Bot 模擬器", page_icon="💬", layout="wide")

    @st.cache_resource(show_spinner="正在啟動 app 與初始化服務（首次載入 NLI 模型約需 10 秒）...")
    def get_simulator() -> LineSimulator:
        # 覆寫必須在 app 任何模組被匯入前完成：Settings 會在匯入時讀取 .env
        # 並做 production 檢查；.env 與模型路徑皆為相對路徑，故一併切換工作目錄。
        os.chdir(PROJECT_ROOT)
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        channel_secret = secrets.token_hex(16)
        data_path = LOG_DIR / "simulator_assets.json"
        seed_data_file(data_path)
        overrides = isolation_overrides(data_path, channel_secret, [DEFAULT_USER_ID])
        for key, value in overrides.items():
            if isinstance(value, list):
                os.environ[key] = ",".join(value)
            else:
                os.environ[key] = "" if value is None else str(value)

        from app.core.config import settings
        for key, value in overrides.items():
            setattr(settings, key, value)
        return LineSimulator(channel_secret=channel_secret)

    sim = get_simulator()
    _inject_css(st)

    # ---------------- 側邊欄：模擬控制面板 ----------------
    with st.sidebar:
        st.header("🎛️ 模擬控制面板")
        user_id = st.text_input("User ID", value=DEFAULT_USER_ID, help="切換不同 user_id 可驗證白名單、session 與 Log 隔離").strip()
        if not user_id:
            st.warning("User ID 不可為空，已改用預設值。")
            user_id = DEFAULT_USER_ID

        authorized = st.toggle(
            "此 User ID 在白名單內",
            value=sim.is_authorized(user_id),
            key=f"auth-{user_id}",
            help="僅影響本模擬程序的 ALLOWED_LINE_USER_IDS，不改 .env",
        )
        if authorized != sim.is_authorized(user_id):
            sim.set_authorized(user_id, authorized)

        st.divider()
        st.subheader("🔎 目前狀態")
        session = sim.get_session(user_id)
        if session:
            st.caption("進行中的多輪對話 session")
            st.json(session, expanded=False)
            if st.button("清除此用戶 session", use_container_width=True):
                sim.clear_session(user_id)
                st.rerun()
        else:
            st.caption("無進行中的 session")
        holdings = sim.get_portfolio()
        with st.expander(f"模擬器持股庫存（{len(holdings)} 檔）"):
            st.json(holdings, expanded=True)
        if st.button(
            "↺ 重設為測試資料",
            use_container_width=True,
            help="以 tests/fixtures/ui_test_assets.json 覆蓋模擬器的持股與設定（同時清除警示冷卻）",
        ):
            if sim.reset_data():
                st.toast("已重設為測試資料", icon="✅")
            else:
                st.toast(f"找不到 {_display_path(SEED_DATA_PATH)}，未重設", icon="⚠️")
            st.rerun()

        st.divider()
        auto_refresh = st.toggle("自動刷新推播 (每 2 秒)", value=True, help="買入掃描等背景工作完成後會以 push 送達")
        if st.button("清空畫面對話紀錄", use_container_width=True):
            sim.clear_records()
            st.rerun()

        with st.expander("🛡️ 環境隔離狀態"):
            from app.core.config import settings
            st.markdown(
                f"- DB：`{settings.DB_BACKEND}` → `{_display_path(settings.JSON_DATA_PATH)}`\n"
                f"- Cloud Tasks：{'⚠️ 已啟用' if sim.line_service.task_dispatcher.enabled else '停用（本地執行）'}\n"
                f"- 排程：{'⚠️ 啟用' if settings.ENABLE_SCHEDULER else '停用'}\n"
                f"- LINE 送出：攔截（token 為假值，MessagingApi 已封鎖）\n"
                f"- SLM：{'啟用' if settings.SLM_ENABLED else '停用'}　Grounding：`{settings.GROUNDING_MODE}`\n"
                f"- Log：`{_display_path(LOG_DIR)}`"
            )

    # ---------------- 主畫面：對話 ----------------
    st.title("💬 LINE Bot 端到端模擬器")
    st.caption(f"目前身分：`{user_id}` {'✅ 已授權' if authorized else '⛔ 未授權'}　·　訊息經簽章後送入真實 `/callback`")

    @st.fragment(run_every=2 if auto_refresh else None)
    def chat_panel() -> None:
        records = sim.records_for(user_id)
        if not records:
            st.info("尚無對話。可在下方輸入文字，或點擊快捷按鈕開始測試（例如「說明」）。")
        for rec in records:
            _render_record(st, sim, user_id, rec)

        with st.container(key="quick_reply_bar"):
            actions = latest_quick_replies(records)
            cols = st.columns(len(actions))
            for col, action in zip(cols, actions):
                col.button(
                    action.get("label", "?"),
                    key=f"qr-{action.get('label')}",
                    use_container_width=True,
                    on_click=_perform_action,
                    args=(sim, user_id, action),
                )

    chat_panel()

    text = st.chat_input("輸入訊息，例如：買入 2330 1050 8 / 設定預算 120000 / 剛在1050買了8股台積電")
    if text:
        _safe_call(st, lambda: sim.send_text(user_id, text))
        st.rerun()

    with st.expander("🪵 應用程式 Log（最近 500 行，完整內容見 data_logs/）"):
        st.code("\n".join(list(sim.log_handler.lines)[-200:]) or "(尚無)", language="log")


def _display_path(path: Any) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _safe_call(st: Any, fn: Any) -> None:
    try:
        result = fn()
        if result and result.get("elapsed_ms", 0) > LINE_REPLY_DEADLINE_SECONDS * 1000:
            st.toast(f"⚠️ 回應耗時 {result['elapsed_ms']} ms，超過 LINE 的 5 秒期限", icon="⏱️")
    except Exception as e:
        st.error(f"送出失敗：{type(e).__name__}: {e}")


def _perform_action(sim: LineSimulator, user_id: str, action: Dict[str, Any]) -> None:
    import streamlit as st

    action_type = action.get("type")
    if action_type == "message":
        _safe_call(st, lambda: sim.send_text(user_id, action.get("text", "")))
    elif action_type == "postback":
        _safe_call(st, lambda: sim.send_postback(user_id, action.get("data", ""), action.get("label", "")))
    else:
        st.toast(f"模擬器不支援的按鈕動作：{action_type}")


def _render_record(st: Any, sim: LineSimulator, user_id: str, rec: Dict[str, Any]) -> None:
    kind = rec["kind"]
    ts = rec["ts"][11:19]

    if kind == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(rec["summary"])
            st.caption(f"{ts} · {rec['event_type']} event")
        return

    if kind == "system":
        status = rec["http_status"]
        note = f"HTTP {status} · {rec['elapsed_ms']} ms"
        if not rec["replied"]:
            note += " · ⚠️ Bot 未回覆此事件"
        if rec["elapsed_ms"] > LINE_REPLY_DEADLINE_SECONDS * 1000:
            note += " · ⏱️ 超過 LINE 5 秒期限"
        st.caption(note)
        return

    label = "↩️ reply" if kind == "reply" else "📨 push"
    for mi, msg in enumerate(rec["messages"]):
        with st.chat_message("assistant", avatar="🤖"):
            if msg.get("type") == "text":
                st.markdown(msg.get("text", "").replace("\n", "  \n"))
            elif msg.get("type") == "flex":
                _render_flex(st, sim, user_id, rec["id"], mi, msg)
            else:
                st.json(msg, expanded=False)
            st.caption(f"{ts} · {label}")


def _render_flex(st: Any, sim: LineSimulator, user_id: str, rid: str, mi: int, msg: Dict[str, Any]) -> None:
    contents = msg.get("contents") or {}
    bubbles = contents.get("contents", []) if contents.get("type") == "carousel" else [contents]
    st.markdown(f"**{msg.get('altText', 'Flex Message')}**")

    for bi, bubble in enumerate(bubbles):
        with st.container(border=True):
            sections: List[List[str]] = [[]]
            buttons: List[Dict[str, Any]] = []
            for kind, value in iter_flex(bubble):
                if kind == "text":
                    text, bold = value
                    sections[-1].append(f"**{text}**" if bold else text)
                elif kind == "separator":
                    sections.append([])
                elif kind == "button":
                    buttons.append(value)
            st.markdown("\n\n---\n\n".join("  \n".join(s) for s in sections if s))
            for ki, action in enumerate(buttons):
                st.button(
                    action.get("label", "按鈕"),
                    key=f"flex-{rid}-{mi}-{bi}-{ki}",
                    on_click=_perform_action,
                    args=(sim, user_id, action),
                )
    with st.expander("原始 Flex JSON"):
        st.json(contents, expanded=False)


def _inject_css(st: Any) -> None:
    st.markdown(
        """
        <style>
        /* 讓快捷按鈕列如 LINE Quick Reply 般懸浮於輸入框上方 */
        .st-key-quick_reply_bar {
            position: sticky;
            bottom: 0;
            z-index: 10;
            padding: 0.5rem 0;
            background: var(--background-color, rgba(255,255,255,0.95));
            border-top: 1px solid rgba(128,128,128,0.25);
        }
        .st-key-quick_reply_bar button { border-radius: 999px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
