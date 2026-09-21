"""耗時工作派送器的測試。

重點在於兩件事不能出錯：
1. **本機未配置 Cloud Tasks 時**必須退回本地背景執行，否則開發環境的指令會無聲失效。
2. **派送失敗時**必須降級而非讓工作消失——使用者按了按鈕就該有結果。
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.config import settings
from app.services.task_dispatcher import JOB_PATHS, TaskDispatcher


def _configured() -> TaskDispatcher:
    return TaskDispatcher(
        service_base_url="https://svc.example.run.app",
        queue_name="investment-jobs",
        queue_location="asia-east1",
        project_id="proj-123",
        api_key="secret-key",
    )


# ---------------------------------------------------------------- 啟用判定

def test_not_enabled_when_unconfigured():
    d = TaskDispatcher(service_base_url="", queue_name="", queue_location="", project_id="")
    assert d.enabled is False


def test_enabled_when_fully_configured():
    assert _configured().enabled is True


@pytest.mark.parametrize("missing", ["service_base_url", "queue_name", "queue_location", "project_id"])
def test_any_missing_field_disables(missing):
    kwargs = dict(
        service_base_url="https://svc.example.run.app",
        queue_name="q",
        queue_location="asia-east1",
        project_id="p",
    )
    kwargs[missing] = ""
    assert TaskDispatcher(**kwargs).enabled is False


def test_trailing_slash_stripped():
    d = TaskDispatcher(
        service_base_url="https://svc.example.run.app/", queue_name="q",
        queue_location="asia-east1", project_id="p",
    )
    assert d.service_base_url == "https://svc.example.run.app"


# ---------------------------------------------------------------- URL 組裝

def test_url_includes_push_and_user():
    d = _configured()
    url = d._build_url("buy-scan", "U123")
    assert url.startswith("https://svc.example.run.app/api/trigger/buy-scan?")
    assert "push=true" in url
    assert "user_id=U123" in url


def test_url_omits_user_when_absent():
    url = _configured()._build_url("market-report", None)
    assert "user_id" not in url
    assert "push=true" in url


def test_all_jobs_have_paths():
    for job in ["buy-scan", "sell-check", "market-report"]:
        assert job in JOB_PATHS
        assert JOB_PATHS[job].startswith("/api/trigger/")


# ---------------------------------------------------------------- 派送行為

@pytest.mark.asyncio
async def test_unconfigured_falls_back_to_local():
    """本機開發：沒有 Cloud Tasks 也必須把工作做掉。"""
    d = TaskDispatcher(service_base_url="", queue_name="", queue_location="", project_id="")
    ran = asyncio.Event()

    async def work():
        ran.set()

    used_queue = await d.dispatch("buy-scan", user_id="U1", fallback=work)
    await asyncio.wait_for(ran.wait(), timeout=1.0)
    assert used_queue is False
    assert ran.is_set()


@pytest.mark.asyncio
async def test_configured_enqueues_without_running_fallback():
    d = _configured()
    ran = False

    async def work():
        nonlocal ran
        ran = True

    with patch.object(d, "_enqueue") as enqueue:
        used_queue = await d.dispatch("sell-check", user_id="U2", fallback=work)

    assert used_queue is True
    enqueue.assert_called_once_with("sell-check", "U2")
    await asyncio.sleep(0)
    assert ran is False, "已交給佇列就不該同時在本地跑一次，否則會重複執行"


@pytest.mark.asyncio
async def test_enqueue_failure_falls_back():
    """派送失敗時不可讓使用者的指令無聲消失。"""
    d = _configured()
    ran = asyncio.Event()

    async def work():
        ran.set()

    with patch.object(d, "_enqueue", side_effect=RuntimeError("queue gone")):
        used_queue = await d.dispatch("market-report", fallback=work)

    await asyncio.wait_for(ran.wait(), timeout=1.0)
    assert used_queue is False
    assert ran.is_set()


@pytest.mark.asyncio
async def test_unknown_job_raises():
    with pytest.raises(ValueError):
        await _configured().dispatch("not-a-job")


@pytest.mark.asyncio
async def test_missing_fallback_does_not_crash():
    d = TaskDispatcher(service_base_url="", queue_name="", queue_location="", project_id="")
    assert await d.dispatch("buy-scan", fallback=None) is False


# ---------------------------------------------------------------- Cloud Tasks 呼叫內容

def test_enqueue_builds_expected_task():
    d = _configured()
    fake_client = MagicMock()
    fake_client.queue_path.return_value = "projects/proj-123/locations/asia-east1/queues/investment-jobs"
    fake_tasks_v2 = MagicMock()
    fake_tasks_v2.CloudTasksClient.return_value = fake_client
    fake_tasks_v2.HttpMethod.POST = "POST"

    with patch.dict("sys.modules", {"google.cloud.tasks_v2": fake_tasks_v2}):
        with patch("google.cloud.tasks_v2", fake_tasks_v2, create=True):
            d._enqueue("buy-scan", "U9")

    fake_client.queue_path.assert_called_once_with("proj-123", "asia-east1", "investment-jobs")
    req = fake_client.create_task.call_args.kwargs["request"]
    http = req["task"]["http_request"]
    assert http["url"].endswith("/api/trigger/buy-scan?push=true&user_id=U9")
    # 端點受 verify_admin 保護，未帶金鑰會被 401 擋掉
    assert http["headers"]["X-API-Key"] == "secret-key"


# ---------------------------------------------------------------- 與 LINE 服務整合

@pytest.mark.asyncio
async def test_line_service_uses_dispatcher_for_heavy_commands():
    """「買入建議」必須走派送器，而不是在回應後於背景執行。"""
    from app.models.domain import SystemConfig
    from app.services.line_bot_service import LineBotService

    config_repo, portfolio_repo, session_repo = AsyncMock(), AsyncMock(), AsyncMock()
    config_repo.get_config.return_value = SystemConfig()
    session_repo.get_session.return_value = None

    dispatcher = MagicMock()
    dispatcher.dispatch = AsyncMock(return_value=True)

    service = LineBotService(
        config_repo, portfolio_repo, session_repo, task_dispatcher=dispatcher
    )
    service.reply_message = MagicMock()

    event = MagicMock()
    event.reply_token = "tok"
    event.source.user_id = "U123456"
    event.message.text = "買入建議"

    with patch.object(settings, "ALLOWED_LINE_USER_IDS", ["U123456"]):
        await service._handle_text_message(event, {"buy_engine": MagicMock()})

    dispatcher.dispatch.assert_awaited_once()
    assert dispatcher.dispatch.await_args.args[0] == "buy-scan"
    assert dispatcher.dispatch.await_args.kwargs["user_id"] == "U123456"
