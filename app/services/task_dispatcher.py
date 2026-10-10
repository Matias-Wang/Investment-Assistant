"""把耗時工作交給佇列執行，而不是在回應後於背景執行。

為什麼需要這個
--------------
LINE 要求 webhook 在 **5 秒內**回傳 200，但買入掃描約 90 秒、市場快報約 25 秒、
賣出檢查約 10 秒，都不可能在期限內做完，因此必須非同步處理。

而原本的做法（`asyncio.create_task` 與 FastAPI `BackgroundTasks`）在
Cloud Run「僅於處理請求時分配 CPU」的模式下不可靠：回應送出後 CPU 會被節流，
背景工作可能永遠跑不完。若改為「CPU 常駐」則需 24 小時開機，每月約 US$56。

解法是讓佇列（Cloud Tasks）發一個**新的 HTTP 請求**回到自己的
`/api/trigger/*` 端點。新請求會讓 Cloud Run 重新分配 CPU，且該端點是同步完成的，
因此工作一定跑得完。Cloud Tasks 每月前 100 萬次操作免費，本專案用量遠低於此。

本機開發
--------
未設定 `SERVICE_BASE_URL` 或 `TASK_QUEUE_NAME` 時，自動退回 `asyncio.create_task`。
本機沒有 CPU 節流問題，行為與原本相同。
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional
from urllib.parse import urlencode

from app.core.config import settings
from app.core.logging import logger

# 可派送的工作名稱 -> 對應的 API 路徑
JOB_PATHS = {
    "buy-scan": "/api/trigger/buy-scan",
    "sell-check": "/api/trigger/sell-check",
    "market-report": "/api/trigger/market-report",
    "industry-report": "/api/trigger/industry-report",
}


class TaskDispatcher:
    """將耗時工作派送至 Cloud Tasks；未配置時退回本地背景執行。"""

    def __init__(
        self,
        service_base_url: Optional[str] = None,
        queue_name: Optional[str] = None,
        queue_location: Optional[str] = None,
        project_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        # 一律以 `is not None` 判斷而非 `or`：後者無法區分「未提供」與「刻意傳空」，
        # 會讓明確停用某一項的呼叫端被設定檔的值悄悄覆蓋。
        def pick(given: Optional[str], fallback: Optional[str]) -> str:
            return (given if given is not None else fallback) or ""

        self.service_base_url = pick(service_base_url, settings.SERVICE_BASE_URL).rstrip("/")
        self.queue_name = pick(queue_name, settings.TASK_QUEUE_NAME)
        self.queue_location = pick(queue_location, settings.TASK_QUEUE_LOCATION)
        self.project_id = pick(project_id, settings.GCP_PROJECT_ID)
        self.api_key = pick(api_key, settings.ADMIN_API_KEY)

    @property
    def enabled(self) -> bool:
        """是否具備派送至 Cloud Tasks 的完整設定。"""
        return bool(
            self.service_base_url and self.queue_name and self.queue_location and self.project_id
        )

    def _build_url(self, job: str, user_id: Optional[str]) -> str:
        params = {"push": "true"}
        if user_id:
            params["user_id"] = user_id
        return f"{self.service_base_url}{JOB_PATHS[job]}?{urlencode(params)}"

    async def dispatch(
        self,
        job: str,
        user_id: Optional[str] = None,
        fallback: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> bool:
        """派送工作。

        回傳 True 代表已成功交給 Cloud Tasks；False 代表改用本地背景執行
        （本機開發，或派送失敗後的降級）。

        `fallback` 為未配置 Cloud Tasks 時要在本地執行的協程工廠。
        """
        if job not in JOB_PATHS:
            raise ValueError(f"未知的工作名稱：{job}")

        if not self.enabled:
            logger.info(f"[TaskDispatcher] 未配置 Cloud Tasks，改以本地背景執行：{job}")
            self._run_locally(fallback)
            return False

        try:
            await asyncio.to_thread(self._enqueue, job, user_id)
            logger.info(f"[TaskDispatcher] 已派送至 Cloud Tasks：{job}")
            return True
        except Exception as e:
            # 派送失敗不可讓使用者的指令無聲消失，降級為本地執行；
            # 本地執行在雲端可能被節流，但總比完全不做好。
            logger.warning(f"[TaskDispatcher] 派送失敗（{type(e).__name__}: {e}），降級為本地背景執行")
            self._run_locally(fallback)
            return False

    def _enqueue(self, job: str, user_id: Optional[str]) -> None:
        """同步呼叫 Cloud Tasks API（由 to_thread 包裝，避免阻塞事件迴圈）。"""
        from google.cloud import tasks_v2

        client = tasks_v2.CloudTasksClient()
        parent = client.queue_path(self.project_id, self.queue_location, self.queue_name)
        task = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": self._build_url(job, user_id),
                "headers": {"X-API-Key": self.api_key},
            }
        }
        client.create_task(request={"parent": parent, "task": task})

    @staticmethod
    def _run_locally(fallback: Optional[Callable[[], Awaitable[None]]]) -> None:
        if fallback is None:
            logger.warning("[TaskDispatcher] 無 fallback 可執行，工作被略過")
            return
        try:
            asyncio.get_running_loop().create_task(fallback())
        except RuntimeError:
            # 不在事件迴圈中（例如同步測試），直接執行完畢
            asyncio.run(fallback())
