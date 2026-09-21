"""產生 Cloud Run 部署用的非機密環境變數檔 (.env.yaml)。

機密金鑰（GEMINI_API_KEY、LINE_CHANNEL_SECRET、LINE_CHANNEL_ACCESS_TOKEN、
ADMIN_API_KEY）一律排除，禁止落地於明文檔案；部署時改由 `--set-env-vars` 或
Secret Manager 提供。

雲端與本機的兩處關鍵差異
------------------------
1. **DB_BACKEND=firestore**：Cloud Run 的可寫檔案系統是記憶體模擬的暫時空間，
   容器停止即消失，`data/assets.json` 無法保存任何資料。
2. **ENABLE_SCHEDULER=false**：容器內若自行排程，就必須 24 小時開機
   （min-instances=1），CPU 常駐費用約每月 US$56。改由 Cloud Scheduler 從外部
   呼叫 `/api/trigger/*`，容器平常關機、只在工作執行的數十秒內計費。

執行
----
    PYTHONPATH=. python scripts/gen_env_yaml.py
"""

import sys

from app.core.config import settings

# 機密欄位清單：僅用於提示，本腳本永不輸出這些值
SECRET_KEYS = [
    "GEMINI_API_KEY",
    "LINE_CHANNEL_SECRET",
    "LINE_CHANNEL_ACCESS_TOKEN",
    "ADMIN_API_KEY",
]


def build_yaml() -> str:
    lines = [
        "# 由 scripts/gen_env_yaml.py 產生，請勿手動編輯。",
        "# 僅含非機密設定；金鑰另以 --set-env-vars 或 Secret Manager 提供。",
        "",
        "# --- 雲端必要差異 ---",
        '# Cloud Run 檔案系統為暫時性，必須用 Firestore 才能保存資料',
        'DB_BACKEND: "firestore"',
        f'GCP_PROJECT_ID: "{settings.GCP_PROJECT_ID or ""}"',
        f'FIRESTORE_DATABASE: "{settings.FIRESTORE_DATABASE}"',
        "",
        "# 容器內不自行排程：改由 Cloud Scheduler 呼叫 /api/trigger/*，",
        "# 使容器得以縮到零，不產生 CPU 常駐費用",
        'ENABLE_SCHEDULER: "false"',
        "",
        "# --- 應用設定 ---",
        'APP_ENV: "production"',
        f'GEMINI_MODEL: "{settings.GEMINI_MODEL}"',
        f'LINE_USER_ID: "{settings.LINE_USER_ID or ""}"',
        f'HTTP_TIMEOUT_SECONDS: "{settings.HTTP_TIMEOUT_SECONDS}"',
        f'MAX_RETRY_ATTEMPTS: "{settings.MAX_RETRY_ATTEMPTS}"',
        "",
        "# --- 買入掃描 ---",
        f'BUY_SCAN_CONCURRENCY: "{settings.BUY_SCAN_CONCURRENCY}"',
        f'NEWS_CACHE_TTL_SECONDS: "{settings.NEWS_CACHE_TTL_SECONDS}"',
        f'NEWS_FILTER_MODE: "{settings.NEWS_FILTER_MODE}"',
        f'BUY_ANALYSIS_MAX_OUTPUT_TOKENS: "{settings.BUY_ANALYSIS_MAX_OUTPUT_TOKENS}"',
        f'MARKET_REPORT_MAX_OUTPUT_TOKENS: "{settings.MARKET_REPORT_MAX_OUTPUT_TOKENS}"',
        "",
        "# --- 幻覺驗證層 ---",
        "# 模型隨映像打包於 models/nli/；缺檔時 fail-open，不影響服務",
        f'GROUNDING_MODE: "{settings.GROUNDING_MODE}"',
        f'GROUNDING_ENTAILMENT_THRESHOLD: "{settings.GROUNDING_ENTAILMENT_THRESHOLD}"',
        f'GROUNDING_INTRA_OP_THREADS: "{settings.GROUNDING_INTRA_OP_THREADS}"',
        "",
        "# --- 以下僅在 ENABLE_SCHEDULER=true 時才有作用，保留供本機除錯用 ---",
        f'SELL_CHECK_INTERVAL_MINUTES: "{settings.SELL_CHECK_INTERVAL_MINUTES}"',
        f'BUY_CHECK_CRON_HOUR: "{settings.BUY_CHECK_CRON_HOUR}"',
        f'BUY_CHECK_CRON_MINUTE: "{settings.BUY_CHECK_CRON_MINUTE}"',
        f'MARKET_REPORT_CRON_HOUR: "{settings.MARKET_REPORT_CRON_HOUR}"',
        f'MARKET_REPORT_CRON_MINUTE: "{settings.MARKET_REPORT_CRON_MINUTE}"',
        f'MARKET_REPORT_CRON_DAYS: "{settings.MARKET_REPORT_CRON_DAYS}"',
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    content = build_yaml()
    with open(".env.yaml", "w", encoding="utf-8") as f:
        f.write(content)

    print("已產生 .env.yaml（不含任何機密值）")
    print()
    if not settings.GCP_PROJECT_ID:
        print("⚠️  GCP_PROJECT_ID 為空，請先在 .env 設定後重新產生。")
        print()
    print("APP_ENV=production 會啟用啟動檢查，下列四項缺一容器就無法啟動，")
    print("必須於部署指令以 --set-env-vars 或 Secret Manager 提供：")
    for k in SECRET_KEYS:
        print(f"    {k}")
    print("    ALLOWED_LINE_USER_IDS   （fail-closed：留空則所有使用者都被拒絕）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
