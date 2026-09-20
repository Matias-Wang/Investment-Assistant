from app.core.config import settings

def main():
    """
    產生非機密環境變數設定檔 (.env.yaml) 供 Cloud Run 部署使用。
    機密金鑰 (GEMINI_API_KEY, LINE_CHANNEL_SECRET, LINE_CHANNEL_ACCESS_TOKEN, ADMIN_API_KEY)
    已依安全規範完全排除，禁止落地於明文檔案中。
    """
    env_yaml_content = (
        f'DB_BACKEND: "firestore"\n'
        f'GCP_PROJECT_ID: "{settings.GCP_PROJECT_ID}"\n'
        f'GEMINI_MODEL: "{settings.GEMINI_MODEL}"\n'
        f'LINE_USER_ID: "{settings.LINE_USER_ID}"\n'
        f'HTTP_TIMEOUT_SECONDS: "15.0"\n'
        f'MAX_RETRY_ATTEMPTS: "3"\n'
        f'ENABLE_SCHEDULER: "true"\n'
        f'SELL_CHECK_INTERVAL_MINUTES: "30"\n'
        f'BUY_CHECK_CRON_HOUR: "13"\n'
        f'BUY_CHECK_CRON_MINUTE: "0"\n'
        f'MARKET_REPORT_CRON_HOUR: "{settings.MARKET_REPORT_CRON_HOUR}"\n'
        f'MARKET_REPORT_CRON_MINUTE: "{settings.MARKET_REPORT_CRON_MINUTE}"\n'
        f'MARKET_REPORT_CRON_DAYS: "{settings.MARKET_REPORT_CRON_DAYS}"\n'
    )
    with open(".env.yaml", "w", encoding="utf-8") as f:
        f.write(env_yaml_content)
    print("Generated .env.yaml (non-secrets only) successfully!")

if __name__ == "__main__":
    main()
