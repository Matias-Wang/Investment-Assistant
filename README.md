# LLM 投資策略與自動追蹤系統 (Investment Assistant)

本專案為一套結合 **Python 量化篩選**、**Google Gemini 2.5 Flash-Lite 文本分析 (RAG)**、**LINE Bot 互動介面** 與 **雲端/本機資料庫** 的智能投資輔助系統。

> ⚠️ **重要原則**：系統**不執行自動下單交易**。所有買入與賣出決策均透過 LINE Bot 推播精美的 Flex Message 建議卡片，並由使用者點擊按鈕手動確認成交後，才更新庫存狀態表。

---

## 系統架構概覽

系統嚴格遵循 **Controller (API 路由) - Service (商業邏輯) - Repository (資料庫存取)** 分層架構：

```
investment_assistant/
├── app/
│   ├── main.py                     # FastAPI 進入點、生命週期管理與排程啟動
│   ├── core/
│   │   ├── config.py               # Pydantic-Settings 環境變數載入 (.env)
│   │   ├── logging.py              # 結構化日誌記錄
│   │   └── retry.py                # 外部 API 逾時、重試與指數退避裝飾器
│   ├── models/
│   │   ├── domain.py               # 領域實體 (SystemConfig, PortfolioItem, SellAlert 等)
│   │   ├── llm.py                  # Gemini 結構化輸出模式 (BuyRecommendation)
│   │   └── line.py                 # LINE 對話會話與 Postback 酬載模型
│   ├── repositories/
│   │   ├── base.py                 # 資料庫抽象介面 (Config, Portfolio, Session)
│   │   ├── json_repo.py            # 本地 JSON 實作 (直接讀寫 data/assets.json)
│   │   ├── firestore_repo.py       # Google Cloud Firestore 雲端資料庫實作
│   │   └── factory.py              # 倉儲工廠模式 (依環境變數無縫切換)
│   ├── services/
│   │   ├── market_service.py       # yfinance 報價與 20MA 計算 (含快取與重試)
│   │   ├── news_service.py         # 近 3 日財經新聞檢索 (Google News RSS + yfinance)
│   │   ├── llm_service.py          # Gemini 2.5 Flash-Lite 結構化推論與防幻覺機制
│   │   ├── buy_engine.py           # 買入策略引擎 (20MA + 預算零股 + 新聞 RAG + LLM)
│   │   ├── sell_engine.py          # 賣出監控引擎 (Rule Engine: 10%~20% 停利, -50% 停損)
│   │   ├── line_bot_service.py     # LINE Webhook 事件路由、指令解析與狀態機
│   │   └── scheduler_service.py    # APScheduler 背景週期掃描
│   ├── controllers/
│   │   ├── line_controller.py      # LINE Webhook 端點 (/callback)
│   │   └── api_controller.py       # REST API 與手動觸發端點 (/api/*, /health)
│   └── utils/
│       └── flex_templates.py       # LINE Flex Message 模板建構器
├── data/
│   └── assets.json                 # 本地庫存與動態設定資料
├── tests/
│   ├── test_rule_engine.py         # 賣出規則引擎單元測試
│   ├── test_quant_filter.py        # 買入量化過濾與預算計算單元測試
│   ├── test_command_parser.py      # LINE 指令文字解析單元測試
│   └── test_json_repo.py           # JSON 倉儲讀寫與買賣更新單元測試
├── Dockerfile                      # Cloud Run / Docker 映像檔建置檔
├── requirements.txt                # 專案依賴清單
└── .env.example                    # 環境變數設定範本
```

---

## 四大核心模組與業務邏輯

### 1. 資料庫與狀態管理 (State Management)
* **雙重資料庫支援**：
  * **JSON 後端 (`DB_BACKEND=json`)**：開箱即用，直接讀寫專案目錄下已建立之 `data/assets.json`。
  * **Firestore 後端 (`DB_BACKEND=firestore`)**：部署至 GCP Cloud Run 時切換為 Google Cloud Firestore。
* **動態參數管理**：支援單次投資預算 (`trade_budget_per_order`，預設 100,000)、停利區間 (`[0.10, 0.20]`)、停損點 (`-0.50`)。
* **持股狀態表**：支援持股代號、名稱、股數、平均成本、建倉日期與持股狀態 (`holding` / `sold`)。加碼時自動計算加權平均成本。

### 2. 賣出監控引擎 (Rule Engine)
* **零 LLM 依賴**：完全採用純規則判斷，降低推論成本與延遲。
* **未實現損益率公式**：
  $$\text{損益率} = \frac{\text{現價} - \text{成本}}{\text{成本}}$$
* **觸發條件**：
  * 當損益率進入 `[10%, 20%]` 區間：觸發 **停利警示 (`TAKE_PROFIT`)**。
  * 當損益率低於或等於 `-50%`：觸發 **停損警示 (`STOP_LOSS`)**。
* **推播內容**：生成 Flex Message 卡片，標註損益率與損益金額，並附帶 `[手動已賣出]` 按鈕。

### 3. 買入策略引擎 (Quantitative + RAG + LLM)
* **量化技術面篩選**：檢查收盤價是否站上 20 日均線（$\text{Close} > \text{SMA}_{20}$）。
* **動態預算與零股計算**：以動態預算（如 10 萬元）除以最新現價，計算出建議買入之零股股數。若單股現價大於總預算則自動排除。
* **新聞 RAG 文本檢索**：自動串接檢索標的近 3 日內中文與全球財經重大新聞，提取摘要作為上下文。
* **Gemini 2.5 Flash-Lite 結構化推論**：
  * 強制設定 `response_mime_type="application/json"` 與 `response_schema`。
  * **嚴格防幻覺限制**：提示詞內強制聲明「僅能依據提供之新聞文本進行推論，嚴格禁止使用預訓練資料與猜測外部資訊。若新聞資訊不足，必須判定為 HOLD 或 AVOID」。
* **推播內容**：若 LLM 評估結果為 `BUY`，即轉換為 Flex Message，附帶情緒摘要、支持理由、風險提示與 `[手動已買入]` 按鈕。

### 4. LINE Bot 互動介面與指令
* **指令解析**：
  * `查詢庫存` 或 `庫存`：回傳現有持股明細、現價、成本與總未實現損益卡片。
  * `市場報告` 或 `盤勢`：立即產出 AI 彙整之大盤、總經與持股受惠分析報告。
  * `查詢設定` 或 `設定`：回傳當前動態策略參數。
  * `設定預算 120000`：即時動態修改單次買入預算。
  * `設定停利 0.1 0.2` 或 `設定停利 15%`：即時修改停利區間。
  * `設定停損 -0.4` 或 `設定停損 40%`：即時修改停損門檻。
  * `買入建議` / `賣出檢查`：手動立即觸發掃描引擎。
  * `說明` / `help`：列出所有操作指南。
* **多輪手動交易確認對話流**：
  * 點擊卡片中的 `[手動已買入]` 或 `[手動已賣出]` 按鈕。
  * 系統自動開啟暫存對話狀態：
    * 直接回覆 `確認`：以建議價格與股數寫入資料庫。
    * 或回覆自訂數值：例如 `買入 2330 1050 10` 或 `賣出 2330 1080 5`。
    * 回覆 `取消`：取消該筆交易記錄。

### 5. 每日晨間市場快報引擎 (Market Report Engine + APScheduler)
* **定時全自動推播**：每天早上 **08:00**（台北時間 Asia/Taipei）自動執行。
* **多維度資訊彙整**：
  * 抓取四大指標行情：台股加權 (`^TWII`)、標普 500 (`^GSPC`)、費城半導體 (`^SOX`)、台積電 (`2330.TW`)。
  * 檢索近 3 日關鍵總經與半導體財經新聞。
  * 結合用戶當前庫存真實持股組合。
* **AI 結構化洞察**：透過 Gemini 3.5 提煉市場氛圍、盤勢重點、3 大頭條、個人持股受惠程度與操盤叮嚀。
* **多元觸發支援**：
  * 容器內 APScheduler 定時器（每天 08:00 自動推播）。
  * 外部排程（GCP Cloud Scheduler）呼叫 `POST /api/trigger/market-report?push=true`。
  * 本地/伺服器 CLI 腳本 `python scripts/send_market_report.py`。

---

## 快速開始

### 1. 安裝環境與依賴
確保已安裝 Python 3.11+：
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 設定環境變數
複製 `.env.example` 並建立 `.env`：
```bash
cp .env.example .env
```
編輯 `.env` 填入您的金鑰：
```ini
DB_BACKEND=json
JSON_DATA_PATH=data/assets.json

GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite

LINE_CHANNEL_SECRET=your_line_secret
LINE_CHANNEL_ACCESS_TOKEN=your_line_access_token
LINE_USER_ID=your_line_user_id
```

### 3. 執行單元測試
系統具備完整的自動化單元測試：
```bash
pytest tests
```

### 4. 啟動本機伺服器
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```
啟動後可存取：
* Swagger 互動文件：`http://localhost:8080/docs`
* 健康檢查：`http://localhost:8080/api/health`
* 庫存評價 API：`http://localhost:8080/api/portfolio`
* 手動觸發買入掃描：`POST http://localhost:8080/api/trigger/buy-scan`
* 手動觸發賣出檢查：`POST http://localhost:8080/api/trigger/sell-check`
* 手動/排程觸發市場報告推播：`POST http://localhost:8080/api/trigger/market-report?push=true`
* LINE Webhook 端點：`POST http://localhost:8080/callback`

### 5. 命令列直接執行市場報告 (CLI)
```bash
# 產出報告並直接推播至 LINE
python scripts/send_market_report.py

# 僅於終端機測試產出，不發送 LINE (Dry-run)
python scripts/send_market_report.py --dry-run
```

---

## 雲端部署 (Docker & GCP Cloud Run)

### 1. 本地 Docker 建置與測試
```bash
docker build -t investment-assistant:latest .
docker run -p 8080:8080 --env-file .env investment-assistant:latest
```

### 2. 前置資源（首次部署或換專案時才需要）

```bash
gcloud config set project [PROJECT-ID]
gcloud billing projects link [PROJECT-ID] --billing-account=[ACCOUNT-ID]
gcloud services enable run.googleapis.com firestore.googleapis.com \
  cloudscheduler.googleapis.com cloudtasks.googleapis.com cloudbuild.googleapis.com \
  secretmanager.googleapis.com artifactregistry.googleapis.com
gcloud firestore databases create --location=asia-east1
gcloud tasks queues create investment-jobs --location=asia-east1

for S in gemini-api-key line-channel-secret line-channel-access-token admin-api-key; do
  gcloud secrets create "$S" --replication-policy=automatic
  gcloud secrets add-iam-policy-binding "$S" \
    --member="serviceAccount:[PROJECT-NUMBER]-compute@developer.gserviceaccount.com" \
    --role=roles/secretmanager.secretAccessor
done
python scripts/push_secrets.py         # 先預覽（只顯示指紋，不顯示金鑰值）
python scripts/push_secrets.py --yes   # 確認無誤後才真正寫入
```

| 漏掉哪一項 | 症狀 |
| :--- | :--- |
| 連結帳單帳戶 | 服務無法啟用，`FAILED_PRECONDITION: Billing account ... is not found` |
| 四個 Secret | 容器起不來（`APP_ENV=production` 會檢查四把金鑰，缺一即拒絕啟動） |
| Secret 讀取權 | revision 建立失敗，`Permission denied on secret ... for Revision service account` |
| Cloud Tasks 佇列 | 耗時工作退回本地背景執行，在 Cloud Run 上會被 CPU 節流而跑不完 |

灌完 Secret 後以 `python scripts/check_secret_sync.py` 確認本機與雲端一致（只顯示指紋，不顯示金鑰值）。

### 3. 部署至 GCP Cloud Run

```bash
# 1. 產生非機密環境變數檔，再提交建置
python scripts/gen_env_yaml.py
gcloud builds submit --tag gcr.io/[PROJECT-ID]/investment-assistant

# 2. 部署（金鑰一律走 Secret Manager，不得用 --set-env-vars 傳遞）
gcloud run deploy investment-assistant \
  --image gcr.io/[PROJECT-ID]/investment-assistant \
  --platform managed --region asia-east1 --allow-unauthenticated \
  --env-vars-file .env.yaml \
  --set-secrets "GEMINI_API_KEY=gemini-api-key:latest,LINE_CHANNEL_SECRET=line-channel-secret:latest,LINE_CHANNEL_ACCESS_TOKEN=line-channel-access-token:latest,ADMIN_API_KEY=admin-api-key:latest" \
  --memory 2Gi --cpu 1 --timeout 600 --min-instances 0 --max-instances 2

# 3. 把服務網址填回去（最容易漏掉）
URL=$(gcloud run services describe investment-assistant --region asia-east1 --format "value(status.url)")
gcloud run services update investment-assistant --region asia-east1 --update-env-vars "SERVICE_BASE_URL=$URL"
```

> PowerShell 使用者請把行尾的 `\` 換成反引號、`$(...)` 換成直接指派。

`ALLOWED_LINE_USER_IDS`、`TASK_QUEUE_NAME`、`TASK_QUEUE_LOCATION` 三項需**併入 `.env.yaml`**
——`--env-vars-file` 與 `--set-env-vars` 互斥，同時給會直接報錯。

最後將 `$URL/callback` 填入 LINE Developers Console 的 Webhook URL 並開啟 Use Webhook。

### 4. 四個不做就會出事、但症狀不明顯的點

| 設定 | 為什麼不能省 |
| :--- | :--- |
| `--memory 2Gi` | 買入掃描載入 NLI 模型時峰值超過 1GB。實測 1Gi 會 `Memory limit of 1024 MiB exceeded with 1091 MiB used`，容器在掃描中被殺 |
| `SERVICE_BASE_URL` | 漏掉則耗時工作退回本地背景執行，在 Cloud Run 上被 CPU 節流。**不會報錯**，只是 LINE 回了「掃描中」之後永遠沒有下文 |
| `ALLOWED_LINE_USER_IDS` | fail-closed 設計：留空則所有使用者都被拒絕，包含你自己 |
| `.gcloudignore` | 缺少時 `gcloud builds submit` 會改用 `.gitignore` 當忽略清單，而後者刻意排除 `models/` 下的權重，導致幻覺驗證層靜默 fail-open。詳見該檔註解 |

金鑰**不要**用 `--set-env-vars` 傳遞：那會讓明文留在 revision 設定與 shell 歷史中。
一律經由 Secret Manager 以 `--set-secrets` 掛載。
