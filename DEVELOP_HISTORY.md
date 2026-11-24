# 研發歷史摘要

只記錄重要研發項目；細節請見 git log 與 README。

---

## 2026-11-24　市場快報拆分為美股晨報與台股盤後快報

* **背景**：本機與 GitHub (PR #1) 同時各自改寫市場快報，分別做了「盤前／盤後版別 + 資料時點標示」與「依市場拆分 us／tw」，合併時衝突。
* **結果**：
  * 以市場拆分為主軸：08:00 美股晨報（週二至六，`market=us`）、14:00 台股盤後快報（週一至五，`market=tw`）。
  * 保留資料時點標示：卡片的「資料基準」與產出時間由程式填入，LLM 只產出內容 (`MarketReportDraft`)。
  * 新增 `strict` 模式：排程或 API 明確指定 `market=tw` 時，台股休市回 200 skipped、交易日抓不到行情回 503 讓 Cloud Scheduler 重試；LINE 手動查詢不檢查休市。
  * 移除 `edition=pre|post` 參數與 `POST_MARKET_REPORT_CRON_*` 設定。
* **後續**：BACKLOG B-003（測試缺口）、B-004（交易日曆失敗時的行為）。
