"""把本機 data/assets.json 的資料搬移到 Firestore。

用途：`DB_BACKEND=json` 的資料存在本機檔案，而 Cloud Run 的檔案系統是暫時性的
（容器停止即消失），因此部署前必須先把既有持股與設定搬到 Firestore。

執行
----
    # 只預覽，不寫入任何資料（建議先跑這個確認內容正確）
    PYTHONPATH=. python scripts/seed_firestore.py --dry-run

    # 實際寫入
    PYTHONPATH=. python scripts/seed_firestore.py

注意
----
* 以 `set()` 寫入，會**覆寫** Firestore 上同 ID 的既有文件。重跑是幂等的，
  但若雲端資料已比本機新，執行本腳本會把雲端資料蓋回舊值——故預設先要求確認。
* 序列化一律使用 `model_dump(mode="json")`，與 `firestore_repo.py` 保持一致。
  不可用 `model_dump()`：`shares`、`average_cost`、`trade_budget_per_order` 為
  `Decimal` 型別，而 **Firestore 不支援 Decimal**，會在寫入時拋錯。
"""

import argparse
import asyncio
import json
import os
import sys

from app.core.config import settings
from app.models.domain import PortfolioItem, SystemConfig


def load_local_data() -> dict:
    data_file = settings.JSON_DATA_PATH
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"找不到本機資料檔：{data_file}")
    with open(data_file, "r", encoding="utf-8") as f:
        return json.load(f)


def build_payloads(data: dict) -> tuple[dict, list[tuple[str, dict]]]:
    """把本機資料轉為可寫入 Firestore 的結構，順便驗證模型。"""
    cfg = SystemConfig(**data.get("system_config", {}))
    config_payload = cfg.model_dump(mode="json")

    items: list[tuple[str, dict]] = []
    for raw in data.get("portfolio", []):
        item = PortfolioItem(**raw)
        doc_id = item.ticker.strip().upper()
        items.append((doc_id, item.model_dump(mode="json")))
    return config_payload, items


def preview(config_payload: dict, items: list[tuple[str, dict]]) -> None:
    print("=" * 62)
    print(f"目標專案：{settings.GCP_PROJECT_ID or '(未設定)'}")
    print(f"資料庫　：{settings.FIRESTORE_DATABASE}")
    print("=" * 62)
    print()
    print(f"[1] {settings.FIRESTORE_COLLECTION_CONFIG}/default")
    for k, v in config_payload.items():
        print(f"      {k:26} = {v}")
    print()
    print(f"[2] {settings.FIRESTORE_COLLECTION_PORTFOLIO}/ — 共 {len(items)} 筆")
    for doc_id, payload in items:
        print(
            f"      {doc_id:8} {payload.get('name', ''):8}"
            f" 股數={payload.get('shares')}"
            f" 平均成本={payload.get('average_cost')}"
            f" 狀態={payload.get('status')}"
        )
    print()


async def write_to_firestore(config_payload: dict, items: list[tuple[str, dict]]) -> None:
    from google.cloud import firestore

    db = firestore.AsyncClient(
        project=settings.GCP_PROJECT_ID, database=settings.FIRESTORE_DATABASE
    )

    cfg_ref = db.collection(settings.FIRESTORE_COLLECTION_CONFIG).document("default")
    await cfg_ref.set(config_payload)
    print(f"✅ 已寫入 {settings.FIRESTORE_COLLECTION_CONFIG}/default")

    col_ref = db.collection(settings.FIRESTORE_COLLECTION_PORTFOLIO)
    for doc_id, payload in items:
        await col_ref.document(doc_id).set(payload)
        print(f"✅ 已寫入 {settings.FIRESTORE_COLLECTION_PORTFOLIO}/{doc_id}")

    print()
    print("搬移完成。請將部署環境的 DB_BACKEND 設為 firestore。")


async def main_async(args) -> int:
    try:
        data = load_local_data()
        config_payload, items = build_payloads(data)
    except Exception as e:
        print(f"錯誤：{type(e).__name__}: {e}")
        return 1

    preview(config_payload, items)

    if args.dry_run:
        print("（--dry-run：以上為預覽，未寫入任何資料）")
        return 0

    if not settings.GCP_PROJECT_ID:
        print("錯誤：GCP_PROJECT_ID 未設定，無法連線 Firestore。")
        return 1

    if not args.yes:
        print("⚠️  這會以上述內容覆寫 Firestore 上同 ID 的既有文件。")
        print("    若雲端資料比本機新，執行後雲端資料會被蓋回舊值。")
        print("    確認無誤請加上 --yes 重新執行，或先用 --dry-run 檢視。")
        return 1

    try:
        await write_to_firestore(config_payload, items)
    except Exception as e:
        print(f"寫入失敗：{type(e).__name__}: {e}")
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="將本機 assets.json 搬移至 Firestore")
    p.add_argument("--dry-run", action="store_true", help="只顯示將寫入的內容，不實際寫入")
    p.add_argument("--yes", action="store_true", help="確認執行寫入（會覆寫同 ID 文件）")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
