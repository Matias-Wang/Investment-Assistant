import json
import os
import asyncio
from google.cloud import firestore
from app.core.config import settings
from app.models.domain import SystemConfig, PortfolioItem


async def seed_data():
    project_id = settings.GCP_PROJECT_ID
    if not project_id:
        print("Error: GCP_PROJECT_ID is not configured in .env")
        return

    print(f"Connecting to Firestore for project: {project_id}...")
    db = firestore.AsyncClient(project=project_id, database=settings.FIRESTORE_DATABASE)

    data_file = settings.JSON_DATA_PATH
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found.")
        return

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. 寫入系統設定
    cfg_raw = data.get("system_config", {})
    cfg = SystemConfig(**cfg_raw)
    cfg_ref = db.collection(settings.FIRESTORE_COLLECTION_CONFIG).document("default")
    await cfg_ref.set(cfg.model_dump())
    print(f"Successfully seeded system_config to Firestore ({settings.FIRESTORE_COLLECTION_CONFIG}/default)")

    # 2. 寫入持股庫存
    portfolio_raw = data.get("portfolio", [])
    col_ref = db.collection(settings.FIRESTORE_COLLECTION_PORTFOLIO)
    for item_data in portfolio_raw:
        item = PortfolioItem(**item_data)
        doc_ref = col_ref.document(item.ticker.strip().upper())
        await doc_ref.set(item.model_dump())
        print(f"Successfully seeded portfolio item: {item.ticker} ({item.name})")

    print("All initial data from assets.json successfully migrated to Firestore!")


if __name__ == "__main__":
    asyncio.run(seed_data())
