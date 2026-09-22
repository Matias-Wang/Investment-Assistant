"""把本機 .env 的憑證灌入 Secret Manager。

為什麼不用 `echo ... | gcloud secrets versions add`
--------------------------------------------------
PowerShell 的 `echo` / `Write-Output` 會自動附加換行字元，使金鑰尾端多一個
看不見的 `\\n`。之後比對永遠失敗，而錯誤訊息只會說「未授權」或「金鑰無效」，
根本猜不到是換行造成的。本腳本以二進位方式寫入暫存檔（不帶 BOM、不帶結尾換行），
寫完即刪除。

**畫面上不會顯示任何憑證值**，只顯示 SHA-256 指紋前 12 位，可安全貼上回報。

執行
----
    $env:PYTHONPATH = "."
    $env:PYTHONIOENCODING = "utf-8"
    .\\.venv\\Scripts\\python.exe scripts\\push_secrets.py            # 需確認
    .\\.venv\\Scripts\\python.exe scripts\\push_secrets.py --yes      # 直接執行
"""

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile

# 環境變數名稱 -> Secret Manager 的 secret 名稱（小寫連字號式）
SECRET_MAP = {
    "GEMINI_API_KEY": "gemini-api-key",
    "LINE_CHANNEL_SECRET": "line-channel-secret",
    "LINE_CHANNEL_ACCESS_TOKEN": "line-channel-access-token",
    "ADMIN_API_KEY": "admin-api-key",
}

ENV_FILE = ".env"


def fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:12] if value else "(空)"


def read_env() -> dict:
    values = {}
    with open(ENV_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    return values


def current_project() -> str:
    r = subprocess.run(
        ["gcloud", "config", "get-value", "project"],
        capture_output=True, text=True, shell=True,
    )
    return r.stdout.strip()


def add_version(secret: str, value: str) -> tuple[bool, str]:
    """以暫存檔寫入新版本，避免換行污染。"""
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(value.encode("utf-8"))   # 不帶 BOM、不帶結尾換行
        r = subprocess.run(
            ["gcloud", "secrets", "versions", "add", secret, "--data-file", path],
            capture_output=True, text=True, shell=True,
        )
        if r.returncode != 0:
            err = (r.stderr or "").strip().splitlines()
            return False, (err[0] if err else "未知錯誤")
        return True, ""
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def main() -> int:
    p = argparse.ArgumentParser(description="把 .env 的憑證灌入 Secret Manager")
    p.add_argument("--yes", action="store_true", help="略過確認直接執行")
    args = p.parse_args()

    try:
        env = read_env()
    except FileNotFoundError:
        print(f"錯誤：找不到 {ENV_FILE}")
        return 1

    project = current_project()
    print(f"目標專案：{project or '(未設定)'}")
    print()

    plan = []
    for env_name, secret in SECRET_MAP.items():
        value = env.get(env_name, "")
        if not value:
            print(f"  ⚠️  {env_name:28} .env 中未設定，略過")
            continue
        if value != value.strip():
            print(f"  ⚠️  {env_name:28} 值的前後有空白，將自動修剪")
            value = value.strip()
        plan.append((env_name, secret, value))
        print(f"  {env_name:28} -> {secret:28} 指紋 {fingerprint(value)}")

    if not plan:
        print("\n沒有任何可寫入的憑證。")
        return 1

    print()
    if not args.yes:
        print(f"以上 {len(plan)} 項將寫入專案 {project} 的 Secret Manager（各建立一個新版本）。")
        print("確認無誤請加上 --yes 重新執行。")
        return 1

    print("開始寫入...")
    failed = []
    for env_name, secret, value in plan:
        ok, err = add_version(secret, value)
        if ok:
            print(f"  ✅ {secret}")
        else:
            print(f"  ❌ {secret}：{err[:120]}")
            failed.append(secret)

    print()
    if failed:
        print(f"有 {len(failed)} 項失敗。若錯誤為 NOT_FOUND，請先建立 secret：")
        for s in failed:
            print(f"    gcloud secrets create {s} --replication-policy=automatic")
        return 1

    print("全部完成。建議接著執行 scripts/check_secret_sync.py 確認同步。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
