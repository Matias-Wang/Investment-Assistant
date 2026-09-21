"""比對本機 .env 與 Secret Manager 的憑證是否同步。

為什麼需要
----------
本機 `.env` 只影響本機執行；部署後的 Cloud Run 讀的是 Secret Manager。
更新其中一邊而忘了另一邊，症狀往往是「本機正常、雲端卻回 401 或 API 錯誤」，
而錯誤訊息不會告訴你是金鑰不同步造成的。

本腳本**只比對 SHA-256 指紋的前 12 位，絕不輸出任何憑證值**，可安全貼上回報。
另外會檢查尾端是否夾帶空白或換行——以 `echo` 導入 Secret Manager 是常見錯誤，
會讓金鑰多一個看不見的 `\\n`，之後比對必定失敗且極難察覺。

執行
----
    $env:PYTHONPATH = "."
    .\\.venv\\Scripts\\python.exe scripts\\check_secret_sync.py
"""

import hashlib
import subprocess
import sys

# 環境變數名稱 -> Secret Manager 上的 secret 名稱
# 本專案的 secret 採小寫連字號命名，與環境變數名稱不同，容易搞錯。
SECRET_MAP = {
    "GEMINI_API_KEY": "gemini-api-key",
    "LINE_CHANNEL_SECRET": "line-channel-secret",
    "LINE_CHANNEL_ACCESS_TOKEN": "line-channel-access-token",
    "ADMIN_API_KEY": "admin-api-key",
}

ENV_FILE = ".env"


def fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:12] if value else "(空)"


def read_env(name: str) -> str:
    try:
        with open(ENV_FILE, encoding="utf-8") as f:
            for line in f:
                if line.startswith(f"{name}="):
                    return line.split("=", 1)[1].rstrip("\n")
    except FileNotFoundError:
        pass
    return ""


def read_secret(secret: str) -> tuple[str, str]:
    """回傳 (值, 錯誤訊息)；成功時錯誤訊息為空字串。"""
    proc = subprocess.run(
        ["gcloud", "secrets", "versions", "access", "latest", "--secret", secret],
        capture_output=True,
        text=True,
        shell=True,
    )
    if proc.returncode != 0:
        return "", proc.stderr.strip().splitlines()[0] if proc.stderr else "讀取失敗"
    return proc.stdout, ""


def describe_whitespace(raw: str) -> str:
    """指出尾端是否夾帶不該有的空白字元。"""
    if raw != raw.strip():
        trailing = raw[len(raw.rstrip()):]
        return f"⚠️ 尾端有 {trailing!r}"
    return ""


def main() -> int:
    print(f"{'環境變數':<28}{'本機':>14}{'雲端':>14}  狀態")
    print("-" * 78)

    drift, problems = [], []
    for env_name, secret_name in SECRET_MAP.items():
        local_raw = read_env(env_name)
        local = local_raw.strip()
        cloud_raw, err = read_secret(secret_name)
        cloud = cloud_raw.strip()

        if err:
            status = f"❌ {err[:40]}"
            problems.append(env_name)
        elif not local:
            status = "本機未設定"
            problems.append(env_name)
        elif local == cloud:
            status = "✅ 相同"
        else:
            status = "⚠️  不同，雲端需更新"
            drift.append((env_name, secret_name))

        note = describe_whitespace(cloud_raw) or describe_whitespace(local_raw)
        print(
            f"{env_name:<28}{fingerprint(local):>14}"
            f"{(fingerprint(cloud) if not err else '-'):>14}  {status} {note}"
        )

    print()
    if not drift and not problems:
        print("兩邊完全同步。")
        return 0

    if drift:
        print("以下憑證需要更新到 Secret Manager（PowerShell）：")
        print()
        print("    function Set-Secret($name, $value) {")
        print('        $tmp = Join-Path $env:TEMP "secret.tmp"')
        print("        # 不帶 BOM、不帶結尾換行；用 echo 導入會多一個看不見的 \\n")
        print("        [System.IO.File]::WriteAllText($tmp, $value, "
              "[System.Text.UTF8Encoding]::new($false))")
        print("        gcloud secrets versions add $name --data-file=$tmp")
        print("        Remove-Item $tmp -Force")
        print("    }")
        print()
        for env_name, secret_name in drift:
            print(f'    Set-Secret "{secret_name}" "<{env_name} 的新值>"')
        print()
        print("更新後需重新部署（或等下次容器冷啟動）才會生效：")
        print("    gcloud run services update investment-assistant --region asia-east1")
    return 1


if __name__ == "__main__":
    sys.exit(main())
