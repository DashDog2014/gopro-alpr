import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

# Load .env from repo root (two levels up from src/db/db.py)
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

def env_required(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name} (check {ENV_PATH})")
    return val

def get_connect():
    host = env_required("DB_HOST")
    port = int(os.getenv("DB_PORT", "55432"))  # default to tunnel port
    dbname = env_required("DB_NAME")
    user = env_required("DB_USER")
    password = env_required("DB_PASSWORD")

    return psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
    )

if __name__ == "__main__":
    print("Loaded .env from:", ENV_PATH)
    print("DB_HOST:", os.getenv("DB_HOST"))
    print("DB_PORT:", os.getenv("DB_PORT"))

    conn = get_connect()
    cur = conn.cursor()
    cur.execute("SELECT 1;")
    print("DB OK:", cur.fetchone())
    cur.close()
    conn.close()