import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg2

ENV_Path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_Path)

def get_connect():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT, 5432")),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        )
if __name__ == "__main__":
    conn = get_connect()
    cur = conn.cursor()
    cur.execute("SELECT 1;")
    print("DB OK:", cur.fetchone())
    cur.close()
    conn.close()