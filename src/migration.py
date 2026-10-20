import os
import psycopg
from dotenv import load_dotenv

load_dotenv()


def swap_table(
    staging_table: str = "documents_staging", target_table: str = "documents"
):
    with psycopg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5434"),
        dbname=os.getenv("DB_NAME", "vectordb"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASS", "postgres"),
        autocommit=False,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                  DROP TABLE IF EXISTS {target_table}_old;
                  ALTER TABLE IF EXISTS {target_table} RENAME TO {target_table}_old;
                  ALTER TABLE {staging_table} RENAME TO {target_table};
                  DROP TABLE IF EXISTS {target_table}_old;
                """
            )
            conn.commit()
