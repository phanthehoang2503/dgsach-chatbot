from psycopg import sql
import os
import psycopg
from dotenv import load_dotenv

load_dotenv()


def swap_table(
    staging_table: str = "documents_staging", target_table: str = "documents"
):
    target_id = sql.Identifier(target_table)
    staging_id = sql.Identifier(staging_table)
    old_id = sql.Identifier(f"{target_table}_old")

    with psycopg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5434"),
        dbname=os.getenv("DB_NAME", "vectordb"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASS", "postgres"),
        autocommit=False,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {};").format(old_id))
            cur.execute(
                sql.SQL("ALTER TABLE IF EXISTS {} RENAME TO {};").format(
                    target_id, old_id
                )
            )
            cur.execute(
                sql.SQL("ALTER TABLE {} RENAME TO {};").format(staging_id, target_id)
            )
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {};").format(old_id))
        conn.commit()
