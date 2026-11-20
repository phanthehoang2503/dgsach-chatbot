from psycopg import sql

from src.database import get_connection


def swap_table(
    staging_table: str = "documents_staging", target_table: str = "documents"
):
    target_id = sql.Identifier(target_table)
    staging_id = sql.Identifier(staging_table)
    old_id = sql.Identifier(f"{target_table}_old")

    with get_connection(
        autocommit=False, register_pgvector=False, role="ingest"
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
            cur.execute(
                sql.SQL("ALTER INDEX IF EXISTS {} RENAME TO {};").format(
                    sql.Identifier(f"{staging_table}_pkey"),
                    sql.Identifier(f"{target_table}_pkey"),
                )
            )
            cur.execute(
                sql.SQL("ALTER SEQUENCE IF EXISTS {} RENAME TO {};").format(
                    sql.Identifier(f"{staging_table}_id_seq"),
                    sql.Identifier(f"{target_table}_id_seq"),
                )
            )
            cur.execute(
                sql.SQL("DROP INDEX IF EXISTS {};").format(
                    sql.Identifier(f"idx_{staging_table}_embedding_hnsw")
                )
            )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {} ON {} "
                    "USING hnsw (embedding vector_cosine_ops);"
                ).format(
                    sql.Identifier(f"idx_{target_table}_embedding_hnsw"),
                    target_id,
                )
            )
        conn.commit()
