#!/usr/bin/env python3
"""
Drop legacy saved_words.context/url columns after encounters migration.

Safety checks:
- Every saved_words row must already contain at least one data.encounters entry.
- The script is idempotent and can be re-run safely.
"""

import json

import pymysql
from pymysql.cursors import DictCursor


DB_CONFIG = {
    "host": "47.79.43.73",
    "user": "root",
    "password": "aZ9s8f7G3j2kL5mN",
    "database": "smashenglish",
    "charset": "utf8mb4",
    "cursorclass": DictCursor,
    "autocommit": False,
}


def parse_json_obj(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def has_column(cursor, column_name: str) -> bool:
    cursor.execute("SHOW COLUMNS FROM saved_words LIKE %s", (column_name,))
    return cursor.fetchone() is not None


def main():
    connection = pymysql.connect(**DB_CONFIG)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT id, word, data FROM saved_words ORDER BY id ASC")
            rows = cursor.fetchall()

            missing_encounters = []
            for row in rows:
                payload = parse_json_obj(row.get("data"))
                encounters = payload.get("encounters")
                if not isinstance(encounters, list) or not encounters:
                    missing_encounters.append(f"{row['id']}:{row['word']}")

            if missing_encounters:
                raise RuntimeError(
                    "Refusing to drop legacy columns because these rows still lack data.encounters: "
                    + ", ".join(missing_encounters[:20])
                )

            dropped = []
            if has_column(cursor, "context"):
                cursor.execute("ALTER TABLE saved_words DROP COLUMN context")
                dropped.append("context")

            if has_column(cursor, "url"):
                cursor.execute("ALTER TABLE saved_words DROP COLUMN url")
                dropped.append("url")

            connection.commit()
            if dropped:
                print(f"Dropped legacy columns: {', '.join(dropped)}")
            else:
                print("No legacy columns found. Nothing to do.")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
