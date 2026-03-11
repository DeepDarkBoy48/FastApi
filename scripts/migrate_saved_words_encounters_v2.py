#!/usr/bin/env python3
"""
Migrate saved_words to Aggregation v2:
- one row per word
- data.schemaVersion = 2
- data.encounters[] keeps multi-source contexts
- rebuild daily_notes.word_count as per-note unique word count

This script is idempotent and runs in one transaction.
"""

import json
import hashlib
from datetime import datetime
from collections import defaultdict

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


def normalize_word(word: str) -> str:
    return (word or "").strip().lower()


def normalize_context(context: str) -> str:
    return " ".join((context or "").strip().split()).lower()


def datetime_to_str(value) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if value is None:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


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


def build_source_key(url, reading_id, video_id, note_id) -> str:
    if reading_id is not None:
        return f"reading:{reading_id}"
    if video_id is not None:
        return f"video:{video_id}"
    if url:
        return f"url:{url.strip()}"
    if note_id is not None:
        return f"note:{note_id}"
    return "manual"


def build_encounter_key(word: str, source_key: str, context: str) -> str:
    raw = f"{normalize_word(word)}|{source_key}|{normalize_context(context)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]

def normalize_other_forms(raw_forms) -> list[dict]:
    if not isinstance(raw_forms, list):
        return []
    normalized = []
    seen = set()
    for item in raw_forms:
        if isinstance(item, str):
            form = item.strip()
            part_of_speech = ""
            meaning = ""
        elif isinstance(item, dict):
            form = str(item.get("form") or item.get("text") or "").strip()
            part_of_speech = str(item.get("partOfSpeech") or item.get("part_of_speech") or "").strip()
            meaning = str(item.get("meaning") or "").strip()
        else:
            continue
        if not form:
            continue
        form_key = form.lower()
        if form_key in seen:
            continue
        seen.add(form_key)
        normalized.append({
            "form": form,
            "partOfSpeech": part_of_speech,
            "meaning": meaning,
        })
    return normalized


def build_lookup_payload(word: str, data: dict) -> dict:
    safe = data if isinstance(data, dict) else {}
    other = safe.get("otherMeanings")
    if not isinstance(other, list):
        other = []
    other_forms = normalize_other_forms(safe.get("otherForms"))
    return {
        "word": safe.get("word") or word,
        "contextMeaning": safe.get("contextMeaning") or safe.get("m") or "",
        "partOfSpeech": safe.get("partOfSpeech") or safe.get("p") or "",
        "grammarRole": safe.get("grammarRole") or "",
        "explanation": safe.get("explanation") or "",
        "baseForm": safe.get("baseForm") or "",
        "otherForms": other_forms,
        "otherMeanings": other,
    }


def strip_compat_fields(payload: dict) -> dict:
    cleaned = dict(payload)
    for field in (
        "context", "url", "note_id", "reading_id", "video_id",
        "word", "contextMeaning", "partOfSpeech", "grammarRole", "explanation", "baseForm", "otherForms", "otherMeanings",
        "m", "p"
    ):
        cleaned.pop(field, None)
    return cleaned


def ensure_v2_payload(word: str, raw_payload: dict, row: dict) -> dict:
    parsed_payload = parse_json_obj(raw_payload)
    payload = strip_compat_fields(parsed_payload)
    root_lookup = build_lookup_payload(word, parsed_payload)
    encounters = []

    raw_encounters = payload.get("encounters")
    if isinstance(raw_encounters, list):
        for enc in raw_encounters:
            if not isinstance(enc, dict):
                continue
            context = enc.get("context") or row.get("context") or ""
            url = enc.get("url") or row.get("url")
            note_id = enc.get("note_id") if enc.get("note_id") is not None else row.get("note_id")
            reading_id = enc.get("reading_id") if enc.get("reading_id") is not None else row.get("reading_id")
            video_id = enc.get("video_id") if enc.get("video_id") is not None else row.get("video_id")
            source_key = build_source_key(url, reading_id, video_id, note_id)
            key = enc.get("key") or build_encounter_key(word, source_key, context)
            lookup_src = enc.get("lookup") if isinstance(enc.get("lookup"), dict) else root_lookup
            encounters.append({
                "key": key,
                "context": context,
                "url": url,
                "note_id": note_id,
                "reading_id": reading_id,
                "video_id": video_id,
                "created_at": datetime_to_str(enc.get("created_at") or row.get("created_at")),
                "lookup": build_lookup_payload(word, lookup_src),
            })

    if not encounters:
        context = row.get("context") or parsed_payload.get("context") or ""
        url = row.get("url") or parsed_payload.get("url")
        note_id = row.get("note_id")
        reading_id = row.get("reading_id")
        video_id = row.get("video_id")
        source_key = build_source_key(url, reading_id, video_id, note_id)
        encounters = [{
            "key": build_encounter_key(word, source_key, context),
            "context": context,
            "url": url,
            "note_id": note_id,
            "reading_id": reading_id,
            "video_id": video_id,
            "created_at": datetime_to_str(row.get("created_at")),
            "lookup": root_lookup,
        }]

    encounters = sorted(encounters, key=lambda e: e.get("created_at") or "", reverse=True)

    merged = strip_compat_fields(payload)
    merged["schemaVersion"] = 2
    merged["encounters"] = encounters
    return merged


def choose_primary(rows):
    def score(r):
        created = r.get("created_at") or datetime.min
        created_ts = created.timestamp() if isinstance(created, datetime) else 0
        return (-int(r.get("reps") or 0), -created_ts, int(r.get("id") or 0))

    return sorted(rows, key=score)[0]


def merge_group(rows):
    primary = choose_primary(rows)
    merged_word = primary["word"]
    seen = set()
    merged_encounters = []

    ordered_rows = sorted(
        rows,
        key=lambda r: (
            -(r.get("created_at").timestamp() if isinstance(r.get("created_at"), datetime) else 0),
            int(r.get("id") or 0),
        ),
    )

    for row in ordered_rows:
        payload = ensure_v2_payload(row["word"], row.get("data"), row)
        for enc in payload.get("encounters") or []:
            if not isinstance(enc, dict):
                continue
            key = enc.get("key")
            if not key or key in seen:
                continue
            seen.add(key)
            merged_encounters.append(enc)

    merged_encounters = sorted(merged_encounters, key=lambda e: e.get("created_at") or "", reverse=True)
    latest = merged_encounters[0]

    merged_payload = {
        "schemaVersion": 2,
        "encounters": merged_encounters,
    }
    merged_payload.update(build_lookup_payload(merged_word, latest.get("lookup")))

    return primary, merged_payload, latest


def extract_note_ids(payload: dict, fallback_note_id=None):
    note_ids = set()
    for enc in payload.get("encounters") or []:
        if isinstance(enc, dict) and isinstance(enc.get("note_id"), int):
            note_ids.add(enc["note_id"])
    if isinstance(fallback_note_id, int):
        note_ids.add(fallback_note_id)
    return note_ids


def main():
    conn = pymysql.connect(**DB_CONFIG)
    stats = {
        "total_rows_before": 0,
        "total_rows_after": 0,
        "merged_groups": 0,
        "deleted_rows": 0,
        "updated_rows": 0,
        "errors": 0,
    }

    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM saved_words ORDER BY LOWER(word), created_at DESC, id ASC")
            rows = cursor.fetchall()
            stats["total_rows_before"] = len(rows)

            groups = defaultdict(list)
            for row in rows:
                groups[normalize_word(row.get("word") or "")].append(row)

            for _, group_rows in groups.items():
                try:
                    primary, merged_payload, latest = merge_group(group_rows)
                    if len(group_rows) > 1:
                        stats["merged_groups"] += 1

                    cursor.execute(
                        """
                        UPDATE saved_words
                        SET word = %s, context = %s, url = %s, data = %s, note_id = %s, reading_id = %s, video_id = %s
                        WHERE id = %s
                        """,
                        (
                            primary["word"],
                            latest.get("context") or primary.get("context") or "",
                            latest.get("url") or primary.get("url"),
                            json.dumps(merged_payload, ensure_ascii=False),
                            latest.get("note_id") if latest.get("note_id") is not None else primary.get("note_id"),
                            latest.get("reading_id") if latest.get("reading_id") is not None else primary.get("reading_id"),
                            latest.get("video_id") if latest.get("video_id") is not None else primary.get("video_id"),
                            primary["id"],
                        ),
                    )
                    stats["updated_rows"] += 1

                    extra_ids = [r["id"] for r in group_rows if r["id"] != primary["id"]]
                    if extra_ids:
                        placeholders = ", ".join(["%s"] * len(extra_ids))
                        cursor.execute(f"DELETE FROM saved_words WHERE id IN ({placeholders})", tuple(extra_ids))
                        stats["deleted_rows"] += len(extra_ids)
                except Exception as e:
                    stats["errors"] += 1
                    print(f"[WARN] merge group failed: {e}")

            # Rebuild daily note counts by unique words linked through encounters
            cursor.execute("SELECT * FROM saved_words")
            merged_rows = cursor.fetchall()
            note_word_sets = defaultdict(set)
            for row in merged_rows:
                payload = ensure_v2_payload(row["word"], row.get("data"), row)
                note_ids = extract_note_ids(payload, row.get("note_id"))
                norm_word = normalize_word(row.get("word") or "")
                for nid in note_ids:
                    note_word_sets[nid].add(norm_word)

            cursor.execute("SELECT id FROM daily_notes")
            note_rows = cursor.fetchall()
            for note in note_rows:
                nid = note["id"]
                count = len(note_word_sets.get(nid, set()))
                cursor.execute("UPDATE daily_notes SET word_count = %s WHERE id = %s", (count, nid))

            conn.commit()

            cursor.execute("SELECT COUNT(*) AS cnt FROM saved_words")
            stats["total_rows_after"] = cursor.fetchone()["cnt"]

        print("=== Migration Complete ===")
        for k, v in stats.items():
            print(f"{k}: {v}")
    except Exception as e:
        conn.rollback()
        print("Migration failed; transaction rolled back.")
        print(f"Error: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
