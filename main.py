from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated, Optional, List
import os
import gemini
import pymysql
from pymysql.cursors import DictCursor
import json
from datetime import date, datetime, timedelta, timezone
import math
import hashlib
from fsrs import Scheduler, Card, Rating, State
from dotenv import load_dotenv
from schemas import (
    AnalysisRequest, AnalysisResult,
    LookupRequest, DictionaryResult,
    WritingRequest, WritingResult,
    ChatRequest, ChatResponse,
    QuickLookupRequest, QuickLookupResult,
    RapidLookupRequest, RapidLookupResult,
    TranslateRequest, AdvancedTranslateRequest, TranslateResult,
    SavedWord, SavedWordsResponse,
    SavedWordsExportResponse, SavedWordsImportRequest, SavedWordsImportResponse,
    DailyNote, DailyNotesResponse, NoteDetailResponse,
    VideoNotebook, VideoNotebookCreate, VideoNotebookListResponse, VideoNotebookUpdate,
    ReadingNotebook, ReadingNotebookCreate, ReadingNotebookListResponse, ReadingNotebookUpdate,
    TodayReviewResponse, ReviewArticle, FSRSFeedbackRequest,
    ReviewPromptResponse, ReviewImportRequest,
    SavedWordUpdate, SavedWordCreate, BatchDeleteRequest,
    FeatureLLMConfigResponse
)


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DB_CONFIG = {
    # 'host': '47.79.43.73',
    'host': 'mysql-container',
    'user': 'root',
    'password': 'aZ9s8f7G3j2kL5mN',
    'database': 'smashenglish',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

def get_db_connection():
    return pymysql.connect(**DB_CONFIG)

def normalize_word(word: str) -> str:
    return (word or "").strip().lower()


def normalize_context(context: str) -> str:
    return " ".join((context or "").strip().split()).lower()


def datetime_to_str(value) -> str:
    if isinstance(value, datetime):
        return value.strftime('%Y-%m-%d %H:%M:%S')
    if value is None:
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return str(value)


def parse_datetime_value(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        normalized = raw.replace('T', ' ')
        if normalized.endswith('Z'):
            normalized = normalized[:-1]
        for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d'):
            try:
                return datetime.strptime(normalized, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(raw.replace('Z', '+00:00')).replace(tzinfo=None)
        except Exception:
            return None
    return None


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


def build_source_key(url: Optional[str], reading_id: Optional[int], video_id: Optional[int], note_id: Optional[int]) -> str:
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
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:24]


def build_lookup_payload(word: str, data: Optional[dict]) -> dict:
    safe_data = data if isinstance(data, dict) else {}
    other_meanings = safe_data.get('otherMeanings')
    if not isinstance(other_meanings, list):
        other_meanings = []
    other_forms = safe_data.get('otherForms')
    if not isinstance(other_forms, list):
        other_forms = []
    return {
        "word": safe_data.get('word') or word,
        "contextMeaning": safe_data.get('contextMeaning') or safe_data.get('m') or "",
        "partOfSpeech": safe_data.get('partOfSpeech') or safe_data.get('p') or "",
        "grammarRole": safe_data.get('grammarRole') or "",
        "explanation": safe_data.get('explanation') or "",
        "baseForm": safe_data.get('baseForm') or "",
        "otherForms": other_forms,
        "otherMeanings": other_meanings
    }


def sanitize_import_payload(word: str, payload: dict, fallback_note_id: int) -> dict:
    sanitized = ensure_v2_payload(word, payload, {
        "context": "",
        "url": None,
        "note_id": fallback_note_id,
        "reading_id": None,
        "video_id": None,
        "created_at": datetime.now()
    })
    encounters = sanitized.get('encounters') if isinstance(sanitized.get('encounters'), list) else []
    normalized_encounters = []
    for enc in encounters:
        if not isinstance(enc, dict):
            continue
        encounter_url = enc.get('url')
        source_key = build_source_key(encounter_url, None, None, fallback_note_id)
        encounter_context = enc.get('context') or ""
        normalized_encounters.append({
            "key": build_encounter_key(word, source_key, encounter_context),
            "context": encounter_context,
            "url": encounter_url,
            "note_id": fallback_note_id,
            "reading_id": None,
            "video_id": None,
            "created_at": datetime_to_str(parse_datetime_value(enc.get('created_at'))),
            "lookup": build_lookup_payload(word, enc.get('lookup'))
        })

    sanitized['schemaVersion'] = 2
    sanitized['encounters'] = sort_encounters_desc(normalized_encounters)
    return sanitized


def sort_encounters_desc(encounters: List[dict]) -> List[dict]:
    return sorted(encounters, key=lambda e: e.get('created_at') or '', reverse=True)


def strip_legacy_payload_fields(payload: dict) -> dict:
    cleaned = dict(payload)
    for field in (
        'context', 'url', 'note_id', 'reading_id', 'video_id',
        'word', 'contextMeaning', 'partOfSpeech', 'grammarRole', 'explanation', 'baseForm', 'otherForms', 'otherMeanings',
        'm', 'p'
    ):
        cleaned.pop(field, None)
    return cleaned


def ensure_v2_payload(word: str, raw_payload: dict, row: Optional[dict] = None) -> dict:
    row = row or {}
    parsed_payload = parse_json_obj(raw_payload)
    legacy_context = parsed_payload.get('context') or row.get('context') or ""
    legacy_url = parsed_payload.get('url') if parsed_payload.get('url') is not None else row.get('url')
    root_lookup = build_lookup_payload(word, parsed_payload)
    payload = strip_legacy_payload_fields(parsed_payload)

    normalized_encounters: List[dict] = []
    raw_encounters = payload.get('encounters')
    if isinstance(raw_encounters, list):
        for enc in raw_encounters:
            if not isinstance(enc, dict):
                continue
            enc_context = enc.get('context') or legacy_context
            enc_url = enc.get('url') if enc.get('url') is not None else legacy_url
            enc_note_id = enc.get('note_id') if enc.get('note_id') is not None else row.get('note_id')
            enc_reading_id = enc.get('reading_id') if enc.get('reading_id') is not None else row.get('reading_id')
            enc_video_id = enc.get('video_id') if enc.get('video_id') is not None else row.get('video_id')
            source_key = build_source_key(enc_url, enc_reading_id, enc_video_id, enc_note_id)
            enc_key = enc.get('key') or build_encounter_key(word, source_key, enc_context)

            lookup_src = enc.get('lookup') if isinstance(enc.get('lookup'), dict) else root_lookup
            lookup = build_lookup_payload(word, lookup_src)
            normalized_encounters.append({
                "key": enc_key,
                "context": enc_context,
                "url": enc_url,
                "note_id": enc_note_id,
                "reading_id": enc_reading_id,
                "video_id": enc_video_id,
                "created_at": datetime_to_str(enc.get('created_at') or row.get('created_at')),
                "lookup": lookup
            })

    if not normalized_encounters:
        fallback_context = legacy_context
        fallback_url = legacy_url
        fallback_note_id = row.get('note_id')
        fallback_reading_id = row.get('reading_id')
        fallback_video_id = row.get('video_id')
        source_key = build_source_key(fallback_url, fallback_reading_id, fallback_video_id, fallback_note_id)
        normalized_encounters.append({
            "key": build_encounter_key(word, source_key, fallback_context),
            "context": fallback_context,
            "url": fallback_url,
            "note_id": fallback_note_id,
            "reading_id": fallback_reading_id,
            "video_id": fallback_video_id,
            "created_at": datetime_to_str(row.get('created_at')),
            "lookup": root_lookup
        })

    payload = dict(payload)
    payload['schemaVersion'] = 2
    payload['encounters'] = sort_encounters_desc(normalized_encounters)
    return payload


def sync_payload_from_latest_encounter(word: str, payload: dict) -> tuple[dict, dict]:
    encounters = payload.get('encounters') if isinstance(payload.get('encounters'), list) else []
    encounters = sort_encounters_desc(encounters)
    if not encounters:
        empty_lookup = build_lookup_payload(word, {})
        fallback = {
            "key": build_encounter_key(word, "manual", ""),
            "context": "",
            "url": None,
            "note_id": None,
            "reading_id": None,
            "video_id": None,
            "created_at": datetime_to_str(None),
            "lookup": empty_lookup
        }
        encounters = [fallback]
    latest = encounters[0]
    payload = strip_legacy_payload_fields(payload)
    payload['schemaVersion'] = 2
    payload['encounters'] = encounters
    return payload, latest


def append_or_get_encounter(
    word: str,
    payload: dict,
    context: str,
    url: Optional[str],
    note_id: Optional[int],
    reading_id: Optional[int],
    video_id: Optional[int],
    lookup_data: dict
) -> tuple[dict, bool, dict]:
    source_key = build_source_key(url, reading_id, video_id, note_id)
    enc_key = build_encounter_key(word, source_key, context)
    encounters = payload.get('encounters') if isinstance(payload.get('encounters'), list) else []

    for enc in encounters:
        if not isinstance(enc, dict):
            continue
        if enc.get('key') == enc_key:
            existing_lookup = build_lookup_payload(word, enc.get('lookup'))
            merged_lookup = dict(existing_lookup)
            merged_lookup.update(build_lookup_payload(word, lookup_data))
            enc['lookup'] = merged_lookup
            return payload, False, enc

    new_encounter = {
        "key": enc_key,
        "context": context,
        "url": url,
        "note_id": note_id,
        "reading_id": reading_id,
        "video_id": video_id,
        "created_at": datetime_to_str(None),
        "lookup": build_lookup_payload(word, lookup_data)
    }
    encounters.append(new_encounter)
    payload['encounters'] = encounters
    return payload, True, new_encounter


def extract_note_ids_from_payload(payload: dict, fallback_note_id: Optional[int] = None) -> set[int]:
    note_ids: set[int] = set()
    encounters = payload.get('encounters') if isinstance(payload.get('encounters'), list) else []
    for enc in encounters:
        if not isinstance(enc, dict):
            continue
        note_id = enc.get('note_id')
        if isinstance(note_id, int):
            note_ids.add(note_id)
    if isinstance(fallback_note_id, int):
        note_ids.add(fallback_note_id)
    return note_ids


def ensure_today_note(cursor) -> int:
    today = date.today().isoformat()
    cursor.execute("SELECT id FROM daily_notes WHERE day = %s", (today,))
    row = cursor.fetchone()
    if row:
        return row['id']
    title = f"{today} 的单词卡片"
    cursor.execute(
        "INSERT INTO daily_notes (day, title, word_count) VALUES (%s, %s, %s)",
        (today, title, 0)
    )
    return cursor.lastrowid


def get_note_encounters(payload: dict, note_id: int) -> List[dict]:
    encounters = payload.get('encounters') if isinstance(payload.get('encounters'), list) else []
    matches = [
        enc for enc in encounters
        if isinstance(enc, dict) and enc.get('note_id') == note_id
    ]
    return sort_encounters_desc(matches)


def decrement_note_counts(cursor, note_ids: set[int], amount: int = 1):
    if amount <= 0:
        return
    for nid in note_ids:
        cursor.execute(
            "UPDATE daily_notes SET word_count = GREATEST(0, word_count - %s) WHERE id = %s",
            (amount, nid)
        )


def save_word_to_db(word: str, context: str, data: dict, url: str = None, reading_id: int = None, video_id: int = None):
    """Save quick lookup results by word; aggregate different sources into encounters."""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                note_id = ensure_today_note(cursor)
                context = (context or "").strip()

                cursor.execute(
                    "SELECT * FROM saved_words WHERE LOWER(word) = LOWER(%s) ORDER BY reps DESC, created_at DESC, id ASC LIMIT 1",
                    (word,)
                )
                existing = cursor.fetchone()

                if not existing:
                    initial_row = {
                        "context": context,
                        "url": url,
                        "note_id": note_id,
                        "reading_id": reading_id,
                        "video_id": video_id,
                        "created_at": datetime.now()
                    }
                    payload = ensure_v2_payload(word, data, initial_row)
                    payload, latest = sync_payload_from_latest_encounter(word, payload)

                    sql = """
                        INSERT INTO saved_words (word, data, note_id, reading_id, video_id)
                        VALUES (%s, %s, %s, %s, %s)
                    """
                    cursor.execute(sql, (
                        word,
                        json.dumps(payload, ensure_ascii=False),
                        latest.get('note_id', note_id),
                        latest.get('reading_id', reading_id),
                        latest.get('video_id', video_id)
                    ))
                    cursor.execute("UPDATE daily_notes SET word_count = word_count + 1 WHERE id = %s", (note_id,))
                else:
                    existing_payload = ensure_v2_payload(existing['word'], existing.get('data'), existing)
                    note_linked_before = note_id in extract_note_ids_from_payload(existing_payload, existing.get('note_id'))
                    updated_payload, appended, _ = append_or_get_encounter(
                        existing['word'],
                        existing_payload,
                        context,
                        url,
                        note_id,
                        reading_id,
                        video_id,
                        data
                    )
                    updated_payload, latest = sync_payload_from_latest_encounter(existing['word'], updated_payload)

                    update_params = [
                        json.dumps(updated_payload, ensure_ascii=False),
                        latest.get('note_id', existing.get('note_id')),
                        latest.get('reading_id', existing.get('reading_id')),
                        latest.get('video_id', existing.get('video_id')),
                    ]
                    update_sql = """
                        UPDATE saved_words
                        SET data = %s, note_id = %s, reading_id = %s, video_id = %s
                    """
                    # 仅在新增 encounter 时刷新收录时间，避免同来源同句去重时时间抖动
                    if appended:
                        update_sql += ", created_at = %s"
                        update_params.append(datetime.now())
                    update_sql += " WHERE id = %s"
                    update_params.append(existing['id'])
                    cursor.execute(update_sql, tuple(update_params))

                    if appended and not note_linked_before:
                        cursor.execute("UPDATE daily_notes SET word_count = word_count + 1 WHERE id = %s", (note_id,))

            connection.commit()
        finally:
            connection.close()
    except Exception as e:
        print(f"Database Save Error: {e}")

# --- FSRS Implementation ---
fsrs = Scheduler()

def get_fsrs_rating(rating: int) -> Rating:
    """将前端 1-4 档转换为官方 Rating 枚举"""
    if rating == 1: return Rating.Again
    if rating == 2: return Rating.Hard
    if rating == 3: return Rating.Good
    return Rating.Easy

def format_saved_word(row):
    """Format DB row to SavedWord schema with v2 encounters + latest derived mirrors."""
    parsed_data = ensure_v2_payload(row['word'], row.get('data'), row)
    parsed_data, latest = sync_payload_from_latest_encounter(row['word'], parsed_data)

    return SavedWord(
        id=row['id'],
        word=row['word'],
        context=latest.get('context') or "",
        url=latest.get('url'),
        data=parsed_data,
        encounters=parsed_data.get('encounters') or [],
        created_at=row['created_at'].strftime('%Y-%m-%d %H:%M:%S') if row['created_at'] else None,
        note_id=latest.get('note_id') if latest.get('note_id') is not None else row.get('note_id'),
        reading_id=latest.get('reading_id') if latest.get('reading_id') is not None else row.get('reading_id'),
        video_id=latest.get('video_id') if latest.get('video_id') is not None else row.get('video_id'),
        stability=row['stability'],
        difficulty=row['difficulty'],
        elapsed_days=row['elapsed_days'],
        scheduled_days=row['scheduled_days'],
        last_review=row['last_review'].strftime('%Y-%m-%d %H:%M:%S') if row['last_review'] else None,
        reps=row['reps'],
        state=row['state']
    )




async def get_user_api_key(x_gemini_api_key: Annotated[Optional[str], Header()] = None):
    return x_gemini_api_key


async def get_llm_config_overrides(x_gemini_feature_config: Annotated[Optional[str], Header()] = None):
    if not x_gemini_feature_config:
        return {}
    try:
        payload = json.loads(x_gemini_feature_config)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}

# --- SmashEnglish Endpoints ---

@app.get("/fastapi/llm-configs", response_model=FeatureLLMConfigResponse)
async def get_llm_configs():
    return FeatureLLMConfigResponse(features=gemini.get_feature_configs())


@app.post("/fastapi/analyze", response_model=AnalysisResult)
async def analyze_sentence(
    request: AnalysisRequest,
    user_api_key: Optional[str] = Depends(get_user_api_key),
    llm_config_overrides: dict = Depends(get_llm_config_overrides)
):
    try:
        result = await gemini.analyze_sentence_service(request.sentence, user_api_key, llm_config_overrides)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fastapi/lookup", response_model=DictionaryResult)
async def lookup_word(
    request: LookupRequest,
    user_api_key: Optional[str] = Depends(get_user_api_key),
    llm_config_overrides: dict = Depends(get_llm_config_overrides)
):
    try:
        result = await gemini.lookup_word_service(request.word, user_api_key, llm_config_overrides)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fastapi/writing", response_model=WritingResult)
async def evaluate_writing(
    request: WritingRequest,
    user_api_key: Optional[str] = Depends(get_user_api_key),
    llm_config_overrides: dict = Depends(get_llm_config_overrides)
):
    try:
        result = await gemini.evaluate_writing_service(request.text, request.mode, user_api_key, llm_config_overrides)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fastapi/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user_api_key: Optional[str] = Depends(get_user_api_key),
    llm_config_overrides: dict = Depends(get_llm_config_overrides)
):
    try:
        response_text = await gemini.chat_service(request, user_api_key, llm_config_overrides)
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fastapi/quick-lookup", response_model=QuickLookupResult)
async def quick_lookup(
    request: QuickLookupRequest,
    user_api_key: Optional[str] = Depends(get_user_api_key),
    llm_config_overrides: dict = Depends(get_llm_config_overrides)
):
    """快速上下文查词 - 返回词条并自动保存到数据库"""
    try:
        result = await gemini.quick_lookup_service(request.word, request.context, user_api_key, llm_config_overrides)
        # 异步/后台保存
        save_word_to_db(
            request.word, 
            request.context, 
            result.model_dump(), 
            request.url,
            request.reading_id,
            request.video_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fastapi/rapid-lookup", response_model=RapidLookupResult)
async def rapid_lookup(
    request: RapidLookupRequest,
    user_api_key: Optional[str] = Depends(get_user_api_key),
    llm_config_overrides: dict = Depends(get_llm_config_overrides)
):
    """极简上下文查词 - 极致速度 (不保存数据库)"""
    try:
        result = await gemini.rapid_lookup_service(request.word, request.context, user_api_key, llm_config_overrides)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fastapi/translate", response_model=TranslateResult)
async def translate_endpoint(
    request: TranslateRequest,
    user_api_key: Optional[str] = Depends(get_user_api_key),
    llm_config_overrides: dict = Depends(get_llm_config_overrides)
):
    """极速翻译接口"""
    try:
        result = await gemini.translate_service(request.text, user_api_key, llm_config_overrides)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fastapi/translate-advanced", response_model=TranslateResult)
async def translate_advanced_endpoint(
    request: AdvancedTranslateRequest,
    user_api_key: Optional[str] = Depends(get_user_api_key),
    llm_config_overrides: dict = Depends(get_llm_config_overrides)
):
    """高级翻译接口 - 支持多语言与自定义指令"""
    try:
        result = await gemini.translate_advanced_service(request, user_api_key, llm_config_overrides)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/fastapi/daily-notes", response_model=DailyNotesResponse)
async def get_daily_notes():
    """获取所有日记概览卡片"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                sql = "SELECT id, title, day, summary, content, word_count, created_at FROM daily_notes ORDER BY day DESC"
                cursor.execute(sql)
                rows = cursor.fetchall()
                notes = []
                for row in rows:
                    # 处理日期和时间戳为字符串
                    row['day'] = str(row['day'])
                    row['created_at'] = row['created_at'].strftime('%Y-%m-%d %H:%M:%S') if row['created_at'] else ""
                    notes.append(DailyNote(**row))
                return DailyNotesResponse(notes=notes)
        finally:
            connection.close()
    except Exception as e:
        print(f"Database Get Notes Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fastapi/daily-notes/{note_id}", response_model=NoteDetailResponse)
async def get_note_detail(note_id: int):
    """获取特定卡片的详情及其单词列表"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                # 1. 查找 Note 详情
                cursor.execute("SELECT * FROM daily_notes WHERE id = %s", (note_id,))
                note_row = cursor.fetchone()
                if not note_row:
                    raise HTTPException(status_code=404, detail="Note not found")
                
                note_row['day'] = str(note_row['day'])
                note_row['created_at'] = note_row['created_at'].strftime('%Y-%m-%d %H:%M:%S') if note_row['created_at'] else ""
                note = DailyNote(**note_row)

                # 2. 基于 encounters 过滤该 Note 的单词
                cursor.execute("SELECT * FROM saved_words ORDER BY created_at DESC")
                word_rows = cursor.fetchall()
                words = []
                for row in word_rows:
                    formatted = format_saved_word(row)
                    if get_note_encounters(formatted.data, note_id):
                        words.append(formatted)

                return NoteDetailResponse(note=note, words=words)
        finally:
            connection.close()
    except Exception as e:
        print(f"Database Get Note Detail Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fastapi/review/feedback")
async def submit_review_feedback(request: FSRSFeedbackRequest):
    """提交复习反馈，使用官方 FSRS 库更新状态"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                # 1. 获取单词当前状态
                cursor.execute("SELECT * FROM saved_words WHERE id = %s", (request.word_id,))
                word = cursor.fetchone()
                if not word:
                    raise HTTPException(status_code=404, detail="Word not found")

                # 2. 构建官方 Card 对象
                # 官方 last_review 要求是 UTC datetime 或 None
                last_review = word['last_review']
                if last_review and last_review.tzinfo is None:
                    last_review = last_review.replace(tzinfo=timezone.utc)

                if word['reps'] == 0:
                    # 新词，使用默认构造函数
                    card = Card(
                        state=State.Learning,
                        due=datetime.now(timezone.utc)
                    )
                else:
                    # 已有记录的词
                    card = Card(
                        due=word['due'].replace(tzinfo=timezone.utc) if word['due'] else datetime.now(timezone.utc),
                        stability=word['stability'],
                        difficulty=word['difficulty'],
                        state=State(word['state']) if word['state'] > 0 else State.Learning,
                        last_review=last_review
                    )

                # 3. 使用官方库计算新状态
                now = datetime.now(timezone.utc)
                rating = get_fsrs_rating(request.rating)
                
                # V6 API 使用 review_card，返回 (new_card, review_log)
                new_card, _ = fsrs.review_card(card, rating, now)

                # 4. 更新数据库
                sql = """
                    UPDATE saved_words SET 
                        stability = %s, 
                        difficulty = %s, 
                        elapsed_days = %s, 
                        scheduled_days = %s, 
                        last_review = %s, 
                        due = %s, 
                        reps = %s, 
                        state = %s 
                    WHERE id = %s
                """
                cursor.execute(sql, (
                    new_card.stability,
                    new_card.difficulty,
                    (now - last_review).days if last_review else 0,
                    (new_card.due - now).days,
                    new_card.last_review,
                    new_card.due,
                    word['reps'] + 1,
                    new_card.state.value,
                    request.word_id
                ))
                connection.commit()
            return {"status": "success", "next_review": new_card.due.strftime('%Y-%m-%d %H:%M:%S')}
        finally:
            connection.close()
    except Exception as e:
        print(f"FSRS Feedback Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fastapi/daily-notes/{note_id}/summarize")
async def summarize_daily_note(
    note_id: int,
    user_api_key: Optional[str] = Depends(get_user_api_key),
    llm_config_overrides: dict = Depends(get_llm_config_overrides)
):
    """为当天的笔记生成 AI 总结博客 (更新标题、简介和内容)"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                # 1. 获取该 note 相关单词（基于 encounters）
                cursor.execute("SELECT * FROM saved_words ORDER BY created_at DESC")
                words_raw = cursor.fetchall()

                words = []
                for row in words_raw:
                    payload = ensure_v2_payload(row['word'], row.get('data'), row)
                    note_encounters = get_note_encounters(payload, note_id)
                    if not note_encounters:
                        continue
                    latest_encounter = note_encounters[0]
                    words.append({
                        "word": row['word'],
                        "context": latest_encounter.get('context') or "",
                        "url": latest_encounter.get('url'),
                        "data": latest_encounter.get('lookup') or build_lookup_payload(row['word'], payload)
                    })

                if not words:
                    raise HTTPException(status_code=400, detail="No words to summarize")
                
                # 2. 调用 Gemini 生成结构化内容
                blog_result = await gemini.generate_daily_summary_service(words, user_api_key, llm_config_overrides)
                
                # 3. 更新到数据库 (title, summary -> prologue, content)
                cursor.execute(
                    "UPDATE daily_notes SET title = %s, summary = %s, content = %s WHERE id = %s", 
                    (blog_result.title, blog_result.prologue, blog_result.content, note_id)
                )
            
            connection.commit()
            return {
                "status": "success", 
                "title": blog_result.title,
                "summary": blog_result.prologue,
                "content": blog_result.content
            }
        finally:
            connection.close()
    except Exception as e:
        print(f"Summarize Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/fastapi/saved-words/{word_id}")
async def delete_saved_word(word_id: int):
    """删除收藏的单词"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM saved_words WHERE id = %s", (word_id,))
                row = cursor.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Word not found")

                payload = ensure_v2_payload(row['word'], row.get('data'), row)
                affected_note_ids = extract_note_ids_from_payload(payload, row.get('note_id'))
                cursor.execute("DELETE FROM saved_words WHERE id = %s", (word_id,))
                decrement_note_counts(cursor, affected_note_ids, 1)
            connection.commit()
            return {"status": "success"}
        finally:
            connection.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/fastapi/saved-words/{word_id}/encounters/{encounter_key}")
async def delete_saved_word_encounter(word_id: int, encounter_key: str):
    """删除单词的一条来源 encounter；若无来源剩余则删除整词。"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM saved_words WHERE id = %s", (word_id,))
                row = cursor.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Word not found")

                payload = ensure_v2_payload(row['word'], row.get('data'), row)
                encounters = payload.get('encounters') if isinstance(payload.get('encounters'), list) else []
                target = next((e for e in encounters if isinstance(e, dict) and e.get('key') == encounter_key), None)
                if not target:
                    raise HTTPException(status_code=404, detail="Encounter not found")

                removed_note_id = target.get('note_id')
                remaining = [e for e in encounters if not (isinstance(e, dict) and e.get('key') == encounter_key)]

                if not remaining:
                    cursor.execute("DELETE FROM saved_words WHERE id = %s", (word_id,))
                    if isinstance(removed_note_id, int):
                        decrement_note_counts(cursor, {removed_note_id}, 1)
                    connection.commit()
                    return {"status": "deleted", "deleted": True, "word_id": word_id}

                payload['encounters'] = remaining
                payload, latest = sync_payload_from_latest_encounter(row['word'], payload)

                cursor.execute(
                    """
                    UPDATE saved_words
                    SET data = %s, note_id = %s, reading_id = %s, video_id = %s
                    WHERE id = %s
                    """,
                    (
                        json.dumps(payload, ensure_ascii=False),
                        latest.get('note_id', row.get('note_id')),
                        latest.get('reading_id', row.get('reading_id')),
                        latest.get('video_id', row.get('video_id')),
                        word_id
                    )
                )

                if isinstance(removed_note_id, int):
                    still_has_note = any(
                        isinstance(e, dict) and e.get('note_id') == removed_note_id
                        for e in remaining
                    )
                    if not still_has_note:
                        decrement_note_counts(cursor, {removed_note_id}, 1)

                cursor.execute("SELECT * FROM saved_words WHERE id = %s", (word_id,))
                updated_row = cursor.fetchone()
            connection.commit()
            return {"status": "updated", "deleted": False, "word": format_saved_word(updated_row).model_dump()}
        finally:
            connection.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fastapi/saved-words", response_model=SavedWordsResponse)
async def get_all_saved_words():
    """获取所有收藏的单词（无视日期）"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM saved_words ORDER BY created_at DESC"
                cursor.execute(sql)
                rows = cursor.fetchall()
                words = []
                for row in rows:
                    words.append(format_saved_word(row))
                return SavedWordsResponse(words=words)
        finally:
            connection.close()
    except Exception as e:
        print(f"Database Get All Words Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fastapi/saved-words/export", response_model=SavedWordsExportResponse)
async def export_saved_words():
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM saved_words ORDER BY created_at DESC")
                rows = cursor.fetchall()
                words = [format_saved_word(row) for row in rows]
                return SavedWordsExportResponse(
                    exported_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    words=words
                )
        finally:
            connection.close()
    except Exception as e:
        print(f"Export Saved Words Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fastapi/saved-words/import", response_model=SavedWordsImportResponse)
async def import_saved_words(request: SavedWordsImportRequest):
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                today_note_id = ensure_today_note(cursor)
                total = len(request.words)
                imported = 0
                merged = 0
                skipped = 0

                for item in request.words:
                    clean_word = (item.word or "").strip()
                    if not clean_word:
                        skipped += 1
                        continue

                    seed_payload = parse_json_obj(item.data)
                    if item.encounters:
                        seed_payload = dict(seed_payload)
                        seed_payload['encounters'] = [enc.model_dump() for enc in item.encounters]

                    if not seed_payload:
                        seed_payload = {
                            "contextMeaning": "",
                            "partOfSpeech": "",
                            "grammarRole": "",
                            "explanation": "",
                            "otherMeanings": []
                        }

                    if item.context and 'context' not in seed_payload:
                        seed_payload['context'] = item.context
                    if item.url and 'url' not in seed_payload:
                        seed_payload['url'] = item.url

                    import_payload = sanitize_import_payload(clean_word, seed_payload, today_note_id)
                    import_payload, latest_import = sync_payload_from_latest_encounter(clean_word, import_payload)
                    import_created_at = parse_datetime_value(
                        latest_import.get('created_at') or item.created_at
                    ) or datetime.now()
                    import_last_review = parse_datetime_value(item.last_review)

                    cursor.execute(
                        "SELECT * FROM saved_words WHERE LOWER(word) = LOWER(%s) ORDER BY reps DESC, created_at DESC, id ASC LIMIT 1",
                        (clean_word,)
                    )
                    existing = cursor.fetchone()

                    if existing:
                        existing_payload = ensure_v2_payload(existing['word'], existing.get('data'), existing)
                        appended_any = False
                        for enc in import_payload.get('encounters') or []:
                            existing_payload, appended, _ = append_or_get_encounter(
                                existing['word'],
                                existing_payload,
                                enc.get('context') or "",
                                enc.get('url'),
                                today_note_id,
                                None,
                                None,
                                enc.get('lookup') or {}
                            )
                            appended_any = appended_any or appended

                        existing_payload, latest = sync_payload_from_latest_encounter(existing['word'], existing_payload)
                        use_import_progress = item.reps > (existing.get('reps') or 0)

                        cursor.execute(
                            """
                            UPDATE saved_words
                            SET data = %s,
                                note_id = %s,
                                reading_id = %s,
                                video_id = %s,
                                created_at = %s,
                                stability = %s,
                                difficulty = %s,
                                elapsed_days = %s,
                                scheduled_days = %s,
                                last_review = %s,
                                reps = %s,
                                state = %s
                            WHERE id = %s
                            """,
                            (
                                json.dumps(existing_payload, ensure_ascii=False),
                                latest.get('note_id', today_note_id),
                                None,
                                None,
                                import_created_at if appended_any else existing.get('created_at'),
                                item.stability if use_import_progress else existing.get('stability', 0.0),
                                item.difficulty if use_import_progress else existing.get('difficulty', 0.0),
                                item.elapsed_days if use_import_progress else existing.get('elapsed_days', 0),
                                item.scheduled_days if use_import_progress else existing.get('scheduled_days', 0),
                                import_last_review if use_import_progress else existing.get('last_review'),
                                item.reps if use_import_progress else existing.get('reps', 0),
                                item.state if use_import_progress else existing.get('state', 0),
                                existing['id']
                            )
                        )
                        merged += 1
                        if appended_any:
                            imported += 1
                            cursor.execute("UPDATE daily_notes SET word_count = word_count + 1 WHERE id = %s", (today_note_id,))
                    else:
                        cursor.execute(
                            """
                            INSERT INTO saved_words (
                                note_id, word, data, created_at, stability, difficulty,
                                elapsed_days, scheduled_days, last_review, reps, state, reading_id, video_id
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                today_note_id,
                                clean_word,
                                json.dumps(import_payload, ensure_ascii=False),
                                import_created_at,
                                item.stability,
                                item.difficulty,
                                item.elapsed_days,
                                item.scheduled_days,
                                import_last_review,
                                item.reps,
                                item.state,
                                None,
                                None
                            )
                        )
                        imported += 1
                        cursor.execute("UPDATE daily_notes SET word_count = word_count + 1 WHERE id = %s", (today_note_id,))

            connection.commit()
            return SavedWordsImportResponse(
                total=total,
                imported=imported,
                merged=merged,
                skipped=skipped
            )
        finally:
            connection.close()
    except Exception as e:
        print(f"Import Saved Words Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fastapi/saved-words", response_model=SavedWord)
async def create_saved_word(word: SavedWordCreate):
    """手动添加收藏单词"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                note_id = word.note_id if word.note_id is not None else ensure_today_note(cursor)
                context = (word.context or "").strip()
                lookup_data = build_lookup_payload(word.word, word.data or {})

                cursor.execute(
                    "SELECT * FROM saved_words WHERE LOWER(word) = LOWER(%s) ORDER BY reps DESC, created_at DESC, id ASC LIMIT 1",
                    (word.word,)
                )
                existing = cursor.fetchone()

                if existing:
                    payload = ensure_v2_payload(existing['word'], existing.get('data'), existing)
                    note_linked_before = note_id in extract_note_ids_from_payload(payload, existing.get('note_id'))
                    payload, appended, _ = append_or_get_encounter(
                        existing['word'],
                        payload,
                        context,
                        word.url,
                        note_id,
                        None,
                        None,
                        lookup_data
                    )
                    payload, latest = sync_payload_from_latest_encounter(existing['word'], payload)
                    update_params = [
                        json.dumps(payload, ensure_ascii=False),
                        latest.get('note_id', existing.get('note_id')),
                        latest.get('reading_id', existing.get('reading_id')),
                        latest.get('video_id', existing.get('video_id')),
                    ]
                    update_sql = """
                        UPDATE saved_words
                        SET data = %s, note_id = %s, reading_id = %s, video_id = %s
                    """
                    # 手动添加同词不同上下文时也刷新收录时间，保持排序与“最近收录”一致
                    if appended:
                        update_sql += ", created_at = %s"
                        update_params.append(datetime.now())
                    update_sql += " WHERE id = %s"
                    update_params.append(existing['id'])
                    cursor.execute(update_sql, tuple(update_params))
                    if appended and not note_linked_before:
                        cursor.execute("UPDATE daily_notes SET word_count = word_count + 1 WHERE id = %s", (note_id,))
                    word_id = existing['id']
                else:
                    initial_row = {
                        "context": context,
                        "url": word.url,
                        "note_id": note_id,
                        "reading_id": None,
                        "video_id": None,
                        "created_at": datetime.now()
                    }
                    payload = ensure_v2_payload(word.word, lookup_data, initial_row)
                    payload, latest = sync_payload_from_latest_encounter(word.word, payload)
                    cursor.execute(
                        """
                        INSERT INTO saved_words (word, data, note_id, reading_id, video_id)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            word.word,
                            json.dumps(payload, ensure_ascii=False),
                            latest.get('note_id', note_id),
                            None,
                            None
                        )
                    )
                    word_id = cursor.lastrowid
                    cursor.execute("UPDATE daily_notes SET word_count = word_count + 1 WHERE id = %s", (note_id,))

                cursor.execute("SELECT * FROM saved_words WHERE id = %s", (word_id,))
                row = cursor.fetchone()
            connection.commit()
            return format_saved_word(row)
        finally:
            connection.close()
    except HTTPException:
        raise
    except Exception as e:
        print(f"Create Word Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/fastapi/saved-words/{word_id}", response_model=SavedWord)
async def update_saved_word(word_id: int, word: SavedWordUpdate):
    """更新收藏单词"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                data = word.model_dump(exclude_unset=True)
                if not data:
                    raise HTTPException(status_code=400, detail="No fields to update")

                cursor.execute("SELECT * FROM saved_words WHERE id = %s", (word_id,))
                existing = cursor.fetchone()
                if not existing:
                    raise HTTPException(status_code=404, detail="Word not found")

                new_word = data.get('word') if isinstance(data.get('word'), str) else existing['word']
                payload = ensure_v2_payload(existing['word'], existing.get('data'), existing)
                for enc in payload.get('encounters') or []:
                    if not isinstance(enc, dict):
                        continue
                    lookup = build_lookup_payload(new_word, enc.get('lookup'))
                    source_key = build_source_key(enc.get('url'), enc.get('reading_id'), enc.get('video_id'), enc.get('note_id'))
                    enc['lookup'] = lookup
                    enc['key'] = build_encounter_key(new_word, source_key, enc.get('context') or '')
                payload, latest = sync_payload_from_latest_encounter(new_word, payload)

                if 'context' in data:
                    latest['context'] = data['context'] or ""
                if 'url' in data:
                    latest['url'] = data['url']
                if 'note_id' in data:
                    latest['note_id'] = data['note_id']

                if 'data' in data and isinstance(data['data'], dict):
                    latest_lookup = build_lookup_payload(new_word, latest.get('lookup'))
                    latest_lookup.update(build_lookup_payload(new_word, data['data']))
                    latest['lookup'] = latest_lookup

                payload['encounters'] = [
                    latest if isinstance(enc, dict) and enc.get('key') == latest.get('key') else enc
                    for enc in (payload.get('encounters') or [])
                ]

                payload, latest = sync_payload_from_latest_encounter(new_word, payload)

                cursor.execute(
                    """
                    UPDATE saved_words
                    SET word = %s, data = %s, note_id = %s, reading_id = %s, video_id = %s
                    WHERE id = %s
                    """,
                    (
                        new_word,
                        json.dumps(payload, ensure_ascii=False),
                        latest.get('note_id', existing.get('note_id')),
                        latest.get('reading_id', existing.get('reading_id')),
                        latest.get('video_id', existing.get('video_id')),
                        word_id
                    )
                )

                cursor.execute("SELECT * FROM saved_words WHERE id = %s", (word_id,))
                row = cursor.fetchone()
            connection.commit()
            return format_saved_word(row)
        finally:
            connection.close()
    except HTTPException:
        raise
    except Exception as e:
        print(f"Update Word Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fastapi/saved-words/batch-delete")
async def batch_delete_words(request: BatchDeleteRequest):
    """批量删除收藏的单词"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                if not request.word_ids:
                    return {"status": "success", "count": 0}

                placeholders = ', '.join(['%s'] * len(request.word_ids))
                cursor.execute(f"SELECT * FROM saved_words WHERE id IN ({placeholders})", tuple(request.word_ids))
                rows = cursor.fetchall()
                if not rows:
                    return {"status": "success", "count": 0}

                note_decrements: dict[int, int] = {}
                for row in rows:
                    payload = ensure_v2_payload(row['word'], row.get('data'), row)
                    note_ids = extract_note_ids_from_payload(payload, row.get('note_id'))
                    for nid in note_ids:
                        note_decrements[nid] = note_decrements.get(nid, 0) + 1

                cursor.execute(f"DELETE FROM saved_words WHERE id IN ({placeholders})", tuple(request.word_ids))
                for nid, cnt in note_decrements.items():
                    cursor.execute(
                        "UPDATE daily_notes SET word_count = GREATEST(0, word_count - %s) WHERE id = %s",
                        (cnt, nid)
                    )

            connection.commit()
            return {"status": "success", "count": len(rows)}
        finally:
            connection.close()
    except HTTPException:
        raise
    except Exception as e:
        print(f"Batch Delete Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Video Notebook Endpoints ---

@app.post("/fastapi/notebooks", response_model=VideoNotebook)
async def create_notebook(notebook: VideoNotebookCreate):
    """创建新的视频笔记本"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                sql = """
                    INSERT INTO video_notebooks (title, video_url, video_id, srt_content, thumbnail_url)
                    VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    notebook.title, 
                    notebook.video_url, 
                    notebook.video_id, 
                    notebook.srt_content, 
                    notebook.thumbnail_url
                ))
                notebook_id = cursor.lastrowid
                
                # 获取创建后的完整对象
                cursor.execute("SELECT * FROM video_notebooks WHERE id = %s", (notebook_id,))
                new_row = cursor.fetchone()
                
                # 格式化日期
                new_row['created_at'] = new_row['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                new_row['updated_at'] = new_row['updated_at'].strftime('%Y-%m-%d %H:%M:%S')
                
                connection.commit()
                return VideoNotebook(**new_row)
        finally:
            connection.close()
    except Exception as e:
        print(f"Create Notebook Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fastapi/notebooks", response_model=VideoNotebookListResponse)
async def list_notebooks():
    """获取笔记本列表（不包含巨大的 srt_content）"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                # 注意：这里特意排除了 srt_content 字段以减小响应体积
                sql = """
                    SELECT id, title, video_url, video_id, thumbnail_url, created_at, updated_at 
                    FROM video_notebooks 
                    ORDER BY created_at DESC
                """
                cursor.execute(sql)
                rows = cursor.fetchall()
                notebooks = []
                for row in rows:
                    row['created_at'] = row['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                    row['updated_at'] = row['updated_at'].strftime('%Y-%m-%d %H:%M:%S')
                    # srt_content 设为 None 或空，因为它在列表中没意义
                    row['srt_content'] = None
                    notebooks.append(VideoNotebook(**row))
                return VideoNotebookListResponse(notebooks=notebooks)
        finally:
            connection.close()
    except Exception as e:
        print(f"List Notebooks Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fastapi/notebooks/{notebook_id}", response_model=VideoNotebook)
async def get_notebook_detail(notebook_id: int):
    """获取笔记本详情（包含 srt_content）"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM video_notebooks WHERE id = %s", (notebook_id,))
                row = cursor.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Notebook not found")
                
                row['created_at'] = row['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                row['updated_at'] = row['updated_at'].strftime('%Y-%m-%d %H:%M:%S')
                return VideoNotebook(**row)
        finally:
            connection.close()
    except HTTPException:
        raise
    except Exception as e:
        print(f"Get Notebook Detail Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/fastapi/notebooks/{notebook_id}", response_model=VideoNotebook)
async def update_notebook(notebook_id: int, notebook: VideoNotebookUpdate):
    """更新视频笔记本"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                # 获取要更新的字段及其值
                data = notebook.model_dump(exclude_unset=True)
                if not data:
                    raise HTTPException(status_code=400, detail="No fields to update")
                
                # 构建动态 SQL
                fields = []
                values = []
                for k, v in data.items():
                    fields.append(f"{k} = %s")
                    values.append(v)
                
                sql = f"UPDATE video_notebooks SET {', '.join(fields)} WHERE id = %s"
                values.append(notebook_id)
                
                cursor.execute(sql, values)
                
                # 获取更新后的完整数据
                cursor.execute("SELECT * FROM video_notebooks WHERE id = %s", (notebook_id,))
                row = cursor.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Notebook not found")
                
                row['created_at'] = row['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                row['updated_at'] = row['updated_at'].strftime('%Y-%m-%d %H:%M:%S')
                
                connection.commit()
                return VideoNotebook(**row)
        finally:
            connection.close()
    except HTTPException:
        raise
    except Exception as e:
        print(f"Update Notebook Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/fastapi/notebooks/{notebook_id}")
async def delete_notebook(notebook_id: int):
    """删除笔记本"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM video_notebooks WHERE id = %s", (notebook_id,))
            connection.commit()
            return {"status": "success"}
        finally:
            connection.close()
    except Exception as e:
        print(f"Delete Notebook Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Reading Notebook Endpoints ---

@app.post("/fastapi/reading-notebooks", response_model=ReadingNotebook)
async def create_reading_notebook(notebook: ReadingNotebookCreate):
    """创建新的精读笔记本"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                sql = """
                    INSERT INTO reading_notebooks (title, content, source_url, cover_image_url, description, word_count)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    notebook.title, 
                    notebook.content, 
                    notebook.source_url, 
                    notebook.cover_image_url, 
                    notebook.description, 
                    notebook.word_count
                ))
                notebook_id = cursor.lastrowid
                
                # 获取创建后的完整对象
                cursor.execute("SELECT * FROM reading_notebooks WHERE id = %s", (notebook_id,))
                new_row = cursor.fetchone()
                
                # 格式化日期
                new_row['created_at'] = new_row['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                new_row['updated_at'] = new_row['updated_at'].strftime('%Y-%m-%d %H:%M:%S')
                
                connection.commit()
                return ReadingNotebook(**new_row)
        finally:
            connection.close()
    except Exception as e:
        print(f"Create Reading Notebook Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fastapi/reading-notebooks", response_model=ReadingNotebookListResponse)
async def list_reading_notebooks():
    """获取精读笔记本列表（不包含巨大的 content）"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                # 注意：这里特意排除了 content 字段以减小响应体积
                sql = """
                    SELECT id, title, source_url, cover_image_url, description, word_count, created_at, updated_at 
                    FROM reading_notebooks 
                    ORDER BY created_at DESC
                """
                cursor.execute(sql)
                rows = cursor.fetchall()
                notebooks = []
                for row in rows:
                    row['created_at'] = row['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                    row['updated_at'] = row['updated_at'].strftime('%Y-%m-%d %H:%M:%S')
                    # content 设为 None
                    row['content'] = None
                    notebooks.append(ReadingNotebook(**row))
                return ReadingNotebookListResponse(notebooks=notebooks)
        finally:
            connection.close()
    except Exception as e:
        print(f"List Reading Notebooks Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fastapi/reading-notebooks/{notebook_id}", response_model=ReadingNotebook)
async def get_reading_notebook_detail(notebook_id: int):
    """获取精读笔记本详情（包含 content）"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM reading_notebooks WHERE id = %s", (notebook_id,))
                row = cursor.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Reading Notebook not found")
                
                row['created_at'] = row['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                row['updated_at'] = row['updated_at'].strftime('%Y-%m-%d %H:%M:%S')
                return ReadingNotebook(**row)
        finally:
            connection.close()
    except HTTPException:
        raise
    except Exception as e:
        print(f"Get Reading Notebook Detail Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/fastapi/reading-notebooks/{notebook_id}", response_model=ReadingNotebook)
async def update_reading_notebook(notebook_id: int, notebook: ReadingNotebookUpdate):
    """更新精读笔记本"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                # 获取要更新的字段及其值
                data = notebook.model_dump(exclude_unset=True)
                if not data:
                    raise HTTPException(status_code=400, detail="No fields to update")
                
                # 构建动态 SQL
                fields = []
                values = []
                for k, v in data.items():
                    fields.append(f"{k} = %s")
                    values.append(v)
                
                sql = f"UPDATE reading_notebooks SET {', '.join(fields)} WHERE id = %s"
                values.append(notebook_id)
                
                cursor.execute(sql, values)
                
                # 获取更新后的完整数据
                cursor.execute("SELECT * FROM reading_notebooks WHERE id = %s", (notebook_id,))
                row = cursor.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Reading Notebook not found")
                
                row['created_at'] = row['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                row['updated_at'] = row['updated_at'].strftime('%Y-%m-%d %H:%M:%S')
                
                connection.commit()
                return ReadingNotebook(**row)
        finally:
            connection.close()
    except HTTPException:
        raise
    except Exception as e:
        print(f"Update Reading Notebook Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/fastapi/reading-notebooks/{notebook_id}")
async def delete_reading_notebook(notebook_id: int):
    """删除精读笔记本"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM reading_notebooks WHERE id = %s", (notebook_id,))
            connection.commit()
            return {"status": "success"}
        finally:
            connection.close()
    except Exception as e:
        print(f"Delete Reading Notebook Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- FSRS Review Endpoints ---

@app.get("/fastapi/review/today", response_model=TodayReviewResponse)
async def get_today_review():
    """获取项目今日复习，如果不存在则立刻创建占位记录以锁定单词队列"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                # 1. 检查今日是否已经有了记录 (使用 review_date)
                today = date.today().isoformat()
                cursor.execute("SELECT * FROM review_articles WHERE review_date = %s", (today,))
                article_row = cursor.fetchone()

                if article_row:
                    # 如果已存在，获取关联单词
                    word_ids = json.loads(article_row['words_json'])
                    if word_ids:
                        placeholders = ', '.join(['%s'] * len(word_ids))
                        cursor.execute(f"SELECT * FROM saved_words WHERE id IN ({placeholders})", tuple(word_ids))
                        article_word_rows = cursor.fetchall()
                        word_map = {r['id']: format_saved_word(r) for r in article_word_rows}
                        words = [word_map[wid] for wid in word_ids if wid in word_map]
                    else:
                        words = []
                    
                    # 格式化数据
                    article_row['words_json'] = word_ids
                    article_row['review_date'] = article_row['review_date'].isoformat()
                    article_row['created_at'] = article_row['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                    return TodayReviewResponse(
                        article=ReviewArticle(**article_row),
                        words=words,
                        is_new_article=False
                    )

                # 2. 如果没有记录，立刻挑选 30 个词并创建占位记录
                # 策略：(已到期的词 + 新词) 混合，优先选到期最久的
                cursor.execute("""
                    SELECT * FROM saved_words 
                    WHERE due <= NOW() OR reps = 0
                    ORDER BY 
                        (CASE WHEN reps = 0 THEN 1 ELSE 0 END) ASC, -- 到期词优先 (0), 新词次之 (1)
                        due ASC 
                    LIMIT 30
                """)
                word_rows = cursor.fetchall()
                if not word_rows:
                    return TodayReviewResponse(article=None, words=[], is_new_article=True)

                words = [format_saved_word(r) for r in word_rows]
                word_ids = [w.id for w in words]

                # 创建占位记录
                sql = "INSERT INTO review_articles (review_date, title, content, article_type, words_json) VALUES (%s, %s, %s, %s, %s)"
                cursor.execute(sql, (
                    today,
                    f"{today} 复习计划",
                    "",  # 空内容，等待 AI 导入
                    "none",
                    json.dumps(word_ids)
                ))
                article_id = cursor.lastrowid
                connection.commit()

                return TodayReviewResponse(
                    article=ReviewArticle(
                        id=article_id,
                        review_date=today,
                        title=f"{today} 复习计划",
                        content="",
                        article_type="none",
                        words_json=word_ids,
                        created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ),
                    words=words,
                    is_new_article=True
                )
        finally:
            connection.close()
    except Exception as e:
        print(f"Review Today Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fastapi/review/prompt", response_model=ReviewPromptResponse)
async def get_review_prompt():
    """获取今日复习单词的 Prompt，锁定单词队列"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                today = date.today().isoformat()
                cursor.execute("SELECT words_json FROM review_articles WHERE review_date = %s", (today,))
                row = cursor.fetchone()
                
                if row:
                    word_ids = json.loads(row['words_json'])
                    placeholders = ', '.join(['%s'] * len(word_ids))
                    cursor.execute(f"SELECT * FROM saved_words WHERE id IN ({placeholders})", tuple(word_ids))
                    word_rows = cursor.fetchall()
                else:
                    # 如果还没占位，执行逻辑选取（保持兜底）
                    cursor.execute("""
                        SELECT * FROM saved_words ORDER BY 
                        CASE WHEN last_review IS NULL THEN 100 ELSE (DATEDIFF(NOW(), last_review) / scheduled_days) END DESC LIMIT 30
                    """)
                    word_rows = cursor.fetchall()

                if not word_rows:
                    return ReviewPromptResponse(prompt="没有待复习单词。", words=[])

                words = [format_saved_word(r) for r in word_rows]
                
                # 构建单词元数据（移除 ID）
                words_info = ""
                for formatted in words:
                    formatted_data = formatted.model_dump()
                    encounters = formatted_data['data'].get('encounters') or []
                    latest_encounter = encounters[0] if encounters else {}
                    lookup = latest_encounter.get('lookup') or formatted_data['data']
                    meaning = lookup.get('contextMeaning') or lookup.get('m') or '未知'
                    pos = lookup.get('partOfSpeech') or ''
                    role = lookup.get('grammarRole') or ''
                    exp = lookup.get('explanation') or ''
                    others = lookup.get('otherMeanings') or []
                    
                    word_meta = {
                        "word": formatted.word,
                        "contextMeaning": meaning,
                        "partOfSpeech": pos,
                        "grammarRole": role,
                        "explanation": exp,
                        "otherMeanings": others,
                        "context": latest_encounter.get('context') or '',
                        "url": latest_encounter.get('url')
                    }
                    words_info += f"- {json.dumps(word_meta, ensure_ascii=False)}\n"

                # 提取完整 ID 数组用于嵌入 JSON 结构
                word_ids_str = ", ".join([str(r['id']) for r in word_rows])

                prompt = f"""
你是一位天才内容创作者，擅长编写极具吸引力的英语学习内容。
今天你需要根据用户复习的 {len(word_rows)} 个单词，编写一篇文章。形式候选：播客、采访、辩论、深度博客、新闻特写。

## 待包含的单词及其详细背景 (JSON 格式)
{words_info}

## 核心任务
1. **创作内容**: 编写一篇生动有趣的英文文章（包含对应的中文翻译）。
2. **自然嵌入**: 单词要自然地出现在情境中。
3. **双语格式**: Markdown 格式。先展示完整的英文版，用 `---` 分隔后展示中文翻译版。
4. **重点突出**: 在英文版中，将这些单词用 **加粗** 标注。
5. **对话格式**（如果是播客/采访）:
   - 使用 `**Host:**` 和 `> **Guest:**` 来区分说话人
   - Guest 的对话用引用符号（`>`）包裹供视觉差异。

## 输出格式要求
严格返回如下格式的 JSON：
```json
{{
  "title": "双语标题",
  "content": "Markdown 正文",
  "article_type": "文章类型"
}}
```
注意：**内容中不要包含 words_ids 字段，系统会自动处理关联**。
"""
                return ReviewPromptResponse(prompt=prompt.strip(), words=words)
        finally:
            connection.close()
    except Exception as e:
        print(f"Review Prompt Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fastapi/review/import")
async def import_review_article(request: ReviewImportRequest):
    """用户手动导入 AI 生成的文章，更新今日记录"""
    try:
        connection = get_db_connection()
        today = date.today().isoformat()
        try:
            with connection.cursor() as cursor:
                # 检查记录是否存在
                cursor.execute("SELECT id FROM review_articles WHERE review_date = %s", (today,))
                row = cursor.fetchone()
                
                if row:
                    # 存在的记录直接 UPDATE，保留原来的 words_json（锁定列表）
                    sql = "UPDATE review_articles SET title = %s, content = %s, article_type = %s WHERE id = %s"
                    cursor.execute(sql, (request.title, request.content, request.article_type, row['id']))
                else:
                    # 如果不存在（比如用户跳过了 today 接口），则创建并根据请求更新
                    # 但通常Today已经创建了占位，这里作为兜底
                    sql = "INSERT INTO review_articles (review_date, title, content, article_type, words_json) VALUES (%s, %s, %s, %s, %s)"
                    words_ids = request.words_ids or [] # 如果有传就用传的
                    cursor.execute(sql, (today, request.title, request.content, request.article_type, json.dumps(words_ids)))
                
                connection.commit()
            return {"status": "success"}
        finally:
            connection.close()
    except Exception as e:
        print(f"Import Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
