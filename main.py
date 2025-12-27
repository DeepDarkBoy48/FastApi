from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import gemini
import pymysql
import json
from datetime import date
from schemas import (
    AnalysisRequest, AnalysisResult,
    LookupRequest, DictionaryResult,
    WritingRequest, WritingResult,
    ChatRequest, ChatResponse,
    QuickLookupRequest, QuickLookupResult,
    RapidLookupRequest, RapidLookupResult,
    TranslateRequest, TranslateResult,
    SavedWord, SavedWordsResponse,
    DailyNote, DailyNotesResponse, NoteDetailResponse,
    VideoNotebook, VideoNotebookCreate, VideoNotebookListResponse, VideoNotebookUpdate
)

app = FastAPI()

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

def save_word_to_db(word: str, context: str, data: dict, url: str = None):
    """Helper function to save word lookup result to database and link to daily note"""
    try:
        connection = get_db_connection()
        today = date.today().isoformat()
        try:
            with connection.cursor() as cursor:
                # 1. 检查今天是否有 note 记录
                cursor.execute("SELECT id FROM daily_notes WHERE day = %s", (today,))
                row = cursor.fetchone()
                
                if row:
                    note_id = row['id']
                    # 更新单词数
                    cursor.execute("UPDATE daily_notes SET word_count = word_count + 1 WHERE id = %s", (note_id,))
                else:
                    # 创建今天的 note
                    title = f"{today} 的单词卡片"
                    cursor.execute(
                        "INSERT INTO daily_notes (day, title, word_count) VALUES (%s, %s, %s)",
                        (today, title, 1)
                    )
                    note_id = cursor.lastrowid
                
                # 2. 插入单词，带上 note_id 和 url
                sql = "INSERT INTO saved_words (word, context, url, data, note_id) VALUES (%s, %s, %s, %s, %s)"
                cursor.execute(sql, (word, context, url, json.dumps(data, ensure_ascii=False), note_id))
            
            connection.commit()
        finally:
            connection.close()
    except Exception as e:
        print(f"Database Save Error: {e}")



# --- SmashEnglish Endpoints ---

@app.post("/fastapi/analyze", response_model=AnalysisResult)
async def analyze_sentence(request: AnalysisRequest):
    try:
        result = await gemini.analyze_sentence_service(request.sentence)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fastapi/lookup", response_model=DictionaryResult)
async def lookup_word(request: LookupRequest):
    try:
        result = await gemini.lookup_word_service(request.word)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fastapi/writing", response_model=WritingResult)
async def evaluate_writing(request: WritingRequest):
    try:
        result = await gemini.evaluate_writing_service(request.text, request.mode)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fastapi/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response_text = await gemini.chat_service(request)
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fastapi/quick-lookup", response_model=QuickLookupResult)
async def quick_lookup(request: QuickLookupRequest):
    """快速上下文查词 - 返回词条并自动保存到数据库"""
    try:
        result = await gemini.quick_lookup_service(request.word, request.context)
        # 异步/后台保存
        save_word_to_db(request.word, request.context, result.model_dump(), request.url)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fastapi/rapid-lookup", response_model=RapidLookupResult)
async def rapid_lookup(request: RapidLookupRequest):
    """极简上下文查词 - 极致速度 (不保存数据库)"""
    try:
        result = await gemini.rapid_lookup_service(request.word, request.context)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fastapi/translate", response_model=TranslateResult)
async def translate_endpoint(request: TranslateRequest):
    """极速翻译接口"""
    try:
        result = await gemini.translate_service(request.text)
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

                # 2. 查找该 Note 下的所有单词
                cursor.execute("SELECT * FROM saved_words WHERE note_id = %s ORDER BY created_at DESC", (note_id,))
                word_rows = cursor.fetchall()
                words = []
                for row in word_rows:
                    data_raw = row['data']
                    data_parsed = json.loads(data_raw) if isinstance(data_raw, str) else data_raw
                    words.append(SavedWord(
                        id=row['id'],
                        word=row['word'],
                        context=row['context'],
                        url=row.get('url'),
                        data=data_parsed,
                        created_at=row['created_at'].strftime('%Y-%m-%d %H:%M:%S') if row['created_at'] else "",
                        note_id=row['note_id']
                    ))
                
                return NoteDetailResponse(note=note, words=words)
        finally:
            connection.close()
    except Exception as e:
        print(f"Database Get Note Detail Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fastapi/daily-notes/{note_id}/summarize")
async def summarize_daily_note(note_id: int):
    """为当天的笔记生成 AI 总结博客 (更新标题、简介和内容)"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                # 1. 获取所有相关的单词
                cursor.execute("SELECT * FROM saved_words WHERE note_id = %s", (note_id,))
                words_raw = cursor.fetchall()
                if not words_raw:
                    raise HTTPException(status_code=400, detail="No words to summarize")
                
                # 处理 JSON 数据字段
                words = []
                for w in words_raw:
                    if isinstance(w['data'], str):
                        w['data'] = json.loads(w['data'])
                    words.append(w)
                
                # 2. 调用 Gemini 生成结构化内容
                blog_result = await gemini.generate_daily_summary_service(words)
                
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
                # 先获取 note_id 方便更新 count
                cursor.execute("SELECT note_id FROM saved_words WHERE id = %s", (word_id,))
                row = cursor.fetchone()
                if row and row['note_id']:
                    note_id = row['note_id']
                    cursor.execute("DELETE FROM saved_words WHERE id = %s", (word_id,))
                    cursor.execute("UPDATE daily_notes SET word_count = GREATEST(0, word_count - 1) WHERE id = %s", (note_id,))
                else:
                    cursor.execute("DELETE FROM saved_words WHERE id = %s", (word_id,))
            connection.commit()
            return {"status": "success"}
        finally:
            connection.close()
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
                    data_raw = row['data']
                    data_parsed = json.loads(data_raw) if isinstance(data_raw, str) else data_raw
                    words.append(SavedWord(
                        id=row['id'],
                        word=row['word'],
                        context=row['context'],
                        url=row.get('url'),
                        data=data_parsed,
                        created_at=row['created_at'].strftime('%Y-%m-%d %H:%M:%S') if row['created_at'] else "",
                        note_id=row['note_id']
                    ))
                return SavedWordsResponse(words=words)
        finally:
            connection.close()
    except Exception as e:
        print(f"Database Get All Words Error: {e}")
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

