from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import gemini
import pymysql
from pymysql.cursors import DictCursor
import json
from datetime import date, datetime, timedelta, timezone
import math
from fsrs import Scheduler, Card, Rating, State
from dotenv import load_dotenv
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
    VideoNotebook, VideoNotebookCreate, VideoNotebookListResponse, VideoNotebookUpdate,
    TodayReviewResponse, ReviewArticle, FSRSFeedbackRequest,
    ReviewPromptResponse, ReviewImportRequest
)


app = FastAPI()

# Database configuration
DB_CONFIG = {
    'host': '47.79.43.73',
    # 'host': 'mysql-container',  
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
                # 0. 检查单词是否已存在 (避免重复保存)
                cursor.execute("SELECT id FROM saved_words WHERE word = %s", (word,))
                if cursor.fetchone():
                    print(f"Word '{word}' already exists in database. Skipping insertion.")
                    return

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

# --- FSRS Implementation ---
fsrs = Scheduler()

def get_fsrs_rating(rating: int) -> Rating:
    """将前端 1-4 档转换为官方 Rating 枚举"""
    if rating == 1: return Rating.Again
    if rating == 2: return Rating.Hard
    if rating == 3: return Rating.Good
    return Rating.Easy

def format_saved_word(row):
    """Utility to format DB row to SavedWord schema"""
    data_val = row['data']
    try:
        parsed_data = json.loads(data_val) if isinstance(data_val, str) else data_val
    except:
        parsed_data = {}
    
    return SavedWord(
        id=row['id'],
        word=row['word'],
        context=row['context'],
        url=row['url'],
        data=parsed_data,
        created_at=row['created_at'].strftime('%Y-%m-%d %H:%M:%S') if row['created_at'] else None,
        note_id=row['note_id'],
        stability=row['stability'],
        difficulty=row['difficulty'],
        elapsed_days=row['elapsed_days'],
        scheduled_days=row['scheduled_days'],
        last_review=row['last_review'].strftime('%Y-%m-%d %H:%M:%S') if row['last_review'] else None,
        reps=row['reps'],
        state=row['state']
    )




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
                    words.append(format_saved_word(row))
                
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
                    words.append(format_saved_word(row))
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
                for r in word_rows:
                    data = json.loads(r['data']) if isinstance(r['data'], str) else r['data']
                    meaning = data.get('contextMeaning') or data.get('m') or '未知'
                    words_info += f"- **{r['word']}**: {meaning}\n  原句: \"{r['context']}\"\n\n"

                # 提取完整 ID 数组用于嵌入 JSON 结构
                word_ids_str = ", ".join([str(r['id']) for r in word_rows])

                prompt = f"""
你是一位天才内容创作者，擅长编写极具吸引力的英语学习内容。
今天你需要根据用户复习的 {len(word_rows)} 个单词，编写一篇文章。形式候选：播客、采访、辩论、深度博客、新闻特写。

## 待包含的单词及其背景
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

@app.post("/fastapi/review/feedback")
async def review_feedback(request: FSRSFeedbackRequest):
    """用户提交复习反馈，更新 FSRS 数值"""
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                # 1. 获取单词当前状态
                cursor.execute("SELECT * FROM saved_words WHERE id = %s", (request.word_id,))
                word = cursor.fetchone()
                if not word:
                    raise HTTPException(status_code=404, detail="Word not found")

                # 2. 检查单日单次限制
                now = datetime.now()
                if word['last_review'] and word['last_review'].date() == now.date():
                    return {"message": "Already reviewed today", "id": request.word_id}

                # 3. FSRS 计算
                rating = request.rating # 1-3
                # 映射用户评分到 FSRS 4 级 (1: Again, 2: Hard, 3: Good, 4: Easy)
                # 用户只有三个选项，映射为: 记得 -> 3(Good), 完全熟悉 -> 4(Easy), 忘记 -> 1(Again)
                fsrs_rating = 1
                if rating == 2: fsrs_rating = 3
                if rating == 3: fsrs_rating = 4

                if word['reps'] == 0 or word['last_review'] is None:
                    # 第一次复习 (新词)
                    new_stability = FSRSCore.init_stability(fsrs_rating)
                    new_difficulty = FSRSCore.init_difficulty(fsrs_rating)
                else:
                    # 后续复习
                    elapsed = (now - word['last_review']).days
                    r = math.pow(0.9, elapsed / word['stability']) if word['stability'] > 0 else 0
                    new_stability = FSRSCore.next_stability(word['stability'], word['difficulty'], r, fsrs_rating)
                    new_difficulty = FSRSCore.next_difficulty(word['difficulty'], fsrs_rating)

                new_interval = FSRSCore.next_interval(new_stability)
                
                # 更新数据库
                cursor.execute("""
                    UPDATE saved_words SET 
                    stability = %s, 
                    difficulty = %s, 
                    elapsed_days = DATEDIFF(NOW(), IFNULL(last_review, created_at)), 
                    scheduled_days = %s,
                    last_review = %s,
                    reps = reps + 1,
                    state = %s
                    WHERE id = %s
                """, (new_stability, new_difficulty, new_interval, now, 2 if fsrs_rating > 1 else 1, request.word_id))
                
            connection.commit()
            return {"status": "success", "next_review_days": new_interval}
        finally:
            connection.close()
    except Exception as e:
        print(f"Feedback Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

