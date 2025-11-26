from fastapi import FastAPI
from pydantic import BaseModel
import yt_dlp
import os
import glob
import uuid
import gemini

app = FastAPI()

@app.get("/fastapi/debug")
def debug_info():
    """Debug endpoint to check cookies file status"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cookies_path = os.path.join(base_dir, "cookies.txt")
    
    # 获取 cookies 文件信息
    cookies_info = {}
    if os.path.exists(cookies_path):
        stat = os.stat(cookies_path)
        cookies_info = {
            "size_bytes": stat.st_size,
            "modified_time": stat.st_mtime
        }
    
    return {
        "cookies_path": cookies_path,
        "cookies_exists": os.path.exists(cookies_path),
        "cookies_info": cookies_info,
        "current_dir": os.getcwd(),
        "files_in_app": os.listdir(base_dir)
    }

@app.get("/fastapi/getsubtitles")
def get_subtitles(youtubeUrl: str):
    video_id = str(uuid.uuid4())
    temp_dir = "/tmp"
    output_template = os.path.join(temp_dir, f"{video_id}.%(ext)s")
    
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'outtmpl': output_template,
        'quiet': True,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
    }

    # Check for cookies.txt to bypass bot detection
    # Use absolute path for Docker compatibility
    cookies_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cookies.txt")
    if os.path.exists(cookies_path):
        ydl_opts['cookiefile'] = cookies_path

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtubeUrl])
        
        # Find the downloaded subtitle file
        # yt-dlp might append language code, e.g., .en.vtt
        files = glob.glob(os.path.join(temp_dir, f"{video_id}*.vtt"))
        
        if not files:
            return {"error": "No subtitles found"}
            
        # Pick the first one for now, or prefer English/Chinese if multiple
        subtitle_file = files[0]
        
        with open(subtitle_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        #use gemini to format
        formatted_content = gemini.get_response(content)

        # Clean up
        for f in files:
            os.remove(f)
            
        return {"subtitles": formatted_content}
        
    except Exception as e:
        return {"error": str(e)}