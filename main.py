from fastapi import FastAPI
from pydantic import BaseModel
import yt_dlp
import os
import glob
import uuid
import gemini

app = FastAPI()

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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    }

    # Check for cookies.txt to bypass bot detection
    if os.path.exists("cookies.txt"):
        ydl_opts['cookiefile'] = "cookies.txt"

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