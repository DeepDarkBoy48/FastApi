from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List
from pydantic import Field

import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 2. 优化：加回 Field 描述，增强 AI 理解
class SubtitleItem(BaseModel):
    # 改为字符串，让 AI 原样返回 VTT 的时间格式
    start: str = Field(description="Original start timestamp string (e.g., '00:01:23.456'). Do NOT convert to seconds.")
    end: str = Field(description="Original end timestamp string (e.g., '00:01:25.789'). Do NOT convert to seconds.")
    text: str = Field(description="Complete, merged sentence text.")

class SubtitlesResponse(BaseModel):
    subtitles: List[SubtitleItem]

def get_response(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SubtitlesResponse,
            thinking_config=types.ThinkingConfig(
                thinking_budget=0
            ),
            system_instruction="""
                    # Role
                    You are an expert subtitle editor. 
                    
                    # Task
                    1. Merge fragmented sentences into complete sentences based on context.
                    2. Deduplicate repeating lines.
                    3. Keep timestamps in their **ORIGINAL format** (HH:MM:SS.mmm).
                    
                    # Rules
                    - strictly maintain the timeline sequence.
                    - Start time = start timestamp of the first fragment.
                    - End time = end timestamp of the last fragment.
                    - Do NOT convert times to math/floats. Just copy the string.
            """),
    )
    # response.parsed is the Pydantic object (SubtitlesResponse)
    return response.parsed.subtitles
