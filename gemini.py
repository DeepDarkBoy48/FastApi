from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import json
import base64
from dotenv import load_dotenv
from schemas import (
    AnalysisResult, AnalysisRequest, ModelLevel,
    DictionaryResult, LookupRequest,
    WritingResult, WritingRequest, WritingMode,
    ChatRequest, TTSRequest, TTSResponse
)

try:
    load_dotenv()
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Warning: GEMINI_API_KEY not found in environment. AI features will fail.")
    api_key = "MISSING_API_KEY"

client = genai.Client(api_key=api_key)

# --- Existing Subtitle Logic ---
class SubtitleItem(BaseModel):
    start: str = Field(description="Original start timestamp string (e.g., '00:01:23.456'). Do NOT convert to seconds.")
    end: str = Field(description="Original end timestamp string (e.g., '00:01:25.789'). Do NOT convert to seconds.")
    text: str = Field(description="Complete, merged sentence text.")

class SubtitlesResponse(BaseModel):
    subtitles: List[SubtitleItem]

async def get_response(prompt):
    response = await client.aio.models.generate_content(
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
    return response.parsed.subtitles

# --- SmashEnglish Logic ---

def get_model_config(level: ModelLevel):
    if level == 'mini':
        return 'gemini-2.5-flash-lite', 0
    elif level == 'quick':
        return 'gemini-2.5-flash-lite', 250
    elif level == 'deep':
        return 'gemini-2.5-flash-lite', 500
    else:
        return 'gemini-2.5-flash-lite', 0

async def analyze_sentence_service(sentence: str, model_level: ModelLevel) -> AnalysisResult:
    model, thinking_budget = get_model_config(model_level)
    
    prompt = f"""
    **CRITICAL RULE**: You are a linguistic expert teaching Chinese students. 
    ALL explanations, descriptions, and analysis text MUST be in **Simplified Chinese (简体中文)**.
    Only the English sentence itself and specific English terms being analyzed should remain in English.

    Please analyze the following English sentence: "{sentence}"

    **Processing Steps (Thinking Process):**
    1.  **Grammar Check (语法检查)**: 
        - Check for grammar errors.
        - If found, create a corrected version.
        - **Note**: All subsequent analysis (chunks, detailedTokens, structure) must be based on the **Corrected** sentence.
        - **Diff Generation**:
          - 'remove': Text to be removed (exact substring).
          - 'add': New text to add.
          - 'keep': Unchanged text.

    2.  **Macro Analysis (宏观结构)**:
        - Identify Core Pattern (句型结构). Format: "English Pattern (中文名称)". 
          - **CRITICAL**: Use HIGH-LEVEL patterns only (S+V, S+V+O, S+V+P, S+V+IO+DO, S+V+O+OC). 
          - Do NOT include "Modal Verb", "Auxiliary", or specific verb types in the pattern. 
          - Example: "S + V + O (主谓宾)", NOT "S + Modal + V + O".
        - Identify Core Tense (核心时态). Format: "English Tense (中文名称)". Example: "Present Simple (一般现在时)".

    3.  **Chunking (意群分块)**:
        - Group words into sense groups (rhythm chunks).
        - Modifiers with heads, prepositions with objects, verb phrases together.

    4.  **Detailed Analysis (逐词详解)**:
        - **Core Rule - Fixed Phrases**:
          - Phrasal verbs, idioms, collocations MUST be treated as a single Token. DO NOT SPLIT.
          - Example: "look forward to", "take care of".
          - **Separable Phrasal Verbs**: If "pop us back", identify "pop back".
        - **Explanation (解释)**: **MUST BE IN CHINESE**. Explain the function and form.
          - Example: "过去分词，与 has 构成现在完成时，表示动作已完成".
        - **Meaning (含义)**: Chinese translation in context.

    Return strictly JSON.
    """

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AnalysisResult,
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget) if thinking_budget > 0 else None,
            )
        )
        
        if not response.parsed:
             raise ValueError("Empty response from Gemini")
             
        result = response.parsed
        # Match frontend logic: use corrected sentence if available
        if result.correction:
            result.englishSentence = result.correction.corrected
        else:
            result.englishSentence = sentence
            
        return result
    except Exception as e:
        print(f"Gemini API Error: {e}")
        raise Exception("无法分析该句子。请检查网络或 API Key 设置。")


async def lookup_word_service(word: str, model_level: ModelLevel) -> DictionaryResult:
    model, thinking_budget = get_model_config(model_level)

    prompt = f"""
    Act as a professional learner's dictionary specifically tailored for students preparing for **IELTS, TOEFL, and CET-6**.
    User Look-up Query: "{word}".
    
    **STEP 1: Normalization & Generalization (CRITICAL)**
    1. Analyze the user's input. Is it a specific instance of a phrasal verb or collocation with specific pronouns?
    2. If yes, convert it to the **Canonical Form** (Headword).
       - Input: "pop us back" -> Output: "pop sth back"
       - Input: "made up my mind" -> Output: "make up one's mind"
    
    **STEP 2: Filtering & Content Generation**
    1. **Target Audience**: Students preparing for exams (IELTS, TOEFL, CET-6) and daily communication.
    2. **Filtering Rule**: 
       - OMIT rare, archaic, obsolete, or highly technical scientific definitions unless the word itself is technical.
       - Focus ONLY on the most common 3-4 meanings used in modern English and exams.
    3. **COCA Frequency per Part of Speech**:
       - For each part of speech (e.g. Noun vs Verb), estimate its specific COCA frequency rank.
       - Example: "address" might be "Rank 1029" as a Noun, but "Rank 1816" as a Verb.
       - Provide a concise string like "Rank 1029" or "Top 2000".

    **STEP 3: Structure**
    - Definitions: Clear, simple English explanation + Concise Chinese meaning.
    - Examples: Must be natural, modern, and relevant to exam contexts or daily life.
    
    **STEP 4: Collocations & Fixed Phrases**
    - Identify 3-5 high-frequency collocations, idioms, or fixed phrases containing this word.
    - Prioritize phrases useful for IELTS/TOEFL writing or speaking.
    - Provide meaning and a sentence example for each.

    Structure the response by Part of Speech (POS).
    Return strictly JSON.
    """

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DictionaryResult,
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget) if thinking_budget > 0 else None,
            )
        )
        
        if not response.parsed:
             raise ValueError("Empty response from Gemini")
        
        return response.parsed
    except Exception as e:
        print(f"Dictionary API Error: {e}")
        raise Exception("无法查询该单词，请重试。")


async def evaluate_writing_service(text: str, mode: WritingMode, model_level: ModelLevel) -> WritingResult:
    model, thinking_budget = get_model_config(model_level)

    mode_instructions = """
    **MODE: BASIC CORRECTION (基础纠错)**
    - Target: General accuracy.
    - Task: Focus STRICTLY on correcting grammar, spelling, punctuation, and serious awkwardness.
    - Do NOT change style, tone, or vocabulary unless it is incorrect.
    - Keep the output very close to the original, only fixing errors.
    """

    prompt = f"""
    Act as a professional English Editor and IELTS Examiner.
    
    {mode_instructions}

    **Task**:
    Analyze the user's text and reconstruct it into the *Improved Version* according to the selected mode.
    You must return the result as a sequence of SEGMENTS that allow us to reconstruct the full text while highlighting exactly what changed.

    **Input Text**: "{text}"

    **Output Logic**:
    - Iterate through the improved text.
    - If a part of the text is the same as original, mark it as 'unchanged'.
    - If you changed, added, or removed something, create a segment of type 'change'.
      - 'text': The NEW/IMPROVED text.
      - 'original': The ORIGINAL text that was replaced (or empty string if added).
      - 'reason': A brief explanation in Chinese.
      - 'category': One of 'grammar', 'vocabulary', 'style', 'punctuation', 'collocation' | 'punctuation'.
    - **CRITICAL - PARAGRAPH PRESERVATION**: 
      - You MUST preserve all paragraph breaks and newlines (\\n) from the original text exactly as they are.
      - When you encounter a newline in the original text, return it as a separate segment: {{ "text": "\\n", "type": "unchanged" }}.
      - Do NOT merge paragraphs.
    
    **Example**:
    Original: "I go store.\\n\\nIt was fun."
    Improved: "I went to the store.\\n\\nIt was fun."
    Segments:
    [
      {{ "text": "I ", "type": "unchanged" }},
      {{ "text": "went", "original": "go", "type": "change", "reason": "Past tense", "category": "grammar" }},
      {{ "text": " to the ", "original": "", "type": "change", "reason": "Preposition", "category": "grammar" }},
      {{ "text": "store.", "type": "unchanged" }},
      {{ "text": "\\n\\n", "type": "unchanged" }},
      {{ "text": "It was fun.", "type": "unchanged" }}
    ]

    Return strictly JSON.
    """
    
    # Define a partial schema for response to match WritingResult structure but without 'mode' which we set manually
    class WritingResponseSchema(BaseModel):
        generalFeedback: str
        segments: List[WritingResult.model_fields['segments'].annotation]

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=WritingResult, # Using the full WritingResult schema, hoping Gemini fills 'mode' or we override it
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget) if thinking_budget > 0 else None,
            )
        )

        if not response.parsed:
            raise ValueError("Empty response")
            
        result = response.parsed
        # Ensure mode matches request
        result.mode = mode
        return result

    except Exception as e:
        print(f"Writing Evaluation API Error: {e}")
        raise Exception("写作分析失败，请检查网络或稍后再试。")


async def chat_service(request: ChatRequest) -> str:
    context_instruction = ""
    if request.contextType == 'sentence':
         context_instruction = f'**当前正在分析的句子**: "{request.contextContent or "用户暂未输入句子"}"。'
    elif request.contextType == 'word':
         context_instruction = f'**当前正在查询的单词/词组**: "{request.contextContent or "用户暂未查询单词"}"。'
    elif request.contextType == 'writing':
         context_instruction = f'**当前正在润色的文章**: "{request.contextContent or "用户暂未输入文章"}"。'

    system_instruction = f"""
        你是一个热情、专业的英语学习助教。
        
        {context_instruction}
        
        **你的任务**：
        1. 解答用户关于英语语法、单词用法、句子结构或词汇辨析的问题。
        2. **始终使用中文**回答。
        3. 使用 **Markdown** 格式来美化你的回答，使其清晰易读：
           - 使用 **加粗** 来强调重点单词或语法术语。
           - 使用列表（1. 或 -）来分点解释。
           - 适当分段。
        4. 语气要鼓励、积极，像一位耐心的老师。
        5. **特殊指令**：如果用户询问类似 "pop us back" 这样的短语，请解释这是一种口语表达，核心是短语动词 "pop back" (迅速回去)，"us" 是宾语。
    """
    
    # Reconstruct history for Gemini
    # Gemini python SDK expects a slightly different history format if using chat.sendMessage
    # But here we might just do a single turn generation with history context if we want to be stateless, 
    # OR use the chat session. Given FastAPI is stateless, we should probably pass the history.
    # However, the `google.genai` SDK `chats.create` creates a session. 
    # We can manually construct the `contents` list from history + new message.
    
    contents = []
    for msg in request.history:
        # 🔥 关键修复：将 'assistant' 转换为 Gemini 期望的 'model'
        role = 'model' if msg.role == 'assistant' else msg.role
        contents.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))
    
    # Add user's new message
    contents.append(types.Content(role='user', parts=[types.Part(text=request.userMessage)]))

    try:
        response = await client.aio.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        return response.text
    except Exception as e:
        print(f"Chat API Error: {e}")
        raise Exception("聊天服务暂时不可用。")


async def generate_speech_service(text: str) -> str:
    try:
        # Use a model that supports audio generation (e.g., gemini-2.0-flash-exp)
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=['AUDIO'],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Kore')
                    )
                )
            )
        )

        # Accessing audio data from response
        # The SDK returns binary data in candidates[0].content.parts[0].inline_data.data
        
        if not response.candidates or not response.candidates[0].content.parts:
             raise ValueError("No content returned")
             
        part = response.candidates[0].content.parts[0]
        if not part.inline_data or not part.inline_data.data:
             raise ValueError("No audio data returned")
             
        # inline_data.data is bytes. We need to return base64 string for JSON response.
        return base64.b64encode(part.inline_data.data).decode('utf-8')

    except Exception as e:
        print(f"TTS API Error: {e}")
        raise Exception("语音生成失败。")
