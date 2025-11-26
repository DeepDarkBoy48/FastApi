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
    return response.parsed.subtitles

# --- SmashEnglish Logic ---

def get_model_config(level: ModelLevel):
    if level == 'mini':
        return 'gemini-2.5-flash', 0
    elif level == 'quick':
        return 'gemini-2.5-flash', 500
    elif level == 'deep':
        return 'gemini-2.5-flash', 2000
    else:
        return 'gemini-2.5-flash', 0

def analyze_sentence_service(sentence: str, model_level: ModelLevel) -> AnalysisResult:
    model, thinking_budget = get_model_config(model_level)
    
    prompt = f"""
    你是一位精通语言学和英语教学的专家 AI。请分析以下英语句子： "{sentence}"。
    目标受众是正在学习英语的学生，因此分析需要**清晰、准确且具有教育意义**。

    **Processing Steps (Thinking Process):**
    1.  **Grammar Check (纠错)**: 
        - 仔细检查句子是否有语法错误。
        - 如果有错，创建一个修正后的版本。
        - **注意**：后续的所有分析（chunks, detailedTokens, structure）必须基于**修正后(Corrected)** 的句子进行。
        - **Diff Generation**: 生成 'changes' 数组时，必须是严格的文本差异对比 (diff)。
          - 'remove': 仅包含被删除的原文片段，**绝对不要**包含 "->" 符号或 "change x to y" 这样的描述。例如原句是 "i go"，修正为 "I go"，则 'remove' text 为 "i"，'add' text 为 "I"。
          - 'add': 仅包含新加入的片段。
          - 'keep': 保持不变的部分。

    2.  **Macro Analysis (宏观结构)**:
        - 识别核心句型结构 (Pattern)，**必须包含中文翻译**。格式要求："English Pattern (中文名称)"。例如："S + V + O (主谓宾)"。
        - 识别核心时态 (Tense)，**必须包含中文翻译**。格式要求："English Tense (中文名称)"。例如："Present Simple (一般现在时)"。

    3.  **Chunking (可视化意群分块)**:
        - 目标是展示句子的“节奏”和“意群”(Sense Groups)。
        - **原则**：
          - 所有的修饰语应与其中心词在一起（例如 "The very tall man" 是一个块）。
          - 介词短语通常作为一个整体（例如 "in the morning" 是一个块）。
          - 谓语动词部分合并（例如 "have been waiting" 是一个块）。
          - 不定式短语合并（例如 "to go home" 是一个块）。

    4.  **Detailed Analysis (逐词/短语详解)**:
        - **核心原则 - 固定搭配优先**：
          - 遇到短语动词 (phrasal verbs)、固定习语 (idioms)、介词搭配 (collocations) 时，**必须**将它们作为一个整体 Token，**绝对不要拆分**。
          - 例如："look forward to", "take care of", "a cup of", "depend on"。
          - **特别处理可分离短语动词 (Separable Phrasal Verbs)**：
            - 如果遇到像 "pop us back", "turn it on" 这样动词与小品词被代词隔开的情况，请务必**识别出其核心短语动词**（如 "pop back"）。
            - 在详细解释 (explanation) 中，**必须**明确指出该词属于短语动词 "pop back" (或相应短语)，并解释该短语动词的含义，而不仅仅是单个单词的意思。
            - 示例：针对 "pop us back"，在解释 "pop" 时，应说明 "pop ... back 是短语动词，意为迅速回去/放回"。
        - **解释 (Explanation)**：
          - 不要只给一个词性标签。要解释它在句子中的**功能**和**为什么用这种形式**。
          - 例如：不要只写"过去分词"，要写"过去分词，与 has 构成现在完成时，表示动作已完成"。
        - **含义 (Meaning)**：提供在当前语境下的中文含义。

    请返回 JSON 格式数据。
    """

    try:
        response = client.models.generate_content(
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


def lookup_word_service(word: str, model_level: ModelLevel) -> DictionaryResult:
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
        response = client.models.generate_content(
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


def evaluate_writing_service(text: str, mode: WritingMode, model_level: ModelLevel) -> WritingResult:
    model, thinking_budget = get_model_config(model_level)

    mode_instructions = ""
    if mode == 'fix':
        mode_instructions = """
        **MODE: BASIC CORRECTION (基础纠错)**
        - Target: General accuracy.
        - Task: Focus STRICTLY on correcting grammar, spelling, punctuation, and serious awkwardness.
        - Do NOT change style, tone, or vocabulary unless it is incorrect.
        - Keep the output very close to the original, only fixing errors.
        """
    elif mode == 'ielts-5.5':
        mode_instructions = """
        **MODE: IELTS BAND 5.5 (Modest User)**
        - Target Level: Partial command of the language.
        - Task: Correct all basic errors. Ensure the overall meaning is clear.
        - Style: Keep vocabulary simple but correct. Avoid complex structures if they risk error.
        - Feedback focus: Basic grammar and clarity.
        """
    elif mode == 'ielts-6.0':
        mode_instructions = """
        **MODE: IELTS BAND 6.0 (Competent User)**
        - Target Level: Generally effective command.
        - Task: Use a mix of simple and complex sentence forms. Correct errors.
        - Style: Use adequate vocabulary. Ensure coherence.
        """
    elif mode == 'ielts-6.5':
        mode_instructions = """
        **MODE: IELTS BAND 6.5 (Between Competent and Good)**
        - Target Level: Stronger competence.
        - Task: Introduce more complex structures. Enhance vocabulary slightly beyond basic.
        - Style: Improve flow and linking words.
        """
    elif mode == 'ielts-7.0':
        mode_instructions = """
        **MODE: IELTS BAND 7.0 (Good User)**
        - Target Level: Operational command, occasional inaccuracies.
        - Task: Use a variety of complex structures. Use less common lexical items.
        - Style: Academic and formal. Show awareness of style and collocation.
        """
    elif mode == 'ielts-7.5':
        mode_instructions = """
        **MODE: IELTS BAND 7.5 (Very Good User)**
        - Target Level: High accuracy.
        - Task: Sophisticated control of vocabulary and grammar. Minimize errors to very occasional slips.
        - Style: Highly polished, natural, and academic.
        """
    elif mode == 'ielts-8.0':
        mode_instructions = """
        **MODE: IELTS BAND 8.0 (Expert-like User)**
        - Target Level: Fully operational command.
        - Task: Use a wide range of vocabulary fluently and flexibly to convey precise meanings.
        - Style: Skillful use of uncommon lexical items. Error-free sentences. Native-like flow.
        """
    else:
        mode_instructions = "**MODE: BASIC CORRECTION**"

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
        response = client.models.generate_content(
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


def chat_service(request: ChatRequest) -> str:
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
        contents.append(types.Content(role=msg.role, parts=[types.Part(text=msg.content)]))
    
    # Add user's new message
    contents.append(types.Content(role='user', parts=[types.Part(text=request.userMessage)]))

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        return response.text
    except Exception as e:
        print(f"Chat API Error: {e}")
        raise Exception("聊天服务暂时不可用。")


def generate_speech_service(text: str) -> str:
    try:
        response = client.models.generate_content(
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
        # But `response.text` won't work.
        # We need to access the part directly.
        
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
