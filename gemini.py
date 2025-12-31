from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import json
from dotenv import load_dotenv
from schemas import (
    AnalysisResult, AnalysisRequest,
    DictionaryResult, LookupRequest,
    WritingResult, WritingRequest, WritingMode,
    ChatRequest,
    QuickLookupResult,
    RapidLookupResult,
    TranslateResult,
    BlogSummaryResult,
    ReviewArticle
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

# --- SmashEnglish Logic ---

# --- Model Configuration Central ---

DEFAULT_MODEL = 'gemini-3-flash-preview'
CHEAP_MODEL = 'gemini-2.5-flash-lite'


def get_analysis_config():
    """句子语法分析卡片模式"""
    return DEFAULT_MODEL, 'low'

def get_dictionary_config():
    """详细词典查询模式"""
    return DEFAULT_MODEL, 'minimal'

def get_writing_config():
    """写作润色与评分模式"""
    return DEFAULT_MODEL, 'low'

def get_chat_config():
    """AI 助教对话模式"""
    return DEFAULT_MODEL, 'minimal'

def get_lookup_config():
    """上下文查词卡片模式 (含快速与极速)"""
    return DEFAULT_MODEL, 'minimal'

def get_translate_config():
    """全文/句子极速翻译模式"""
    return DEFAULT_MODEL, 'minimal'

def get_crawl_config():
    """网页抓取与排版模式 - 使用最便宜的 Lite 模型"""
    return LITE_MODEL, 'minimal'



async def analyze_sentence_service(sentence: str) -> AnalysisResult:
    model, thinking_level = get_analysis_config()
    
    prompt = f"""
    你是一位精通语言学和英语教学的专家 AI。请分析以下英语句子： "{sentence}"。
    目标受众是正在学习英语的学生，因此分析需要**清晰、准确且具有教育意义**。

    **Language Constraint (语言约束)**:
    - 所有的 `role` (角色) 和 `partOfSpeech` (词性) 字段**必须且只能使用简体中文**。
    - 严禁输出 "Noun", "Verb", "Subject", "Object", "Attribute", "Predicate" 等英文术语。
    - 示例词性： "名词", "动词", "形容词", "副词", "介词", "代词", "连词", "限定词", "分词", "动词短语", "介词短语"。
    - 示例角色： "主语", "谓语", "宾语", "表语", "状语", "定语", "补语", "宾语补足语", "同位语"。

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

    4.  **Detailed Analysis (逐词/短语详解 - 核心要求)**:
        - **全覆盖与意义分块原则 (Comprehensive & Meaningful Chunking)**:
          - 你的分析必须覆盖句子中的**所有内容**，确保没有遗漏任何语义成分。
          - **不要机械地拆分每一个单词**。如果几个词共同构成一个紧密的语义单位（如限定词+形容词+名词，或介词短语），应当将它们作为一个 Token 整体分析。
          - 示例：对于 "a new language"，应作为一个整体分析，而不是拆分为 "a", "new", "language"。
          - 示例：对于 "from a proton to the observable universe"，应根据语义节奏拆分为合理的块，如 "from a proton", "to the observable universe"，而不是逐词拆分。
          - **标点符号**：除非标点符号在语法结构上有特殊意义（如破折号、分号），否则通常不需要作为独立的 Token 进行分析。
        - **核心原则 - 固定搭配与意群优先**：
          - 遇到短语动词、习语、固定搭配、或紧密的名词短语时，**必须**将它们作为一个整体 Token。
          - 最终的 `detailedTokens` 列表按顺序拼接起来应能体现句子的完整逻辑流。
        - **标签要求 (Tags)**：
          - `partOfSpeech` (词性) 和 `role` (角色) 必须使用**简体中文**。
        - **解释 (Explanation)**：
          - 不要只给一个词性标签。要解释它在句子中的**功能**和**语义作用**。
        - **含义 (Meaning)**：提供该意群在当前语境下的中文含义。

    请返回符合 JSON 格式的数据。
    """

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AnalysisResult,
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level=thinking_level
                ) if thinking_level != 'minimal' else types.ThinkingConfig(thinking_level='minimal'),
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


async def lookup_word_service(word: str) -> DictionaryResult:
    model, thinking_level = get_dictionary_config()

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
    - Definitions: Provide a clear and concise meaning in **Simplified Chinese**. 
    - Explanation: Provide a detailed explanation of the usage, nuances, or grammatical context **EXCLUSIVELY in Simplified Chinese**. (DO NOT provide English explanations).
    - Examples: Must be natural, modern, and relevant to exam contexts or daily life.
    - Example Translation: Provide a natural translation of the example in **Simplified Chinese**.
    
    **STEP 4: Collocations & Fixed Phrases**
    - Identify 3-5 high-frequency collocations, idioms, or fixed phrases containing this word.
    - Prioritize phrases useful for IELTS/TOEFL writing or speaking.
    - Provide the meaning in **Simplified Chinese** and a sentence example with its Chinese translation for each.

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
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level=thinking_level
                ) if thinking_level != 'minimal' else types.ThinkingConfig(thinking_level='minimal'),
            )
        )
        
        if not response.parsed:
             raise ValueError("Empty response from Gemini")
        
        return response.parsed
    except Exception as e:
        print(f"Dictionary API Error: {e}")
        raise Exception("无法查询该单词，请重试。")


async def evaluate_writing_service(text: str, mode: WritingMode) -> WritingResult:
    model, thinking_level = get_writing_config()

    mode_instructions = """
    **MODE: BASIC CORRECTION (基础纠错)**
    - Target: General accuracy.
    - Task: Focus STRICTLY on correcting grammar, spelling, punctuation, and serious awkwardness.
    - Do NOT change style, tone, or vocabulary unless it is incorrect.
    - Keep the output very close to the original, only fixing errors.
    """

    prompt = f"""
    Act as a professional English Writing Coach and Editor.
    
    {mode_instructions}

    **Task**:
    Analyze the user's text and reconstruct it into the *Improved Version*.
    
    **Target Standard (CRITICAL)**:
    - **US High School Student Level**: The improved text should flow naturally like a native US high school student's writing. 
    - **Beyond Basic Grammar**: Do not just fix grammatical errors. Improve sentence structure, vocabulary choice, and flow to make it sound idiomatic and cohesive.
    - **Maintain Meaning**: Improve the expression but keep the original meaning and intent.

    **Input Text**: "{text}"

    **Output Logic**:
    1. **Overall Comment**: Provide a comprehensive summary of the writing (in Simplified Chinese). Mention the good points and the main areas for improvement (e.g., "Sentence variety", "Vocabulary depth", "Logic flow").
    2. **Segments**:
       - Iterate through the improved text.
       - If a part of the text is unchanged, mark it as 'unchanged'.
       - If you changed, added, or removed something, create a segment of type 'change'.
         - 'text': The NEW/IMPROVED text.
         - 'original': The ORIGINAL text that was replaced (or empty string if added).
         - 'reason': A specific, educational explanation in **Simplified Chinese**. Explain WHY the change improves the text (e.g., "Change 'happy' to 'elated' for better vocabulary", "Combine sentences for better flow").
         - 'category': One of 'grammar', 'vocabulary', 'style', 'punctuation', 'collocation', 'flow'.
    
    **CRITICAL - PARAGRAPH PRESERVATION**: 
    - You MUST preserve all paragraph breaks and newlines (\\n) from the original text exactly as they are.
    - When you encounter a newline in the original text, return it as a separate segment: {{ "text": "\\n", "type": "unchanged" }}.
    - Do NOT merge paragraphs.

    **Example**:
    Original: "I go store today. It big."
    Improved: "I went to the store today. It was huge."
    Segments:
    [
      {{ "text": "I ", "type": "unchanged" }},
      {{ "text": "went", "original": "go", "type": "change", "reason": "时态修正：应使用过去时", "category": "grammar" }},
      {{ "text": " to the ", "original": "", "type": "change", "reason": "缺失介词和冠词", "category": "grammar" }},
      {{ "text": "store today. It was ", "type": "unchanged" }},
      {{ "text": "huge", "original": "big", "type": "change", "reason": "词汇升级：'huge' 比 'big' 更具体", "category": "vocabulary" }},
      {{ "text": ".", "type": "unchanged" }}
    ]

    Return strictly JSON.
    """
    
    # Define a partial schema for response to match WritingResult structure but without 'mode' which we set manually
    class WritingResponseSchema(BaseModel):
        generalFeedback: str
        overall_comment: str
        segments: List[WritingResult.model_fields['segments'].annotation]

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=WritingResult, # Using the full WritingResult schema, hoping Gemini fills 'mode' or we override it
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level=thinking_level
                ) if thinking_level != 'minimal' else types.ThinkingConfig(thinking_level='minimal'),
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
    model, thinking_level = get_chat_config()
    context_instruction = ""
    if request.contextType == 'sentence':
         context_instruction = f'**当前正在分析的句子**: "{request.contextContent or "用户暂未输入句子"}"。'
    elif request.contextType == 'word':
         context_instruction = f'**当前正在查询的单词/词组**: "{request.contextContent or "用户暂未查询单词"}"。'
    elif request.contextType == 'writing':
         context_instruction = f'**当前正在润色的文章**: "{request.contextContent or "用户暂未输入文章"}"。'

    system_instruction = f"""
        你是一个热情、专业的英语学习助教。你现在拥有访问 **Google 搜索** 的能力，可以提供最前沿、最地道的英语用法参考。
        
        {context_instruction}
        
        **你的任务**：
        1. 解答用户关于英语语法、单词用法、句子结构或词汇辨析的问题。
        2. **利用实时搜索**：如果用户询问的是最新的网络流行语、俚语、或者涉及特定文化/时事背景的英语表达，请务必使用搜索功能来获取最准确、最新的解释和实例。
        3. **提供地道例句**：在解释词汇时，可以主动通过搜索从权威媒体（如 BBC, NYT, The Economist）中提取真实例句，帮助用户理解该词在现代英语中的实际应用。
        4. **引用来源**：如果你的回答引用了搜索结果，请根据搜索元数据提供清晰的来源链接（格式如 [标题](链接)），增加回答的可信度。
        5. **始终使用中文**回答。
        6. 使用 **Markdown** 格式来美化你的回答，使其清晰易读：
           - 使用 **加粗** 来强调重点单词或语法术语。
           - 使用列表（1. 或 -）来分点解释。
           - 适当分段。
        7. 语气要鼓励、积极，像一位耐心的老师。
        8. **特殊指令**：如果用户询问类似 "pop us back" 这样的短语，请解释这是一种口语表达，核心是短语动词 "pop back" (迅速回去)，"us" 是宾语。
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
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[types.Tool(google_search=types.GoogleSearch())],
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level=thinking_level
                ) if thinking_level != 'minimal' else types.ThinkingConfig(thinking_level='minimal'),
            )
        )
        return response.text
    except Exception as e:
        print(f"Chat API Error: {e}")
        raise Exception("聊天服务暂时不可用。")


async def quick_lookup_service(word: str, context: str) -> QuickLookupResult:
    """快速上下文查词服务 - 给出单词在上下文中的释义和解释"""
    model, thinking_level = get_lookup_config()

    prompt = f"""
    你是一位英语教学专家。请分析单词 "{word}" 在以下句子上下文中的具体含义、词性、语法成分和用法：
    
    **句子上下文**: "{context}"
    
    **任务要求**:
    1. **contextMeaning**: 给出这个词在当前上下文中的**具体中文释义**（简洁，1-2个词）。
    2. **partOfSpeech**: 给出这个词在当前语境下的**精准词性缩写**（如：及物动词 vt., 不及物动词 vi., 名词 n., 形容词 adj., 副词 adv., 介词 prep., 连词 conj. 等）。
    3. **grammarRole**: 给出这个词在句子中的**语法成分**（如：主语、谓语、宾语、定语、状语、表语、宾补、同位语等）或**固定搭配/短语**。
    4. **explanation**: 结合语境进行地道翻译与深度解析。
       - **全文翻译(必须)**: 首先给出整个原句的地道、口语化的中文翻译。
       - **用法解析**: 简要解释单词为什么是这个意思及其在句中的具体用法细节（如：是并列结构吗？指代什么？）。
       - 如果涉及固定搭配（如 "upload...to..."），请务必指出来。
       - 结合上下文背景，说明该词传达的语气或具体指代的对象。
    5. **otherMeanings**: 提供该单词的**其他常见且高频**的释义。
       - **过滤规则**: 严禁提供生僻、古僻、过于专业或罕见的释义。只保留在中高考、雅思、托福或日常口语中常见的 2-3 个其他意思。
       - 每个意思需包含 `meaning` (中文释义)、`partOfSpeech` (词性) 以及一个简短的英文例句 `example`。
    
    **输出语言**: 全部使用简体中文（释义和解释部分）。
    **输出格式**: 严格 JSON。
    
    示例输出:
    {{
      "word": "footage",
      "contextMeaning": "素材，视频剪辑",
      "partOfSpeech": "n.",
      "grammarRole": "宾语 (与 upload 构成动宾短语)",
      "explanation": "【句子翻译】在这个句子中意为：‘将你的视频素材上传到 YouTube’。\\n\\n【解析】这里 'footage' 指的是拍摄好的视频素材。固定搭配 'upload your footage to YouTube' 展示了其在数字媒体语境下的典型用法，特指已完成拍摄、准备进行后期制作或直接上传的视频内容。",
      "otherMeanings": [
        {{ "meaning": "英尺长度", "partOfSpeech": "n.", "example": "The room has a lot of square footage." }}
      ]
    }}

    """


    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=QuickLookupResult,
                thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
            )
        )
        
        if not response.parsed:
            raise ValueError("Empty response from Gemini")
        
        result = response.parsed
        result.word = word  # Ensure correct word is returned
        return result
    except Exception as e:
        print(f"Quick Lookup API Error: {e}")
        raise Exception("快速查词失败，请重试。")


async def rapid_lookup_service(word: str, context: str) -> RapidLookupResult:
    """极速查词服务 - 极致简短的 Prompt 以提高响应速度"""
    model, thinking_level = get_lookup_config()
    
    # 使用更快的模型或配置
    # 强制不使用 thinking 以减少延迟
    prompt = f"Word: {word}\nContext: {context}\nOutput: Concise Chinese meaning (m) and POS (p) in JSON."

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RapidLookupResult,
                # 尽量禁用所有额外开销
                thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
            )
        )
        
        if not response.parsed:
            raise ValueError("Empty response")
        
        return response.parsed
    except Exception as e:
        print(f"Rapid Lookup API Error: {e}")
        # 返回一个降级的响应
        return RapidLookupResult(m="查询失败", p="?")

async def translate_service(text: str) -> TranslateResult:
    """极速翻译服务 - 将英文句子翻译为地道的中文"""
    model, thinking_level = get_translate_config()

    system_instruction = """
    你是一个极速翻译助手。
    你的任务是将用户输入的英文句子翻译成地道、自然、简洁的简体中文。
    只返回翻译结果，不要有任何额外的解释或说明。
    """

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
            )
        )
        
        if not response.text:
            raise ValueError("Empty response from Gemini")
        
        return TranslateResult(translation=response.text.strip())
    except Exception as e:
        print(f"Translate API Error: {e}")
async def generate_daily_summary_service(words: List[dict]) -> BlogSummaryResult:
    """用 AI 结合 Google 搜索对当天的单词及来源链接进行串联总结 (结构化输出)"""
    model = DEFAULT_MODEL
    
    # 构建单词和 URL 信息字符串
    words_info = ""
    for w in words:
        data = w['data']
        meaning = data.get('contextMeaning') or data.get('m') or '未知'
        pos = data.get('partOfSpeech') or ''
        role = data.get('grammarRole') or ''
        exp = data.get('explanation') or ''
        others = data.get('otherMeanings') or []
        
        word_meta = {
            "word": w['word'],
            "contextMeaning": meaning,
            "partOfSpeech": pos,
            "grammarRole": role,
            "explanation": exp,
            "otherMeanings": others,
            "context": w['context'],
            "url": w.get('url') or data.get('url', '')
        }
        words_info += f"- {json.dumps(word_meta, ensure_ascii=False)}\n"
    
    prompt = f"""
    你是一位对排版美学有极致追求的英语学习播客导演。
    以下是用户今天学习并收藏的英语单词，以及它们详细的背景元数据 (JSON 格式)：
    
    {words_info}
    
    **你的目标**:
    创作一期名为“英语沉浸视界”的**访谈实录**。你需要先展示极具设计感的全英文对话，再提供排版优雅的中文对照。
    
    **核心要求**:
    1. **利用 Google 搜索 (CRITICAL)**: 
       - 必须搜索 URL（如果有），让嘉宾 (Guest) 的英文内容充满基于真实背景的深度和趣味。
    2. **Catchy Title**: 
       - [Emoji] + [中文主题标题]。Emoji 根据内容动态自选，严禁机械使用播客图标。标题仅中文。
    3. **Concise Prologue**: 80-120 字的中文开场白。
    4. **Transcript Layout (排版美学 - 重点)**: 
       - **区分访谈部分**: 使用 `### 🎙️ English Transcript` 和 `### 📄 中文访谈记录` 作为二级标题，且两者之间使用 `---` 分隔线。
       - **角色视觉区分 (非侵入性)**: 
         - **Host:** 表现为常规粗体开头。 (例如: **Host:** Hello there...)
         - **Guest:** 表现为粗体开头并**加上 Blockquote 引用符号**。 (例如: > **Guest:** Well, in my opinion...) 
         - **理由**: 引用符号在 MD 中通常会产生色彩侧边条，能完美区分对话双方且不失优雅。
       - **重点标注**: 仅在英文实录中使用 **粗体** 标注收藏的词汇。
    5. **English Interview**: 
       - Host 和 Guest 全程使用极度口语化、生动的英文。对话要一气呵成。
    6. **Chinese Translation**: 
       - 在分隔线后提供完整的翻译。翻译版也要遵循相同的 **Host** 和 > **Guest** 排版规则。
    
    请输出简体中文作为主控语言。
    """

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                response_mime_type="application/json",
                response_schema=BlogSummaryResult,
                thinking_config=types.ThinkingConfig(thinking_level='low'), 
            )
        )
        if response.parsed:
            return response.parsed
        
        # Fallback if parsing fails
        return BlogSummaryResult(
            title="今日学习回顾 📖",
            prologue="这是一份基于你今日学习词汇自动生成的总结。",
            content=response.text.strip() if response.text else "今天学习了这些词，要继续加油哦！"
        )
    except Exception as e:
        print(f"Summary Generation Error: {e}")
        return BlogSummaryResult(
            title="生成失败",
            prologue="AI 在尝试深入了解这些单词背景时遇到了一些挑战。",
            content=f"错误详情: {str(e)}\n\n{words_info}"
        )

async def generate_review_article_service(words: List[dict]) -> ReviewArticle:
    """为 FSRS 复习模式生成每日趣味文章 (播客、辩论、采访、博客等)"""
    model = DEFAULT_MODEL
    
    # 随机选择文章类型
    import random
    types_list = [
        ("podcast", "播客"), 
        ("interview", "采访"), 
        ("debate", "辩论"), 
        ("blog", "深度博客"),
        ("news", "新闻特写")
    ]
    article_type_code, article_type_name = random.choice(types_list)

    # 构建单词元数据
    words_info = ""
    for w in words:
        data = json.loads(w['data']) if isinstance(w['data'], str) else w['data']
        meaning = data.get('contextMeaning') or data.get('m') or '未知'
        pos = data.get('partOfSpeech') or ''
        role = data.get('grammarRole') or ''
        exp = data.get('explanation') or ''
        others = data.get('otherMeanings') or []
        
        word_meta = {
            "word": w['word'],
            "contextMeaning": meaning,
            "partOfSpeech": pos,
            "grammarRole": role,
            "explanation": exp,
            "otherMeanings": others,
            "context": w['context']
        }
        words_info += f"- {json.dumps(word_meta, ensure_ascii=False)}\n"

    prompt = f"""
    你是一位天才内容创作者，擅长编写极具吸引力的英语学习内容。
    今天你需要根据用户复习的 30 个单词，编写一篇文章，形式为：**{article_type_name}**。

    **待包含的单词及其详细背景 (JSON 格式)**:
    {words_info}

    **核心任务**:
    1. **创作内容**: 编写一篇生动有趣的英文文章（包含对应的中文翻译）。
    2. **自然嵌入**: 绝对不要生硬地罗列单词，要让这 30 个单词自然地出现在情境中。
    3. **利用背景**: 参考提供的 `Context` (语境)，如果某个词是在 YouTube 视频中出现的，可以在文中提及相关的背景话题。
    4. **双语格式**: 使用 Markdown 编写。先展示完整的英文版，然后是中文翻译版。
    5. **重点突出**: 在英文版中，将这 30 个单词用 **加粗** 标注。

    **输出格式**: 严格 JSON，匹配 Schema。
    `title`: 给文章起一个吸引人的双语标题。
    `content`: Markdown 格式的文章正文。
    `article_type`: 固定为 "{article_type_code}"。

    请注意：文章要有深度，不要太幼稚。如果是辩论，请展现两种不同的观点；如果是播客，请展现两位主持人之间的碰撞。
    """

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ReviewArticle,
                thinking_config=types.ThinkingConfig(thinking_level='low'), 
            )
        )
        if response.parsed:
            return response.parsed
        
        return ReviewArticle(
            title="今日单词复习",
            content="文章生成失败，但你可以直接在此复习你的单词列表。",
            article_type=article_type_code,
            words_json=[]
        )
    except Exception as e:
        print(f"Review Generation Error: {e}")
        return ReviewArticle(
            title="AI 创作暂时休息中",
            content=f"由于技术原因未能生成文章。错误: {str(e)}",
            article_type="none",
            words_json=[]
        )

