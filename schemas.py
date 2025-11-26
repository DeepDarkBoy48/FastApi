from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union

# --- Shared Enums ---
ModelLevel = Literal['mini', 'quick', 'deep']
WritingMode = Literal['fix', 'ielts-5.5', 'ielts-6.0', 'ielts-6.5', 'ielts-7.0', 'ielts-7.5', 'ielts-8.0']
ContextType = Literal['sentence', 'word', 'writing']

# --- Analysis Schemas ---
class AnalysisChunk(BaseModel):
    text: str
    grammarDescription: str
    partOfSpeech: str
    role: str

class DetailedToken(BaseModel):
    text: str
    partOfSpeech: str
    role: str
    explanation: str
    meaning: str

class CorrectionChange(BaseModel):
    type: Literal['add', 'remove', 'keep']
    text: str

class Correction(BaseModel):
    original: str
    corrected: str
    errorType: str
    reason: str
    changes: List[CorrectionChange]

class AnalysisResult(BaseModel):
    chunks: List[AnalysisChunk]
    detailedTokens: List[DetailedToken]
    chineseTranslation: str
    englishSentence: str
    correction: Optional[Correction] = None
    sentencePattern: Optional[str] = None
    mainTense: Optional[str] = None

class AnalysisRequest(BaseModel):
    sentence: str
    modelLevel: ModelLevel = 'mini'


# --- Dictionary Schemas ---
class DictionaryDefinition(BaseModel):
    meaning: str
    explanation: str
    example: str
    exampleTranslation: str

class DictionaryCollocation(BaseModel):
    phrase: str
    meaning: str
    example: str
    exampleTranslation: str

class DictionaryEntry(BaseModel):
    partOfSpeech: str
    cocaFrequency: Optional[str] = None
    definitions: List[DictionaryDefinition]

class DictionaryResult(BaseModel):
    word: str
    phonetic: str
    entries: List[DictionaryEntry]
    collocations: Optional[List[DictionaryCollocation]] = None

class LookupRequest(BaseModel):
    word: str
    modelLevel: ModelLevel = 'mini'


# --- Writing Schemas ---
class WritingSegment(BaseModel):
    type: Literal['unchanged', 'change']
    text: str
    original: Optional[str] = None
    reason: Optional[str] = None
    category: Optional[Literal['grammar', 'vocabulary', 'style', 'collocation', 'punctuation']] = None

class WritingResult(BaseModel):
    mode: WritingMode
    generalFeedback: str
    segments: List[WritingSegment]

class WritingRequest(BaseModel):
    text: str
    mode: WritingMode
    modelLevel: ModelLevel = 'mini'


# --- Chat Schemas ---
class Message(BaseModel):
    role: Literal['user', 'assistant']
    content: str

class ChatRequest(BaseModel):
    history: List[Message]
    contextContent: Optional[str] = None
    userMessage: str
    contextType: ContextType = 'sentence'

class ChatResponse(BaseModel):
    response: str


# --- TTS Schemas ---
class TTSRequest(BaseModel):
    text: str

class TTSResponse(BaseModel):
    audioData: str  # Base64 encoded audio


