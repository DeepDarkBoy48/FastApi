from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union

# --- Shared Enums ---
WritingMode = Literal['fix']
ContextType = Literal['sentence', 'word', 'writing']

# --- Analysis Schemas ---
class AnalysisChunk(BaseModel):
    text: str = Field(description="The text content of this chunk.")
    grammarDescription: str = Field(description="Grammatical description of this chunk (e.g., 'Prepositional Phrase', 'Noun Phrase'). MUST be in Simplified Chinese.")
    partOfSpeech: str = Field(description="The part of speech for the head of this chunk (e.g., 'noun', 'verb'). MUST be in Simplified Chinese.")
    role: str = Field(description="The grammatical role of this chunk in the sentence (e.g., 'Subject', 'Predicate', 'Object'). MUST be in Simplified Chinese.")

class DetailedToken(BaseModel):
    text: str = Field(description="The specific word or phrase being analyzed.")
    partOfSpeech: str = Field(description="Part of speech of the token. MUST be in Simplified Chinese.")
    role: str = Field(description="Grammatical role of the token. MUST be in Simplified Chinese.")
    explanation: str = Field(description="Detailed explanation of the token's usage, form, or function in this specific context. MUST be in Simplified Chinese.")
    meaning: str = Field(description="The meaning of the token in this specific context. MUST be in Simplified Chinese.")

class CorrectionChange(BaseModel):
    type: Literal['add', 'remove', 'keep'] = Field(description="Type of change: 'add' (new text), 'remove' (delete text), or 'keep' (unchanged).")
    text: str = Field(description="The text content associated with the change.")

class Correction(BaseModel):
    original: str = Field(description="The original English sentence with errors.")
    corrected: str = Field(description="The corrected English sentence.")
    errorType: str = Field(description="General category of the error (e.g., 'Grammar', 'Spelling').")
    reason: str = Field(description="Explanation of why the correction was made. MUST be in Simplified Chinese.")
    changes: List[CorrectionChange] = Field(description="List of specific changes (diff) between original and corrected sentences.")

class AnalysisResult(BaseModel):
    chunks: List[AnalysisChunk] = Field(description="The sentence broken down into rhythmic/sense chunks.")
    detailedTokens: List[DetailedToken] = Field(description="Detailed analysis of key words and phrases in the sentence.")
    chineseTranslation: str = Field(description="Natural translation of the full sentence into Simplified Chinese.")
    englishSentence: str = Field(description="The English sentence being analyzed (corrected version if applicable).")
    correction: Optional[Correction] = Field(default=None, description="Correction details if the original sentence had errors.")
    sentencePattern: Optional[str] = Field(default=None, description="The core sentence pattern (e.g., 'S+V+O').")
    mainTense: Optional[str] = Field(default=None, description="The primary tense of the sentence (e.g., 'Present Simple').")

class AnalysisRequest(BaseModel):
    sentence: str


# --- Dictionary Schemas ---
class DictionaryDefinition(BaseModel):
    meaning: str = Field(description="Concise meaning in Simplified Chinese.")
    explanation: str = Field(description="Detailed explanation in Simplified Chinese. MUST NOT be English.")
    example: str
    exampleTranslation: str = Field(description="Translation of the example in Simplified Chinese.")

class DictionaryCollocation(BaseModel):
    phrase: str
    meaning: str = Field(description="Meaning of the collocation in Simplified Chinese.")
    example: str
    exampleTranslation: str = Field(description="Translation of the example in Simplified Chinese.")

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


# --- Writing Schemas ---
class WritingSegment(BaseModel):
    type: Literal['unchanged', 'change']
    text: str
    original: Optional[str] = None
    reason: Optional[str] = None
    category: Optional[Literal['grammar', 'vocabulary', 'style', 'collocation', 'punctuation']] = None

class WritingResult(BaseModel):
    mode: WritingMode
    generalFeedback: str = Field(description="General feedback on the writing.")
    overall_comment: str = Field(description="A summary of the user's writing quality and main issues in Simplified Chinese.")
    segments: List[WritingSegment]

class WritingRequest(BaseModel):
    text: str
    mode: WritingMode


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


# --- Quick Lookup Schemas (上下文快速查词) ---
class QuickLookupRequest(BaseModel):
    word: str = Field(description="The word to look up")
    context: str = Field(description="The sentence context where the word appears")

class QuickLookupResult(BaseModel):
    word: str = Field(description="The word being looked up")
    contextMeaning: str = Field(description="The meaning of the word in the given context, in Simplified Chinese")
    partOfSpeech: str = Field(description="The part of speech abbreviation (e.g., 'n.', 'v.', 'adj.'), in Simplified Chinese")
    grammarRole: str = Field(description="The grammatical role of the word in the sentence (e.g., 'Subject', 'Object', 'Fixed collocation'), in Simplified Chinese")
    explanation: str = Field(description="Explanation of why this meaning applies in the context, in Simplified Chinese")

