import asyncio
import os
import sys
from dotenv import load_dotenv

# Add current directory to sys.path so we can import gemini
sys.path.append(os.getcwd())

import gemini
from schemas import AnalysisRequest, ModelLevel

async def test_analyze():
    print("Testing Analyze...")
    try:
        result = await gemini.analyze_sentence_service("I go to school yesterday.", "mini")
        print(f"Analyze Success: {result.englishSentence}")
    except Exception as e:
        print(f"Analyze Error: {e}")

async def main():
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not found in environment.")
        return

    await test_analyze()

if __name__ == "__main__":
    asyncio.run(main())
