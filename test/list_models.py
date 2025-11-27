import asyncio
import os
from google import genai
from dotenv import load_dotenv

async def list_models():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    print("Listing models...")
    # The SDK might have a way to list models, but it's not always straightforward in the new SDK.
    # Let's try the standard way if available, or just try a few known ones.
    # client.models.list() might work.
    try:
        # client.aio.models.list() is a coroutine that returns an async iterator or a list
        # Based on the error "coroutine 'AsyncModels.list' was never awaited", we must await it.
        # It likely returns an async iterator or a standard iterator.
        models = await client.aio.models.list()
        async for model in models:
            print(f"Model: {model.name}")
            print(f"  DisplayName: {model.display_name}")
            print(f"  Supported Actions: {model.supported_actions}")
            print("-" * 20)
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    asyncio.run(list_models())
