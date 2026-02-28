"""
Test script to verify Groq API is working
"""
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY not found in .env file")
    exit(1)

print(f"API Key found: {GROQ_API_KEY[:20]}...")

# Initialize client
client = Groq(api_key=GROQ_API_KEY)

print("\n1. Testing Groq API connection...")

# Test different models
models_to_test = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

working_model = None

for model in models_to_test:
    try:
        print(f"\n   Testing model: {model}")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Generate pandas code to get the first 5 rows: df.head()"}
            ]
        )
        
        print(f"   ✅ SUCCESS! Model '{model}' works!")
        print(f"   Response: {response.choices[0].message.content[:100]}...")
        working_model = model
        break
        
    except Exception as e:
        print(f"   ❌ Failed: {str(e)[:150]}")

if working_model:
    print(f"\n✅ WORKING MODEL FOUND: {working_model}")
else:
    print(f"\n❌ No working model found")

print("\n" + "="*50)
print("Test complete!")
