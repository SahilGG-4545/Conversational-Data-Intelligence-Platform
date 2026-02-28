"""
Test script to check Grok AI API and find available models
"""
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
XAI_API_KEY = os.getenv('XAI_API_KEY')

if not XAI_API_KEY:
    print("ERROR: XAI_API_KEY not found in .env file")
    exit(1)

print(f"API Key found: {XAI_API_KEY[:20]}...")

# Initialize client
client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1"
)

print("\n1. Testing API connection...")

# Test different model names
model_names_to_test = [
    "grok-beta",
    "grok-2-latest",
    "grok-1",
    "grok",
    "grok-2",
    "grok-2-1212",
    "grok-vision-beta"
]

print("\n2. Testing different model names:")
for model_name in model_names_to_test:
    try:
        print(f"\n   Testing model: {model_name}")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Say hello"}
            ],
            max_tokens=10
        )
        print(f"   ✅ SUCCESS! Model '{model_name}' works!")
        print(f"   Response: {response.choices[0].message.content}")
        break  # Stop after first successful model
    except Exception as e:
        print(f"   ❌ Failed: {str(e)[:100]}")

print("\n3. Trying to list available models:")
try:
    models = client.models.list()
    print("   Available models:")
    for model in models.data:
        print(f"   - {model.id}")
except Exception as e:
    print(f"   ❌ Cannot list models: {str(e)[:150]}")

print("\n4. Testing with a simple pandas question:")
try:
    # Find the working model (try common names)
    working_model = None
    for model_name in ["grok-2-1212", "grok-vision-beta", "grok-beta"]:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": "Generate pandas code to get the first 5 rows: df.head()"}
                ],
                max_tokens=50
            )
            working_model = model_name
            print(f"   ✅ Working model found: {working_model}")
            print(f"   Response: {response.choices[0].message.content}")
            break
        except:
            continue
    
    if not working_model:
        print("   ❌ No working model found")
except Exception as e:
    print(f"   ❌ Error: {str(e)}")

print("\n" + "="*50)
print("Test complete!")
