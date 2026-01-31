"""Test script to list available Gemini models."""
import google.generativeai as genai
from app.core.config import settings

# Configure with your API key
genai.configure(api_key=settings.GOOGLE_API_KEY)

print("Available Gemini models:")
print("-" * 50)

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"Model: {model.name}")
        print(f"  Display Name: {model.display_name}")
        print(f"  Description: {model.description}")
        print(f"  Supported methods: {model.supported_generation_methods}")
        print()
