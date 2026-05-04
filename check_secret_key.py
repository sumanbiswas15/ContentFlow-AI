"""
Quick script to check if SECRET_KEY is being loaded correctly from .env
"""
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

secret_key_from_env = os.getenv("SECRET_KEY")
print(f"SECRET_KEY from .env file: {secret_key_from_env[:20]}... (length: {len(secret_key_from_env) if secret_key_from_env else 0})")

# Now check what the app loads
from app.core.config import settings

print(f"SECRET_KEY loaded by app: {settings.SECRET_KEY[:20]}... (length: {len(settings.SECRET_KEY)})")

if secret_key_from_env == settings.SECRET_KEY:
    print("\n✓ SECRET_KEY matches! The app is loading the correct key from .env")
else:
    print("\n✗ SECRET_KEY MISMATCH! The app is NOT loading the key from .env")
    print("This means JWT tokens will be invalid after restart!")
