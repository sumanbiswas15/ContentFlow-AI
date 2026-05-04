"""
Test script to verify JWT token generation and validation.
"""
from jose import jwt, JWTError
from datetime import datetime, timedelta

# Use the SECRET_KEY from .env
SECRET_KEY = "1d5ad13d00d603189912c8094c12154e0ef20ad103ec3537c4122cf01e6d5fef"
ALGORITHM = "HS256"

# Create a test token
test_email = "rksb1507@gmail.com"
test_data = {"sub": test_email, "user_id": "test_user_id"}
expire = datetime.utcnow() + timedelta(days=7)
test_data.update({"exp": expire})

# Generate token
token = jwt.encode(test_data, SECRET_KEY, algorithm=ALGORITHM)
print(f"Generated token: {token[:50]}...")
print(f"Full token: {token}")

# Try to decode it
try:
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    print(f"\nDecoded successfully!")
    print(f"Email: {payload.get('sub')}")
    print(f"User ID: {payload.get('user_id')}")
    print(f"Expires: {datetime.fromtimestamp(payload.get('exp'))}")
except JWTError as e:
    print(f"\nFailed to decode: {e}")

# Now try with a token from your browser (paste it here)
print("\n" + "="*50)
print("Testing your actual token from browser...")
print("="*50)

# Get token from user
your_token = input("\nPaste your JWT token from browser console: ").strip()

if your_token:
    try:
        payload = jwt.decode(your_token, SECRET_KEY, algorithms=[ALGORITHM])
        print(f"\nYour token decoded successfully!")
        print(f"Email: {payload.get('sub')}")
        print(f"User ID: {payload.get('user_id')}")
        print(f"Expires: {datetime.fromtimestamp(payload.get('exp'))}")
    except JWTError as e:
        print(f"\nFailed to decode your token: {e}")
        print("This means the token was signed with a different SECRET_KEY")
