import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

def get_api_key():
    """
    Get the Google API key from environment variables.
    Returns None if the key is not set.
    """
    if not GOOGLE_API_KEY:
        print("⚠️ Warning: GOOGLE_API_KEY not found in environment variables.")
        print("Please set your API key in the .env file or environment variables.")
        return None
    return GOOGLE_API_KEY

def validate_api_key(api_key):
    """
    Validate the format of the API key.
    Returns True if valid, False otherwise.
    """
    if not api_key:
        return False
    # Basic validation - you might want to add more specific checks
    return len(api_key) > 20 and api_key.startswith('AI') 