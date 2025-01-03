"""Authentication utilities."""
import os
from typing import Annotated
from fastapi import Header, HTTPException, status
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is not set")


async def verify_api_key(x_api_key: Annotated[str, Header()]) -> None:
    """Verify the API key from request header."""
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )