"""API server for remote inference."""
from typing import Annotated, Union
from fastapi import Depends, FastAPI, File, UploadFile
from pydantic import BaseModel

from remote_inference.auth import verify_api_key

app = FastAPI(title="Remote Inference API")


class TextQuery(BaseModel):
    """Text query for fashion embedding."""
    text: str


@app.post("/api/marqo-fashionsiglip/image")
async def embed_image(
    file: UploadFile = File(...),
    _: None = Depends(verify_api_key)
) -> dict:
    """Generate fashion embedding from image."""
    # TODO: Implement image embedding
    return {
        "embedding": [],  # Will be vector
        "metadata": {
            "filename": file.filename,
            "content_type": file.content_type
        }
    }


@app.post("/api/marqo-fashionsiglip/text")
async def embed_text(
    query: TextQuery,
    _: None = Depends(verify_api_key)
) -> dict:
    """Generate fashion embedding from text."""
    # TODO: Implement text embedding
    return {
        "embedding": [],  # Will be vector
        "metadata": {
            "text": query.text
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)