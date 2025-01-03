"""API server for remote inference."""
from typing import Annotated, List
from fastapi import Depends, FastAPI
from pydantic import BaseModel, HttpUrl

from remote_inference.auth import verify_api_key

app = FastAPI(title="Remote Inference API")


class ImageQuery(BaseModel):
    """Image URLs for fashion embedding."""
    image_urls: List[HttpUrl]


class TextQuery(BaseModel):
    """Text queries for fashion embedding."""
    texts: List[str]


@app.post("/api/marqo-fashionsiglip/image")
async def embed_image(
    query: ImageQuery,
    _: None = Depends(verify_api_key)
) -> dict:
    """Generate fashion embeddings from image URLs."""
    # TODO: Implement image embedding
    # For now return dummy vectors (512-dimensional zeros) for each URL
    dummy_vector = [0.0] * 512
    return {
        "embeddings": [dummy_vector for _ in query.image_urls],
        "metadata": {
            "urls": [str(url) for url in query.image_urls]
        }
    }


@app.post("/api/marqo-fashionsiglip/text")
async def embed_text(
    query: TextQuery,
    _: None = Depends(verify_api_key)
) -> dict:
    """Generate fashion embeddings from text queries."""
    # TODO: Implement text embedding
    # For now return dummy vectors (512-dimensional zeros) for each text
    dummy_vector = [0.0] * 512
    return {
        "embeddings": [dummy_vector for _ in query.texts],
        "metadata": {
            "texts": query.texts
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)