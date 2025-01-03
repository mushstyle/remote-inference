"""API server for remote inference."""
from typing import Annotated, List
from fastapi import Depends, FastAPI
from pydantic import BaseModel, HttpUrl

from remote_inference.auth import get_api_key
from remote_inference.ml import embedder

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
    _: None = Depends(get_api_key)
) -> dict:
    """Generate fashion embeddings from image URLs."""
    embeddings = embedder.get_image_embeddings(
        [str(url) for url in query.image_urls]
    )
    return {
        "embeddings": embeddings,
        "metadata": {
            "urls": [str(url) for url in query.image_urls]
        }
    }


@app.post("/api/marqo-fashionsiglip/text")
async def embed_text(
    query: TextQuery,
    _: None = Depends(get_api_key)
) -> dict:
    """Generate fashion embeddings from text queries."""
    embeddings = embedder.get_text_embeddings(query.texts)
    return {
        "embeddings": embeddings,
        "metadata": {
            "texts": query.texts
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)