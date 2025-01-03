"""API server for remote inference."""
import time
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
    start_time = time.time()
    image_urls = [str(url) for url in query.image_urls]
    embeddings = embedder.get_image_embeddings(image_urls)
    elapsed = time.time() - start_time
    print(f"Generated {len(image_urls)} image embeddings in {elapsed:.2f}s")
    return {
        "embeddings": embeddings,
        "metadata": {
            "urls": image_urls,
            "time_seconds": elapsed
        }
    }


@app.post("/api/marqo-fashionsiglip/text")
async def embed_text(
    query: TextQuery,
    _: None = Depends(get_api_key)
) -> dict:
    """Generate fashion embeddings from text queries."""
    start_time = time.time()
    embeddings = embedder.get_text_embeddings(query.texts)
    elapsed = time.time() - start_time
    print(f"Generated {len(query.texts)} text embeddings in {elapsed:.2f}s")
    return {
        "embeddings": embeddings,
        "metadata": {
            "texts": query.texts,
            "time_seconds": elapsed
        }
    }


if __name__ == "__main__":
    port = 8000
    print(f"Starting API server on port {port}")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)