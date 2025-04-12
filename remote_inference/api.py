"""API server for remote inference."""
import time
from typing import Annotated, List
from fastapi import Depends, FastAPI, Response, status
from pydantic import BaseModel, HttpUrl

from remote_inference.auth import get_api_key
from remote_inference.ml import embedder
from remote_inference import background_removal

app = FastAPI(title="Remote Inference API")


class ImageQuery(BaseModel):
    """Image URLs for fashion embedding."""
    image_urls: List[HttpUrl]


class TextQuery(BaseModel):
    """Text queries for fashion embedding."""
    texts: List[str]


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    # Returns a simple 200 OK response to indicate the service is running.
    # No body content is required by the health check specifications.
    return {"status": "ok"}


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


class BackgroundRemovalRequest(BaseModel):
    """Request for background removal."""
    image_url: HttpUrl


@app.post("/api/v1/remove-background", response_class=Response)
async def remove_background_endpoint(
    request: BackgroundRemovalRequest,
    _: None = Depends(get_api_key)
) -> Response:
    """Remove background from image at URL and return the processed image."""
    print(f"\nProcessing background removal request for: {request.image_url}")
    try:
        import sys
        sys.stdout.flush()  # Ensure prints are flushed immediately
        image_buffer, mime_type = background_removal.remove_background(request.image_url)
        print("Background removal completed successfully")
        sys.stdout.flush()
        return Response(
            content=image_buffer.getvalue(),
            media_type=mime_type
        )
    except ValueError as e:
        return Response(
            content=str(e),
            media_type="text/plain",
            status_code=400
        )


if __name__ == "__main__":
    port = 8000
    print(f"Starting API server on port {port}")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)