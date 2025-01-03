# Remote Inference

API server for remote fashion embeddings.

## Development Setup

1. Requirements:
   - Python 3.11
   - [uv](https://github.com/astral-sh/uv)

2. Environment setup:
   ```bash
   # Clone the repository
   git clone git@github.com:mushstyle/remote-inference.git
   cd remote-inference

   # Create and activate virtual environment
   uv venv --python=3.11
   source .venv/bin/activate

   # Install in development mode
   uv pip install -e .
   ```

3. Environment configuration:
   ```bash
   # Copy example environment file
   cp .env.example .env

   # Edit .env and set your API key:
   API_KEY=your_secret_key_here
   ```

## Running the Server

After installation and configuration:
```bash
remote-inference
```

This will start the server on port 8000. The API is protected by the API key specified in `.env`.

## API Endpoints

### Authentication

All endpoints require an API key passed in the `X-API-Key` header:
```bash
curl -H "X-API-Key: your_api_key_here" ...
```

### Image Embedding

Generate embeddings from image URLs:
```bash
curl -X POST http://localhost:8000/api/marqo-fashionsiglip/image \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{
    "image_urls": [
      "https://example.com/image1.jpg",
      "https://example.com/image2.jpg"
    ]
  }'
```

### Text Embedding

Generate embeddings from text queries:
```bash
curl -X POST http://localhost:8000/api/marqo-fashionsiglip/text \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{
    "texts": [
      "blue denim jacket",
      "red silk dress"
    ]
  }'
```

### Response Format

Both endpoints return embeddings in the format:
```json
{
    "embeddings": [
        [0.1, 0.2, ...],  // 512-dimensional vector for first input
        [0.3, 0.4, ...]   // 512-dimensional vector for second input
    ],
    "metadata": {
        "urls": ["...", "..."]  // For images
        // or
        "texts": ["...", "..."] // For text
    }
}
```