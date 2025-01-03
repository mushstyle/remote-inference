# Remote Inference

Python package for remote inference.

## Installation

Using uv (recommended):
```bash
uv pip install -e .
```

## Usage

### Running the Server

After installation, run:
```bash
remote-inference
```

This will start the server on port 8000.

### Authentication

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and set your API key:
   ```
   API_KEY=your_secret_key_here
   ```

3. Include the API key in all requests using the `X-API-Key` header.

### API Endpoints

#### Image Embedding
```bash
curl -X POST http://localhost:8000/api/marqo-fashionsiglip/image \
  -H "Content-Type: multipart/form-data" \
  -H "X-API-Key: your_api_key_here" \
  -F "file=@image.jpg"
```

#### Text Embedding
```bash
curl -X POST http://localhost:8000/api/marqo-fashionsiglip/text \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{"text": "blue denim jacket"}'
```

Both endpoints return embeddings in the format:
```json
{
    "embedding": [],
    "metadata": {
        // Request-specific metadata
    }
}
```