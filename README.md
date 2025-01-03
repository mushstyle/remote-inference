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

### API Endpoints

#### Image Embedding
```bash
curl -X POST http://localhost:8000/api/marqo-fashionsiglip/image \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

#### Text Embedding
```bash
curl -X POST http://localhost:8000/api/marqo-fashionsiglip/text \
  -H "Content-Type: application/json" \
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