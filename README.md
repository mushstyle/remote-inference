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
   *Note: Recent updates include modifications to `remote_inference/ml.py` and pinning dependencies like `accelerate==1.6.0` for stability.*
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

All endpoints require an API key passed in the `X-API-Key` header. First set your API key:
```bash
export API_KEY=your_api_key_here
curl -s -H "X-API-Key: $API_KEY" ...
```

### Image Embedding

Generate embeddings from image URLs:
```bash
# Set your remote endpoint and API key
export REMOTE_HOST=localhost:8000  # or your remote host:port
export API_KEY=your_api_key_here   # your API key

# Example image URLs from our S3 bucket
export IMAGE_URL1="https://dev-mush-frontend-authstack-profilebucket8bf528d8-osju5g4in0xb.s3.eu-central-1.amazonaws.com/d7dba2b1-3bb2-49af-b142-71a9ad7457ae.png"
export IMAGE_URL2="https://dev-mush-frontend-authstack-profilebucket8bf528d8-osju5g4in0xb.s3.eu-central-1.amazonaws.com/fb00bcea-964c-43e4-9e3b-09e4ccf845fa.png"

# Multiple images
curl -s -X POST "http://$REMOTE_HOST/api/v1/embeddings" -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" -d "{\"image_urls\":[\"$IMAGE_URL1\",\"$IMAGE_URL2\"]}"

# Single image
curl -s -X POST "http://$REMOTE_HOST/api/v1/embeddings" -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" -d "{\"image_urls\":[\"$IMAGE_URL1\"]}"

# Example using the new /embeddings endpoint
curl -s -X POST "http://$REMOTE_HOST/api/v1/embeddings" -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" -d "{\"image_urls\":[\"$IMAGE_URL1\"]}"
```

### Text Embedding

Generate embeddings from text queries:
```bash
# Example text queries
export TEXT1="blue denim jacket"
export TEXT2="red silk dress"

# Multiple texts
curl -s -X POST "http://$REMOTE_HOST/api/marqo-fashionsiglip/text" -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" -d "{\"texts\":[\"$TEXT1\",\"$TEXT2\"]}"

# Single text
curl -s -X POST "http://$REMOTE_HOST/api/marqo-fashionsiglip/text" -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" -d "{\"texts\":[\"$TEXT1\"]}"

# Example using the new /api/v1/text-embeddings endpoint
curl -s -X POST "http://$REMOTE_HOST/api/v1/text-embeddings" -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" -d "{\"texts\":[\"$TEXT1\"]}"
```

### Response Format

Both endpoints return embeddings in the format:
```json
{
    "embeddings": [
        [0.1, 0.2, ...],  // 768-dimensional vector for first input
        [0.3, 0.4, ...]   // 768-dimensional vector for second input
    ],
    "metadata": {
        "urls": ["...", "..."]  // For images
        // or
        "texts": ["...", "..."] // For text
    }
}
```

### Background Removal

Remove background from an image and return transparent PNG:
```bash
# Example removing background from an image
export IMAGE_URL="https://static.zara.net/assets/public/3e84/6a36/8df34cee99cb/d4fd95f854ac/04087429600-e1/04087429600-e1.jpg?ts=1722438373817&w=850"
curl -X POST "http://$REMOTE_HOST/api/v1/remove-background" -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" -d "{\"image_url\": \"$IMAGE_URL\"}" --output removed_background.png
```

The endpoint returns a PNG image with transparent background. Any errors will be returned as plain text with a 400 status code.