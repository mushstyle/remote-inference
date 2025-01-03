"""Command line interface."""
import uvicorn

from remote_inference.api import app


def main():
    """Run the API server."""
    port = 70001
    print(f"Starting API server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()