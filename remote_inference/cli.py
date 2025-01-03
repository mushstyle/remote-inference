"""Command line interface."""
import uvicorn

from remote_inference.api import app


def main():
    """Run the API server."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()