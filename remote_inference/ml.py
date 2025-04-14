"""Machine learning models and inference functionality."""
from typing import List, Union
import torch
from transformers import AutoModel, AutoProcessor
import requests
from PIL import Image
from io import BytesIO
import os # Import os to check environment variables potentially controlling device

class FashionEmbedder:
    """Fashion embedding model using Marqo FashionSigLIP."""

    def __init__(self):
        """Initialize the model and processor."""
        # Determine the device explicitly
        # Allow overriding via environment variable for flexibility
        if torch.cuda.is_available() and os.environ.get("DISABLE_CUDA") != "1":
            self.device = torch.device("cuda")
            print("FashionEmbedder: Using CUDA device.")
        else:
            self.device = torch.device("cpu")
            print("FashionEmbedder: Using CPU device.")

        # --- Model Loading Strategy ---
        # 1. Load model explicitly onto CPU first to avoid potential 'meta' device issues
        #    during the internal `from_pretrained` process, especially with complex models
        #    that might use libraries like `open-clip` internally.
        # 2. Move the model to the target device (CPU or CUDA) *after* loading.
        print("Loading Marqo/marqo-fashionSigLIP model onto CPU...")
        self.model = AutoModel.from_pretrained(
            'Marqo/marqo-fashionSigLIP',
            trust_remote_code=True,
            # Explicitly load on CPU initially
            device_map=None # Ensure transformers doesn't automatically use accelerate's device_map='auto' yet
            # torch_dtype=torch.float16 # Optional: uncomment if using GPU and want fp16
        ).to(torch.device("cpu")) # Force CPU loading initially

        print(f"Moving model to target device: {self.device}...")
        self.model = self.model.to(self.device) # Now move to the final target device

        # Load processor (usually device-agnostic or handled internally)
        print("Loading Marqo/marqo-fashionSigLIP processor...")
        self.processor = AutoProcessor.from_pretrained(
            'Marqo/marqo-fashionSigLIP',
            trust_remote_code=True
        )

        self.model.eval() # Set to evaluation mode
        print("FashionEmbedder initialized successfully.")

    @staticmethod
    def open_image(url: str) -> Image.Image:
        """Open an image from a URL."""
        try:
            response = requests.get(url, timeout=10) # Add timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return Image.open(BytesIO(response.content)).convert('RGB') # Ensure RGB
        except requests.exceptions.RequestException as e:
            print(f"Error fetching image from {url}: {e}")
            raise # Re-raise after logging
        except Exception as e:
            print(f"Error opening image from {url}: {e}")
            raise # Re-raise other potential errors (e.g., PIL errors)


    def get_image_embeddings(self, image_urls: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of image URLs."""
        images = []
        valid_urls = []
        for url in image_urls:
            try:
                img = self.open_image(url)
                images.append(img)
                valid_urls.append(url)
            except Exception as e:
                # Log and skip invalid images/URLs
                print(f"Skipping image URL {url} due to error: {e}")
                # Optionally, return placeholder or handle differently
                continue # Skip this URL

        if not images:
            print("No valid images found to process.")
            return [] # Return empty list if no images could be loaded

        # Process valid images
        try:
            processed = self.processor(
                images=images,
                padding='max_length', # Consider padding=True for dynamic padding
                return_tensors="pt"
            ).to(self.device) # Move processed data to the model's device

            with torch.no_grad():
                features = self.model.get_image_features(
                    processed['pixel_values'],
                    normalize=True
                )
            # Detach, move to CPU, convert to list
            embeddings = features.detach().cpu().tolist()
            return embeddings
        except Exception as e:
            print(f"Error during image embedding inference: {e}")
            # Depending on desired behavior, you might raise, return [], or partial results
            raise # Re-raise the error for now


    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text queries."""
        if not texts:
            return []

        try:
            processed = self.processor(
                text=texts,
                padding='max_length', # Consider padding=True for dynamic padding
                return_tensors="pt"
            ).to(self.device) # Move processed data to the model's device

            with torch.no_grad():
                features = self.model.get_text_features(
                    processed['input_ids'],
                    # Assuming attention_mask is handled correctly by the processor/model
                    # attention_mask=processed['attention_mask'], # Usually needed, check model requirements
                    normalize=True
                )
            # Detach, move to CPU, convert to list
            embeddings = features.detach().cpu().tolist()
            return embeddings
        except Exception as e:
            print(f"Error during text embedding inference: {e}")
            raise # Re-raise the error


# Global instance - lazy initialization might be better in complex apps,
# but for this structure, initializing at import time is intended.
try:
    embedder = FashionEmbedder()
except Exception as e:
    print(f"FATAL: Failed to initialize FashionEmbedder: {e}")
    # Depending on the application structure, you might want to exit or raise
    # For a server, allowing it to start but log the error might be preferable initially.
    embedder = None # Set to None to indicate failure