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
        # Determine the preferred device, but let accelerate handle the final placement
        if torch.cuda.is_available() and os.environ.get("DISABLE_CUDA") != "1":
            self.target_device_type = "cuda"
            print("FashionEmbedder: Target device type set to CUDA.")
        else:
            self.target_device_type = "cpu"
            print("FashionEmbedder: Target device type set to CPU.")

        # --- Model Loading Strategy ---
        # Use `device_map="auto"` to let `accelerate` handle device placement.
        # This is often necessary for models initialized on the 'meta' device,
        # as `accelerate` knows how to properly transition them.
        print("Loading Marqo/marqo-fashionSigLIP model using device_map='auto'...")
        model_load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto", # Let accelerate handle device placement
        }

        # Optionally use float16 on CUDA for potentially faster inference and lower memory
        # Note: Check if the specific model supports fp16 well.
        if self.target_device_type == "cuda":
            model_load_kwargs["torch_dtype"] = torch.float16
            print("FashionEmbedder: Using torch_dtype=torch.float16 for CUDA.")


        # Load the model using accelerate's automatic device mapping
        self.model = AutoModel.from_pretrained(
            'Marqo/marqo-fashionSigLIP',
            **model_load_kwargs
        )

        # Determine the actual device the model (or its parts) ended up on.
        # For simple "auto" mapping without multiple GPUs/offloading,
        # this will likely be cuda:0 or cpu.
        # We use the device of the first parameter as a proxy.
        self.device = next(self.model.parameters()).device
        print(f"FashionEmbedder: Model loaded. Parameters are on device: {self.device}")


        # Load processor (usually device-agnostic or handled internally)
        print("Loading Marqo/marqo-fashionSigLIP processor...")
        self.processor = AutoProcessor.from_pretrained(
            'Marqo/marqo-fashionSigLIP',
            trust_remote_code=True
        )

        self.model.eval() # Set to evaluation mode
        print(f"FashionEmbedder initialized successfully on device: {self.device}")

    @staticmethod
    def open_image(url: str) -> Image.Image:
        """Open an image from a URL."""
        try:
            response = requests.get(url, timeout=10) # Add timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            # Ensure conversion to RGB after opening
            return Image.open(BytesIO(response.content)).convert('RGB')
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
            # Ensure inputs are moved to the same device as the model
            processed = self.processor(
                images=images,
                padding='max_length', # Consider padding=True for dynamic padding
                return_tensors="pt"
            ).to(self.device) # Move processed data to the model's device

            with torch.no_grad():
                # If using float16, consider using autocast for potential performance gains
                with torch.autocast(device_type=self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.model.dtype == torch.float16):
                    features = self.model.get_image_features(
                        processed['pixel_values'],
                        normalize=True
                    )
            # Detach, move to CPU, convert to list (converting from potential float16)
            embeddings = features.detach().cpu().float().tolist()
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
            # Ensure inputs are moved to the same device as the model
            processed = self.processor(
                text=texts,
                padding='max_length', # Consider padding=True for dynamic padding
                return_tensors="pt"
            ).to(self.device) # Move processed data to the model's device

            with torch.no_grad():
                 # If using float16, consider using autocast
                with torch.autocast(device_type=self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.model.dtype == torch.float16):
                    features = self.model.get_text_features(
                        processed['input_ids'],
                        # Assuming attention_mask is handled correctly by the processor/model
                        # attention_mask=processed['attention_mask'], # Usually needed, check model requirements
                        normalize=True
                    )
            # Detach, move to CPU, convert to list (converting from potential float16)
            embeddings = features.detach().cpu().float().tolist()
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
    # Log the traceback for more details
    import traceback
    traceback.print_exc()
    embedder = None # Set to None to indicate failure