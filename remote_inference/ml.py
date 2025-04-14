"""Machine learning models and inference functionality."""
from typing import List, Union
import torch
from transformers import AutoModel, AutoProcessor
import requests
from PIL import Image
from io import BytesIO
import os # Import os to check environment variables potentially controlling device
import traceback # For improved error logging

class FashionEmbedder:
    """Fashion embedding model using Marqo FashionSigLIP."""

    def __init__(self):
        """Initialize the model and processor."""
        # Determine the preferred device type
        if torch.cuda.is_available() and os.environ.get("DISABLE_CUDA") != "1":
            self.target_device_type = "cuda"
            print("FashionEmbedder: Target device type set to CUDA.")
        else:
            self.target_device_type = "cpu"
            print("FashionEmbedder: Target device type set to CPU.")

        # --- Model Loading Strategy ---
        # Explicitly load to CPU first, then move to target device.
        # Avoid device_map="auto" as it seems problematic with the specific
        # combination of old accelerate, new torch/transformers, and potentially the newer driver.
        print("Loading Marqo/marqo-fashionSigLIP model explicitly onto CPU...")
        model_load_kwargs = {
            "trust_remote_code": True,
            # No device_map="auto"
            # No torch_dtype here (apply later if needed)
        }

        # Load the model explicitly to CPU first
        try:
            self.model = AutoModel.from_pretrained(
                'Marqo/marqo-fashionSigLIP',
                **model_load_kwargs
            ).to("cpu") # Force to CPU after loading
            print("Model loaded to CPU successfully.")
        except Exception as e:
            print(f"Error during initial model loading: {e}")
            raise # Re-raise to prevent proceeding with a non-loaded model

        self.device = torch.device(self.target_device_type) # Get target device (cuda or cpu)

        # Move model from CPU to the target device
        try:
            print(f"Moving model from CPU to target device: {self.device}...")
            self.model = self.model.to(self.device) # Move to target device
            print("Model moved to target device successfully.")
        except Exception as e:
            print(f"Error moving model to device {self.device}: {e}")
            # Handle error appropriately, maybe fall back to CPU?
            # For now, just raise to make the failure clear.
            raise

        # If using CUDA and want float16, cast *after* moving to GPU
        # Note: Ensure the model supports .half() correctly.
        if self.device.type == "cuda":
            try:
                print("Converting model to float16 on CUDA device...")
                self.model = self.model.half() # Use .half() for float16
                print("Model converted to float16 successfully.")
            except Exception as e:
                 print(f"Warning: Could not convert model to float16: {e}. Proceeding with float32.")
                 # Continue with the model in float32 if half() fails


        print(f"FashionEmbedder: Model initialization process complete. Final model device: {self.device}, dtype: {self.model.dtype}")

        # Load processor (usually device-agnostic or handled internally)
        print("Loading Marqo/marqo-fashionSigLIP processor...")
        self.processor = AutoProcessor.from_pretrained(
            'Marqo/marqo-fashionSigLIP',
            trust_remote_code=True
        )

        self.model.eval() # Set to evaluation mode
        print(f"FashionEmbedder ready.")

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
                # Use autocast if model ended up in float16
                current_dtype = self.model.dtype
                use_autocast = current_dtype == torch.float16 and self.device.type != 'mps'
                with torch.autocast(device_type=self.device.type if self.device.type != 'mps' else 'cpu', enabled=use_autocast):
                    features = self.model.get_image_features(
                        processed['pixel_values'],
                        normalize=True
                    )
                    # Explicitly cast back to float32 if autocast was used, before moving to CPU
                    if use_autocast:
                        features = features.float()

            # Detach, move to CPU, convert to list
            embeddings = features.detach().cpu().tolist() # Already float32 if autocast was used correctly
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
                # Use autocast if model ended up in float16
                current_dtype = self.model.dtype
                use_autocast = current_dtype == torch.float16 and self.device.type != 'mps'
                with torch.autocast(device_type=self.device.type if self.device.type != 'mps' else 'cpu', enabled=use_autocast):
                    features = self.model.get_text_features(
                        processed['input_ids'],
                        # Assuming attention_mask is handled correctly by the processor/model
                        # attention_mask=processed['attention_mask'], # Usually needed, check model requirements
                        normalize=True
                    )
                    # Explicitly cast back to float32 if autocast was used, before moving to CPU
                    if use_autocast:
                        features = features.float()

            # Detach, move to CPU, convert to list
            embeddings = features.detach().cpu().tolist() # Already float32 if autocast was used correctly
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
    traceback.print_exc()
    embedder = None # Set to None to indicate failure