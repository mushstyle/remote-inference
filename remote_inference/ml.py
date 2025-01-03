"""Machine learning models and inference functionality."""
from typing import List, Union
import torch
from transformers import AutoModel, AutoProcessor
import requests
from PIL import Image
from io import BytesIO


class FashionEmbedder:
    """Fashion embedding model using Marqo FashionSigLIP."""

    def __init__(self):
        """Initialize the model and processor."""
        self.model = AutoModel.from_pretrained(
            'Marqo/marqo-fashionSigLIP',
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            'Marqo/marqo-fashionSigLIP',
            trust_remote_code=True
        )
        self.model.eval()  # Set to evaluation mode
    
    @staticmethod
    def open_image(url: str) -> Image.Image:
        """Open an image from a URL."""
        response = requests.get(url)
        return Image.open(BytesIO(response.content))

    def get_image_embeddings(self, image_urls: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of image URLs."""
        embeddings = []
        
        for url in image_urls:
            image = self.open_image(url)
            processed = self.processor(
                images=image,
                padding='max_length',
                return_tensors="pt"
            )
            
            with torch.no_grad():
                features = self.model.get_image_features(
                    processed['pixel_values'],
                    normalize=True
                )
                # Convert to list and append
                embeddings.append(features[0].tolist())
        
        return embeddings

    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text queries."""
        processed = self.processor(
            text=texts,
            padding='max_length',
            return_tensors="pt"
        )
        
        with torch.no_grad():
            features = self.model.get_text_features(
                processed['input_ids'],
                normalize=True
            )
            # Convert to numpy and return as list of lists
            return features.tolist()


# Global instance
embedder = FashionEmbedder()