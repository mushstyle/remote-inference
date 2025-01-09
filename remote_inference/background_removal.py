"""Background removal using BiRefNet."""
import base64
import io
from typing import Tuple, Union
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from PIL import Image
import requests
from pydantic import HttpUrl


class BackgroundRemover:
    def __init__(self):
        """Initialize the BiRefNet model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.image_size = (1024, 1024)
        self.transform_image = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_model(self):
        """Load the BiRefNet model if not already loaded."""
        if self.model is None:
            self.model = AutoModelForImageSegmentation.from_pretrained(
                'ZhengPeng7/BiRefNet', 
                trust_remote_code=True
            )
            torch.set_float32_matmul_precision(['high', 'highest'][0])
            self.model.to(self.device)
            self.model.eval()

    def load_image_from_url(self, url: Union[str, HttpUrl]) -> Image.Image:
        """Load an image from a URL."""
        response = requests.get(str(url))
        response.raise_for_status()
        image_bytes = io.BytesIO(response.content)
        return Image.open(image_bytes).convert('RGB')

    def extract_object(self, image: Union[str, HttpUrl, Image.Image]) -> Tuple[Image.Image, Image.Image]:
        """
        Extract object from image using BiRefNet.
        
        Args:
            image: URL string, file path, or PIL Image
            
        Returns:
            tuple: (extracted_image, mask)
                - extracted_image: PIL Image with transparency
                - mask: PIL Image of the mask
        """
        self.load_model()

        # Handle input image
        if isinstance(image, (str, HttpUrl)):
            if str(image).startswith(('http://', 'https://')):
                source_image = self.load_image_from_url(image)
            else:
                source_image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            source_image = image
        else:
            raise ValueError("image must be a URL string, file path, or PIL Image")

        # Transform and predict
        input_tensor = self.transform_image(source_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid().cpu()
        
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(source_image.size)
        source_image.putalpha(mask)
        
        return source_image, mask

    def remove_background(self, image_url: HttpUrl) -> Tuple[io.BytesIO, str]:
        """
        Remove background from image at URL and return buffer.
        
        Returns:
            Tuple[BytesIO, str]: (image_buffer, mime_type)
                - image_buffer: BytesIO object containing the PNG image
                - mime_type: MIME type of the image (always "image/png")
        """
        try:
            # Process image and get result
            result_image, _ = self.extract_object(image_url)
            
            # Convert to buffer
            img_buffer = io.BytesIO()
            result_image.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            
            return img_buffer, "image/png"
            
        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch image: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")


# Global instance for reuse
_background_remover = None

def get_background_remover() -> BackgroundRemover:
    """Get singleton instance of BackgroundRemover."""
    global _background_remover
    if _background_remover is None:
        _background_remover = BackgroundRemover()
    return _background_remover

def remove_background(image_url: HttpUrl) -> Tuple[io.BytesIO, str]:
    """
    Remove background from image at URL and return buffer.
    
    Returns:
        Tuple[BytesIO, str]: (image_buffer, mime_type)
            - image_buffer: BytesIO object containing the PNG image
            - mime_type: MIME type of the image (always "image/png")
    """
    remover = get_background_remover()
    return remover.remove_background(image_url)