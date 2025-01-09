"""Background removal using BiRefNet."""
import base64
import io
import time
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
        print(f"BackgroundRemover initializing with device: {self.device}")
        if self.device == "cuda":
            print(f"CUDA device name: {torch.cuda.get_device_name()}")
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"CUDA memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
            # Pin memory for faster data transfer to GPU
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
        self.model = None
        self.image_size = (1024, 1024)
        self.transform_image = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # Pre-allocate reusable tensors
        self.input_tensor = torch.zeros((1, 3, *self.image_size), device=self.device)

    def load_model(self):
        """Load the BiRefNet model if not already loaded."""
        if self.model is None:
            print("Loading BiRefNet model...")
            load_start = time.time()
            self.model = AutoModelForImageSegmentation.from_pretrained(
                'ZhengPeng7/BiRefNet', 
                trust_remote_code=True
            )
            torch.set_float32_matmul_precision(['high', 'highest'][0])
            self.model.to(self.device)
            self.model.eval()
            load_time = time.time() - load_start
            print(f"Model loaded in {load_time:.2f}s")
            if self.device == "cuda":
                print(f"CUDA memory after model load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    def load_image_from_url(self, url: Union[str, HttpUrl]) -> Image.Image:
        """Load an image from a URL."""
        # Use stream=True to start downloading immediately
        with requests.get(str(url), stream=True) as response:
            response.raise_for_status()
            image_bytes = io.BytesIO(response.raw.read())
            # Use PILLOW_LOAD_TRUNCATED_IMAGES=1 for faster loading
            Image.LOAD_TRUNCATED_IMAGES = True
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
        start_time = time.time()
        self.load_model()
        print(f"\nProcessing image on {self.device}...")
        
        # Download time measurement
        download_start = time.time()
        if isinstance(image, (str, HttpUrl)):
            if str(image).startswith(('http://', 'https://')):
                source_image = self.load_image_from_url(image)
            else:
                source_image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            source_image = image
        else:
            raise ValueError("image must be a URL string, file path, or PIL Image")
        download_time = time.time() - download_start
        print(f"Download time: {download_time:.2f}s")

        # Preprocessing time measurement
        preprocess_start = time.time()
        # Reuse pre-allocated tensor
        self.transform_image(source_image).unsqueeze_(0).to(self.device, non_blocking=True, copy=self.input_tensor)
        preprocess_time = time.time() - preprocess_start
        print(f"Preprocess time: {preprocess_time:.2f}s")
        
        # Inference
        with torch.no_grad(), torch.cuda.amp.autocast():
            inference_start = time.time()
            preds = self.model(self.input_tensor)[-1].sigmoid()
            # Keep on GPU for post-processing
            inference_time = time.time() - inference_start
            print(f"Inference time: {inference_time:.2f}s")
            if self.device == "cuda":
                print(f"CUDA memory during inference: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        # Post-processing time measurement
        postprocess_start = time.time()
        pred = preds[0].squeeze()
        if self.device == "cuda":
            pred = pred.cpu()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(source_image.size)
        source_image.putalpha(mask)
        postprocess_time = time.time() - postprocess_start
        print(f"Postprocess time: {postprocess_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f}s")
        
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