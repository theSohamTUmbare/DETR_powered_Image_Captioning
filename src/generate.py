from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tokenizers.processors import TemplateProcessing
from torch.utils.data import Dataset, DataLoader  # Fixed import
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize, Resize
from transformers import CLIPTokenizer

def _transform():
    return Compose([
        Resize((800, 800)),          # Resize to 800
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transform = _transform() 
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
tokenizer.add_special_tokens({ "pad_token": "[PAD]" })

def generate(image, img_path, model, device):
    """
    Generate caption for a given image using the provided model and tokenizer.
    
    Args:
        img_path (str): Path to the input image.
        text_tokenizer: Tokenizer for processing text inputs.
        model: The model used for generating captions.
        device: Device to run the model on (e.g., 'cpu' or 'cuda').
    
    Returns:
        str: Generated caption for the image.
    """
    

    # Load and preprocess the image
    if img_path is None and image is None:
        raise ValueError("Either img_path or image must be provided.")
    if image is None:
        image = Image.open(img_path)
    image_tensor = transform(image).unsqueeze(0).to(device)  
    image_tensor = image_tensor
    
    # Generate caption!!!
    with torch.no_grad():
        generated_ids = model(image_tensor)

    # Decode the generated ids to text
    caption = tokenizer.decode(generated_ids.view(-1).tolist(), skip_special_tokens=True)
    
    return caption