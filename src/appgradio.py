import torch
from PIL import Image
import gradio as gr
from huggingface_hub import snapshot_download

from config import Config        
from imgcap_model import model
from generate import generate

HF_MODEL = "SohamUmbare/DETR_powered_imgCAP"
local_dir = snapshot_download(repo_id=HF_MODEL)

cfg    = Config()
device = torch.device(cfg.DEVICE)
weights_path = f"{local_dir}/DETR_CAP012.pth"
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device).eval()

def caption_fn(img: Image.Image):
    return generate(img, img_path=None, model=model, device=device)

demo = gr.Interface(
    fn=caption_fn,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="DETR Image Captioning",
    description="Upload an image to get a caption."
)

if __name__ == "__main__":
    demo.launch()
