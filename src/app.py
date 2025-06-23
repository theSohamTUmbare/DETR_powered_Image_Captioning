import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import torch

from config import Config
from imgcap_model import model  
from generate import generate  

app = FastAPI(title="Image Captioning API")


cfg = Config()
device = torch.device(cfg.DEVICE)
model.to(device).eval()

@app.post("/generate_detrcap", summary="Generate a caption for an uploaded image")
async def generate_caption(file: UploadFile = File(...)):
    # Check image type
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="File is not an image.")
    # Read into PIL
    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data.")

    caption = generate(image, img_path=None, model=model, device=device)

    return JSONResponse({"caption": caption})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=3000, reload=True)
