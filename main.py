from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from typing import List
from pydantic import BaseModel
import os
from model import process_image

app = FastAPI(title="Number Plate Detection API")

# Configure directories
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output_images"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class DetectionResponse(BaseModel):
    message: str
    detected_plates: List[str]
    output_image: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/detect", response_model=DetectionResponse)
async def detect_plate(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        # Save the uploaded file
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process the image
        output_path, detected_plates = process_image(input_path)

        # Clean up the input file
        os.remove(input_path)

        return {
            "message": "Success",
            "detected_plates": detected_plates,
            "output_image": f"/get-image/output_image.jpg"
        }
    except Exception as e:
        if os.path.exists(input_path):
            os.remove(input_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-image/{image_name}")
async def get_image(image_name: str):
    image_path = os.path.join(OUTPUT_FOLDER, image_name)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)


@app.get("/api/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", 8000))  # Get port from environment variable or default to 8000
    uvicorn.run("main:app", host="0.0.0.0", port=port)