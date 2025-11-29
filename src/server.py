from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import os

app = FastAPI()

current_dir = Path(__file__).parent
project_root = current_dir.parent
os.chdir(project_root)
# Template directory
templates = Jinja2Templates(directory="templates")

# Ensure uploads folder exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
@app.get("/home/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/upload-form/")
def upload_form(request: Request):
    return templates.TemplateResponse("upload_form.html", {"request": request})

@app.post("/upload-audio/")
async def upload_audio(request: Request, file: UploadFile = File(...)):
    
    # Save file to uploads directory
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Redirect to success page with query params
    return RedirectResponse(
        url=f"/upload-success?filename={file.filename}", 
        status_code=303
    )


@app.get("/upload-success")
def upload_success(request: Request, filename: str):
    return templates.TemplateResponse(
        "upload_form.html",
        {
            "request": request,
            "message": f"File '{filename}' uploaded successfully!"
        }
    )
