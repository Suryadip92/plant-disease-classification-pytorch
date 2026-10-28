import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image

# Add project root to path so we can import plant_disease_inference
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from plant_disease_inference import (
    load_preprocessing_pipeline,
    load_trained_model,
    predict,
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pt"
PIPELINE_PATH = ARTIFACTS_DIR / "preprocessing_pipeline.joblib"
MODEL_CONFIG_PATH = ARTIFACTS_DIR / "model_config.json"
CLASS_MAPPING_PATH = ARTIFACTS_DIR / "class_mapping.json"

app = FastAPI(title="Plant Disease Classification API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold loaded models
pipeline = None
model = None
class_names = []
model_config = {}
class_mapping = {}


@app.on_event("startup")
async def load_models():
    global pipeline, model, class_names, model_config, class_mapping
    
    try:
        # Load artifacts
        pipeline = load_preprocessing_pipeline(PIPELINE_PATH)
        model, checkpoint = load_trained_model(MODEL_PATH)
        
        # Load configs
        with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
            model_config = json.load(f)
        with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
            class_mapping = json.load(f)
        
        class_names = model_config.get("class_names", class_mapping.get("class_names", []))
        print(f"Loaded model with {len(class_names)} classes")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise


@app.get("/")
async def root():
    return {
        "message": "Plant Disease Classification API",
        "status": "running",
        "num_classes": len(class_names)
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "pipeline_loaded": pipeline is not None,
        "num_classes": len(class_names)
    }


@app.get("/classes")
async def get_classes():
    return {
        "classes": class_names,
        "num_classes": len(class_names)
    }


@app.get("/model-info")
async def get_model_info():
    return {
        "model_config": model_config.get("model_config", {}),
        "input": model_config.get("input", {}),
        "test_metrics": model_config.get("test_metrics", {}),
        "num_classes": len(class_names)
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if model is None or pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    tmp_path = None
    try:
        # Save uploaded file temporarily
        suffix = Path(file.filename).suffix.lower() if file.filename else ".jpg"
        if suffix not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            suffix = '.jpg'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        # Preprocess: resize to 256x256, convert to RGB
        try:
            img = Image.open(tmp_path)
            img = img.convert('RGB')
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
            resized_path = tmp_path.replace(suffix, '_resized.jpg')
            img.save(resized_path, format='JPEG', quality=95)
            os.unlink(tmp_path)
            tmp_path = resized_path
        except Exception as pe:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(pe)}")
        
        # Use existing pipeline for normalization/tensor conversion
        try:
            processed = pipeline.transform(tmp_path)
            # Verify dimensions
            if processed.shape != (3, 256, 256):
                raise ValueError(f"Invalid tensor shape: {processed.shape}, expected (3, 256, 256)")
            result = predict(model, processed, class_names=class_names, top_k=3)
            return result
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Serve frontend static files
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    
    @app.get("/app")
    async def serve_frontend():
        index_path = FRONTEND_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        raise HTTPException(status_code=404, detail="Frontend not found")
    
    @app.get("/frontend")
    async def serve_frontend_alt():
        index_path = FRONTEND_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        raise HTTPException(status_code=404, detail="Frontend not found")
    
    @app.get("/")
    async def serve_root():
        index_path = FRONTEND_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        # fallback to API info
        return {
            "message": "Plant Disease Classification API",
            "status": "running",
            "num_classes": len(class_names)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
