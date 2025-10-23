# 

# main.py - FastAPI Backend with Video Serving
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import joblib
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import cv2
import os
import shutil
import warnings
from fastapi.responses import StreamingResponse
import mimetypes
import uuid  # Added for unique video names
warnings.filterwarnings("ignore")

# Sensor Data Model (12 features)
class SensorData(BaseModel):
    Plant_ID: int = 1
    Soil_Moisture: float
    Ambient_Temperature: float
    Soil_Temperature: float
    Humidity: float
    Light_Intensity: float
    Soil_pH: float
    Nitrogen_Level: float
    Phosphorus_Level: float
    Potassium_Level: float
    Chlorophyll_Content: float
    Electrochemical_Signal: float

app = FastAPI(title="Strawberry Disease Prediction API", version="1.0.0")

# Load models using relative paths (best practice: avoid absolute paths)
print("Loading models...")
yolo_model = YOLO("models/best.pt")
dt_model = joblib.load("models/Decision_Tree.pkl")
scaler = joblib.load("models/scaler.pkl")
print("Models loaded successfully!")

label_map = {0: "Healthy", 1: "Moderate Stress", 2: "High Stress"}

@app.post("/predict/health")
async def predict_health(data: SensorData):
    try:
        input_data = np.array([[
            data.Plant_ID, data.Soil_Moisture, data.Ambient_Temperature, 
            data.Soil_Temperature, data.Humidity, data.Light_Intensity,
            data.Soil_pH, data.Nitrogen_Level, data.Phosphorus_Level,
            data.Potassium_Level, data.Chlorophyll_Content, data.Electrochemical_Signal
        ]])
        
        input_scaled = scaler.transform(input_data)
        pred = dt_model.predict(input_scaled)[0]
        confidence = dt_model.predict_proba(input_scaled)[0].max() * 100
        
        health_status = label_map.get(int(pred), "Unknown")
        
        return {
            "plant_health_status": health_status,
            "confidence": f"{confidence:.2f}%",
            "prediction_code": int(pred)
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        results = yolo_model(image)
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    detections.append({
                        "class": r.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy.tolist()[0]
                    })
        
        return {"detections": detections, "total_detections": len(detections)}
    except Exception as e:
        return {"error": f"Image detection failed: {str(e)}"}

@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    try:
        # Temp directory
        temp_dir = "temp_video"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded video with unique name
        unique_id = str(uuid.uuid4())[:8]
        input_filename = f"input_{unique_id}.mp4"
        temp_video_path = os.path.join(temp_dir, input_filename)
        with open(temp_video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Run YOLO prediction with unique project name
        results = yolo_model.predict(
            source=temp_video_path,
            save=True,
            project="runs/detect",
            name=f"predict_{unique_id}",
            exist_ok=True  # Overwrite if exists
        )
        
        # Output path (YOLO saves as .avi by default)
        output_video_path = f"runs/detect/predict_{unique_id}/{input_filename[:-4]}.avi"
        
        # Cleanup temp input
        os.remove(temp_video_path)
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return {
            "message": "Video processed successfully",
            "output_video_path": output_video_path,
            "total_frames": len(results)
        }
    except Exception as e:
        return {"error": f"Video detection failed: {str(e)}"}

# NEW: Serve processed video
@app.get("/video/{video_path:path}")
async def get_video(video_path: str):
    """Serve processed video file for streaming/download."""
    try:
        full_path = os.path.join(os.getcwd(), video_path)
        
        if not os.path.exists(full_path):
            return {"error": f"Video not found: {full_path}"}
        
        def file_iterator(file_path, chunk_size=8192):
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        
        mime_type, _ = mimetypes.guess_type(full_path)
        if mime_type is None:
            mime_type = 'video/avi'
        
        return StreamingResponse(
            file_iterator(full_path),
            media_type=mime_type,
            headers={"Content-Disposition": f"attachment; filename={os.path.basename(full_path)}"}
        )
    except Exception as e:
        return {"error": f"Failed to serve video: {str(e)}"}

@app.get("/")
async def root():
    return {"message": "Plant Disease Prediction API", "endpoints": [
        "POST /predict/health - Sensor data prediction",
        "POST /detect/image - Image detection",
        "POST /detect/video - Video detection",
        "GET /video/{path} - Download processed video"
    ]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8502)