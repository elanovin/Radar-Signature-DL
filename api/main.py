from fastapi import FastAPI, File, UploadFile
import numpy as np
from src.inference import RadarInference
import io

app = FastAPI(
    title="Radar Target Classification API",
    description="API for classifying radar targets using micro-Doppler signatures",
    version="1.0.0"
)

# Initialize model
model = RadarInference('models/best_model.pth')

@app.get("/")
async def root():
    return {"message": "Radar Target Classification API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict the class of a radar signal.
    
    Args:
        file: Binary file containing radar signal data
        
    Returns:
        Prediction results including class and confidence
    """
    # Read file
    contents = await file.read()
    
    # Convert to numpy array
    signal = np.frombuffer(contents, dtype=np.float32)
    
    # Get prediction
    prediction = model.predict(signal)
    
    return prediction

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"} 