import os
import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# Initialize FastAPI App
app = FastAPI(title="Kidney Disease Classification API")

# Path to your trained model from Stage 3
MODEL_PATH = os.path.join("artifacts", "training", "model.h5")
model = None

# 4-Class Labels matching your new dataset structure
CLASS_NAMES = ["Cyst", "Normal", "Tumor", "Stone"]  # Adjust the order to match your generator's class_indices if needed!

def load_prediction_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            # Clear backend session to prevent graph/threading conflicts
            tf.keras.backend.clear_session()
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully!")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}. Prediction endpoints will fail.")

@app.on_event("startup")
async def startup_event():
    load_prediction_model()

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Resizes, converts to array, and rescales the input image."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224)) # Must match IMAGE_SIZE in params.yaml (excluding channels)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0 # Rescaling matching training pipeline
    return img_array

@app.get("/")
def home():
    return {"status": "Active", "message": "Kidney Disease Classification API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Model is not loaded on server."})
    
    try:
        # Read and preprocess the image
        contents = await file.read()
        processed_image = preprocess_image(contents)
        
        # Run inference
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        return {
            "prediction": CLASS_NAMES[predicted_class_idx],
            "confidence": f"{confidence * 100:.2f}%",
            "probabilities": {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)