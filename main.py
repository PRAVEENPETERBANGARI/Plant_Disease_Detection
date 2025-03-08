from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn
import numpy as np
from io import BytesIO
import tensorflow as tf  # or torch
import json
import gdown
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to ["http://localhost:3000"] for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
# model = tf.keras.models.load_model("plant_disease_detector_84.358007.h5")  # Adjust path if needed
model_url = "https://drive.google.com/drive/folders/11FZot6t8A3FuQvidtaZb7EuLdvPZCbzJ?usp=drive_link"
model_path = "plant_disease_detector_84.358007.h5"
if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)
    print("Model downloaded successfully")

# Load class labels
def load_class_labels(json_path='class_labels.json'):
    with open(json_path, 'r') as file:
        return json.load(file)
class_labels = load_class_labels()

@app.get("/health")
def read_root():
    return {"name": "Plant Disease Detection API New"}

# Define preprocessing function
def preprocess_image(image):
    image = Image.open(BytesIO(image))
    image = image.resize((224, 224))  # Adjust size based on your model's requirement
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array /= 255.0  # Normalize pixel values
    return image_array

    
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess the image
    # image = Image.open(file.file)
    try:
        input_data = preprocess_image(await file.read())
        # Make prediction
        predictions = model.predict(input_data)
        predicted_class_index = np.argmax(predictions, axis=1)[0]  # Get index of max prediction
        # Get predicted class name
        predicted_class = class_labels[str(predicted_class_index)]  # Get class label
        confidence = np.max(predictions) * 100  # Get confidence as a percentage
        # Return the predicted class and confidence as a response
        result =  {
            "predicted_plant_disease": predicted_class,
            "confidence": confidence,
            "predicted_class_index" : int(predicted_class_index)
           }
        return JSONResponse(content=result)
        
    except Exception as e:
        print("Error:", str(e))  # Log error in terminal
        return JSONResponse(content={"error": str(e)}, status_code=400)
    
if __name__ == '__main__':
   uvicorn.run(app, host='127.0.0.1', port = 8001)
 