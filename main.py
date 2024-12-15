from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
from io import BytesIO
import tensorflow as tf  # or torch
import json

app = FastAPI()

# Load the model
model = tf.keras.models.load_model("plant_disease_detector_84.358007.h5")  # Adjust path if needed

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
    input_data = preprocess_image(await file.read())
    # Make prediction
    predictions = model.predict(input_data)
    predicted_class_index = np.argmax(predictions, axis=1)[0]  # Get index of max prediction

    # Get predicted class name
    predicted_class = class_labels[str(predicted_class_index)]  # Get class label
    confidence = np.max(predictions) * 100  # Get confidence as a percentage

    # Return the predicted class and confidence as a response
    return {
        "predicted_plant_disease": predicted_class,
        "confidence": confidence,
        "predicted_class_index" : int(predicted_class_index)
    }
    
if __name__ == '__main__':
   unicorn.run(app, host='127.0.0.1', port = 8000)
 