import streamlit as st
import plotly.graph_objects as go
from PIL import Image
import numpy as np
import tensorflow as tf
import json
# Streamlit app layout
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌱", layout="centered")

# Load the trained model
@st.cache_resource
def load_model(model_name):
    return tf.keras.models.load_model(model_name)

model_name = "plant_disease_detector_84.358007.h5"
model = load_model(model_name)

# Load class labels from JSON
@st.cache_data
def load_class_labels(json_path='class_labels.json'):
    with open(json_path, 'r') as file:
        return json.load(file)

class_labels = load_class_labels()
categories = list(class_labels.values())
values = list(class_labels.keys())

# Function to preprocess the image
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image, dtype=np.float64) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

st.title("🌱 Plant Disease Detection")
st.write("Upload an image of a plant leaf, and this app will predict whether the plant is healthy or has a disease.")

# # Sidebar for settings
# st.sidebar.header("Settings")
# show_examples = st.sidebar.checkbox("Show example images", value=True)

# if show_examples:
#     st.sidebar.image("example_image.jpg", caption="Sample Healthy Leaf", use_column_width=True)

# File uploader
uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    st.write("Analyzing the image...")
    processed_image = preprocess_image(image, target_size=(224, 224))

    # Make predictions
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[str(predicted_class_index)]
    confidence = np.max(predictions) * 100

    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
    st.bar_chart(predictions[0])

    st.subheader("Class_Names")
    fig = go.Figure(data=[
        go.Bar(
            x=categories,  # Categories from JSON
            y=values,      # Values from JSON
            text=values,   # Text to display on hover
            textposition='auto',  # Optionally show labels directly on bars
            hovertemplate='<b>%{x}</b><br>Value: %{y}<extra></extra>'  # Custom hover tooltip
        )
    ])
    
    # Add layout configurations (optional)
    fig.update_layout(
        title="Interactive Bar Chart",
        xaxis_title="Categories",
        yaxis_title="Values",
        template="plotly_white"
    )
    st.plotly_chart(fig)
    
    # Add a call-to-action button
    st.markdown("### 🌟 [Learn more about plant diseases](https://example.com)")

# Footer
st.markdown(
    """
    ---
    **Plant Disease Detector**  
    Built with ❤️ using Streamlit and TensorFlow.
    """
)
