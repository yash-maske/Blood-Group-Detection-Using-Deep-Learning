import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import time

# Define available models
MODEL_PATHS = {
    "LeNet": "./models/model_blood_group_detection_lenet.keras",
    "AlexNet": "./models/model_blood_group_detection_alextnet.keras",
    "ResNet": "./models/model_blood_group_detection_resNet.h5",
    "VGG16": "./models/model_blood_group_detection_vgg16.h5",
}

# Blood group labels
labels = {0: "A+", 1: "A-", 2: "AB+", 3: "AB-", 4: "B+", 5: "B-", 6: "O+", 7: "O-"}

# Streamlit UI
st.title("🩸 Blood Group Detection from Fingerprint")
st.write("📌 Upload a fingerprint image and select a model to predict the blood group.")

# Model selection
model_choice = st.selectbox("🔍 Select Model", list(MODEL_PATHS.keys()))

# File uploader for image
uploaded_file = st.file_uploader("📂 Upload a fingerprint image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    try:
        # Load model and get expected input shape
        with st.spinner("⏳ Loading model... Please wait!"):
            try:
                model = load_model(MODEL_PATHS[model_choice])
                expected_shape = model.input_shape[1:]  # Exclude batch dimension
                st.write(f"ℹ️ Model expects input shape: {expected_shape}")
                time.sleep(1.5)
            except FileNotFoundError:
                st.error(f"❌ Model file not found at {MODEL_PATHS[model_choice]}")
                st.stop()
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
                st.stop()

        # Adjust target size based on model
        try:
            target_size = (expected_shape[0], expected_shape[1])  # height, width
            
            # Special handling for LeNet if it expects a different shape
            if model_choice == "LeNet":
                # Adjust this based on your LeNet's actual expected shape
                # Example: If LeNet expects 32x32x1, set accordingly
                if len(expected_shape) == 3 and expected_shape[2] == 1:  # Grayscale
                    target_size = (expected_shape[0], expected_shape[1])
                    img = image.load_img(uploaded_file, target_size=target_size, color_mode="grayscale")
                else:
                    target_size = (expected_shape[0], expected_shape[1])
                    img = image.load_img(uploaded_file, target_size=target_size)
            else:
                img = image.load_img(uploaded_file, target_size=target_size)

            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # If model expects flattened input (common for LeNet)
            if len(expected_shape) == 1:
                img_array = img_array.reshape(1, -1)
                st.write(f"ℹ️ Flattened input to shape: {img_array.shape}")

            # Verify shape
            if img_array.shape[1:] != expected_shape and len(expected_shape) != 1:
                st.error(f"❌ Processed image shape {img_array.shape[1:]} doesn't match expected {expected_shape}")
                st.stop()

        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.stop()

        # Progress bar
        progress_bar = st.progress(0)
        try:
            for percent_complete in range(100):
                time.sleep(0.02)
                progress_bar.progress(percent_complete + 1)
        except Exception as e:
            st.error(f"❌ Error during progress simulation: {str(e)}")
            st.stop()

        # Model prediction
        try:
            result = model.predict(img_array)
            predicted_class = np.argmax(result)
            predicted_label = labels[predicted_class]
            confidence = result[0][predicted_class] * 100
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.stop()

        # Display results
        progress_bar.empty()
        try:
            st.image(img, caption="✅ Uploaded Image", use_column_width=True)
            st.success("🎉 Prediction Complete!")
            st.subheader(f"🔬 **Prediction: {predicted_label}**")
            st.write(f"🎯 **Confidence: {confidence:.2f}%**")
        except Exception as e:
            st.error(f"❌ Error displaying results: {str(e)}")
            st.stop()

    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        st.stop()

else:
    st.info("ℹ️ Please upload an image to begin prediction.")