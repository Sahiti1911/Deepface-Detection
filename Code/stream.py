import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np
import tempfile

# Load your CNN model
st.sidebar.text("Loading the model... Please wait.")
cnn_model = load_model('deepfake_detection_cnn_90_26.h5')  # Replace with the path to your trained CNN model file
st.sidebar.text("Model loaded successfully!")

# Streamlit app
st.title('Deepfake Detection')

# Page layout
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .st-sidebar {
            background-color: #333;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar styling
st.sidebar.markdown("""
    #### Instructions
    - Upload an image (JPG) or a video (MP4).
    - The app will predict if it's a deepfake or real.
""")

# File uploader for both image and video
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "mp4"])

if uploaded_file is not None:
    # Check if the uploaded file is an image or video
    if uploaded_file.type.startswith('image'):
        # Image processing
        target_size = cnn_model.input_shape[1:3]
        img = image.load_img(uploaded_file, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the pixel values

        # Make predictions
        predictions = cnn_model.predict(img_array)

        # Display the predictions
        result_text = "REAL" if predictions[0][0] > 0.6 else "DEEPFAKE"
        st.subheader(f"Prediction: {result_text}")

        # Display the uploaded image with a smaller size
        st.image(img, caption='Uploaded Image', use_column_width=False, width=300)
    else:
        # Save the uploaded video to a temporary file
        temp_video_path = tempfile.NamedTemporaryFile(delete=False).name
        with open(temp_video_path, 'wb') as temp_video:
            temp_video.write(uploaded_file.read())

        # OpenCV video capture using the temporary file
        video_capture = cv2.VideoCapture(temp_video_path)

        # Loop to process frames
        while True:
            # Read a frame from the video
            ret, frame = video_capture.read()
            if not ret:
                break

            # Resize the frame to match the input size of the model
            target_size = cnn_model.input_shape[1:3]
            frame = cv2.resize(frame, target_size)

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Preprocess the frame for model prediction
            img_array = image.img_to_array(rgb_frame)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize the pixel values

            # Make predictions
            predictions = cnn_model.predict(img_array)

            # Update prediction result
            prediction_probability = predictions[0][0]
            prediction_result = "Real" if prediction_probability > 0.86 else "Fake"

            # Display the final prediction result and probability
            st.subheader(f"Prediction: {prediction_result} (Probability: {prediction_probability:.2f})")

            # Display the processed video feed with a smaller size
            st.image(rgb_frame, caption='Processed Video Frame', use_column_width=False, width=300)

            # Break out of the loop after the first frame is processed
            break

        # Release the video capture object
        video_capture.release()
