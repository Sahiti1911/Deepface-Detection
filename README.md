# Deepface-Detection

Deepfake Detection Model:

This repository contains a machine learning model for detecting Deepfake
The model has been trained with a test accuracy of 90.12%.


Classes: 
   Deepfake  
   Real

Model Details:
The model is based on a Conventional neural network architecture and has been trained on a comprehensive dataset of deepfake Videos. The training process achieved an impressive 90.12% accuracy on the test set, demonstrating its robustness in detecting Deepfake.

Usage:
To use the trained model for Detecting Deepfake based on an input image, follow these steps:

1) Ensure you have the required dependencies installed. 

2) Run the Training_Deepfake.ipnb script to deploy the model using Streamlit. This will start a local server.
streamlit run streamf.py

3) Access the provided URL in your web browser.
4) Upload an image in PNG, JPG, JPEG or Mp4 format.
5) The model will Detect Deepfake of the given input.

Note:
The model's accuracy is based on the training dataset, and its performance on new, unseen data may vary.

Feel free to explore and contribute to the project. If you encounter any issues or have suggestions, please open an issue in the repository.
