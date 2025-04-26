import sys
import io

# Set the console output to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os

# Load the trained model
model = tf.keras.models.load_model("dysarthric_classifier_model.h5")

def audio_to_spectrogram(audio_path, output_path="spectrogram.png"):
    """
    Converts an audio file to a spectrogram and saves it as an image.
    
    Parameters:
        audio_path (str): Path to the audio file.
        output_path (str): Path to save the spectrogram image.
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Generate Mel spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Plot and save spectrogram
    plt.figure(figsize=(5, 5))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis="time", y_axis="mel")
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    
    return output_path

def load_and_preprocess_image(image_path):
    """
    Loads an image and preprocesses it for the model.
    
    Parameters:
        image_path (str): Path to the image file.
    
    Returns:
        np.array: Preprocessed image array.
    """
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_dysarthria(audio_path):
    """
    Predicts whether the given audio has dysarthric or non-dysarthric speech.
    
    Parameters:
        audio_path (str): Path to the audio file.
    
    Returns:
        str: Prediction result ("Dysarthric" or "Non-Dysarthric").
    """
    # Convert audio to spectrogram
    spectrogram_path = audio_to_spectrogram(audio_path)
    
    # Preprocess the spectrogram image
    spectrogram_img = load_and_preprocess_image(spectrogram_path)
    
    # Get raw prediction probabilities
    prediction = model.predict(spectrogram_img)
    probability = prediction[0][0]  # Model outputs probability for class 1 (Dysarthric)
    
    # Print raw probability for debugging
    print(f"Raw Prediction Probability: {probability}")
    
    # Set class label based on threshold (adjust if needed)
    threshold = 0.5  # Try tweaking this if needed
    class_label = "Non-Dysarthric" if probability >= threshold else "Dysarthric"
    
    # Clean up (optional)
    os.remove(spectrogram_path)
    
    return class_label


# Example usage
audio_file = "D:\\datasets\\TORGO male without dysarthria\\MC01\\Session1\\wav_arrayMic\\0015.wav"  # Replace with the path to your audio file
result = predict_dysarthria(audio_file)
print(f"Prediction: {result}")
