import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Paths to the audio folders
audio_folders = [
    "D:/datasets/TORGO male without dysarthria",
    "D:/datasets/TORGO female without dysarthria",
    "D:/datasets/TORGO male with dysarthria",
    "D:/datasets/TORGO female with dysarthria"
]

# Corresponding output folders for spectrograms
output_folder_dysarthric = "D:/datasets/Spectrograms/dysarthric"
output_folder_non_dysarthric = "D:/datasets/Spectrograms/non_dysarthric"

# Function to check if the file is a valid .wav file
def is_valid_wav(file_path):
    try:
        # Try loading the audio file
        audio, _ = librosa.load(file_path, sr=None)
        return True
    except:
        # If loading fails, it's likely not a valid .wav file
        return False

# Function to generate and save spectrograms
def save_spectrogram(audio_path, output_path):
    audio, sr = librosa.load(audio_path, sr=22050)
    # Generate Mel spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Save the spectrogram as an image
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Loop through each folder, generate spectrograms, and save them
for folder_path in audio_folders:
    # Determine the class based on folder name
    if "with dysarthria" in folder_path.lower():
        output_folder = output_folder_dysarthric
    else:
        output_folder = output_folder_non_dysarthric

    # Make sure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Walk through each folder and subfolder
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".wav"):
                file_path = os.path.join(root, file)
                
                # Check if the .wav file is valid
                if is_valid_wav(file_path):
                    # Create a unique name by including parts of the path
                    relative_path = os.path.relpath(file_path, folder_path)
                    unique_filename = os.path.splitext(relative_path.replace(os.sep, '_'))[0] + ".png"
                    output_path = os.path.join(output_folder, unique_filename)
                    
                    # Generate and save the spectrogram
                    save_spectrogram(file_path, output_path)
                    print(f"Saved spectrogram for {file_path} to {output_path}")

print("Spectrogram generation completed for all audio files.")
