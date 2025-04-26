import sys
import io

# Set the console output to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# Paths to spectrogram folders
train_data_path = "D:/datasets/Spectrograms"
dysarthric_folder = os.path.join(train_data_path, "dysarthric")
non_dysarthric_folder = os.path.join(train_data_path, "non_dysarthric")

# Create directories for training and validation sets
train_dir = os.path.join(train_data_path, 'train')
val_dir = os.path.join(train_data_path, 'val')

# Function to split data and create train/val folders if they don't exist
def split_data(source_folder, train_folder, val_folder, split_ratio=0.8):
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    
    # Get list of all files in the source folder
    files = [f for f in os.listdir(source_folder) if f.endswith('.png')]
    train_files, val_files = train_test_split(files, train_size=split_ratio, random_state=42)

    # Copy files to train folder
    for file in train_files:
        shutil.copy(os.path.join(source_folder, file), os.path.join(train_folder, file))

    # Copy files to val folder
    for file in val_files:
        shutil.copy(os.path.join(source_folder, file), os.path.join(val_folder, file))

# Split data for both classes
split_data(dysarthric_folder, os.path.join(train_dir, 'dysarthric'), os.path.join(val_dir, 'dysarthric'))
split_data(non_dysarthric_folder, os.path.join(train_dir, 'non_dysarthric'), os.path.join(val_dir, 'non_dysarthric'))

# Define the image data generator
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Create generators for train and validation datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    class_mode='binary',
    batch_size=32
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    class_mode='binary',
    batch_size=32
)

# Define the model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with progress tracking
epochs = 10
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    verbose=1  # Set to 1 for progress bar, can be set to 2 for less output
)

# Evaluate the model on the validation set
val_generator.reset()
predictions = model.predict(val_generator)
predicted_classes = np.round(predictions).astype(int)

# Print classification report
print("Classification Report:")
print(classification_report(val_generator.classes, predicted_classes))

# Save the model
model.save("dysarthric_classifier_model.h5")

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
