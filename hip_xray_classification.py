
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from scipy.stats import skew
import matplotlib.pyplot as plt

def measure_luminosity_and_inclination(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Measure luminosity as the mean brightness
    luminosity = np.mean(gray)
    
    # Measure inclination by calculating the skewness of edges
    edges = cv2.Canny(gray, 100, 200)
    inclination = skew(edges.flatten())
    
    return luminosity, inclination

def analyze_dataset(dataset_path):
    luminosity_list = []
    inclination_list = []
    
    for subdir, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(subdir, file)
                image = cv2.imread(image_path)
                luminosity, inclination = measure_luminosity_and_inclination(image)
                luminosity_list.append(luminosity)
                inclination_list.append(inclination)
    
    return luminosity_list, inclination_list

dataset_path = 'HIPS'
luminosity_list, inclination_list = analyze_dataset(dataset_path)

print(f"Average Luminosity: {np.mean(luminosity_list)}")
print(f"Average Inclination: {np.mean(inclination_list)}")

# Plotting the distribution of luminosity and inclination
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(luminosity_list, bins=20, color='blue', alpha=0.7)
plt.title('Luminosity Distribution')
plt.subplot(1, 2, 2)
plt.hist(inclination_list, bins=20, color='green', alpha=0.7)
plt.title('Inclination Distribution')
plt.show()

# Define image data generators for training with augmentation and validation without augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

# Load training and validation datasets
train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_path, 'train'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(dataset_path, 'val'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load ResNet50 with pre-trained weights, exclude the top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers for classification
x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
output = Dense(train_generator.num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=output)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Unfreeze some layers for fine-tuning
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Recompile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training the model
fine_tune_history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Save the model
model.save('hip_xray_classification_model.h5')
