import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
import os


# Define a directory to save the model

# Load the ResNet50 model pre-trained on ImageNet data
base_model = ResNet50(weights='imagenet', include_top=False)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Add a global spatial average pooling layer
x = Dense(1024, activation='relu')(x)  # Add a fully-connected layer
predictions = Dense(1, activation='sigmoid')(x)  # Add a logistic layer for binary classification

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Function to load and preprocess the image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Image path
img_path = 'normal/normalPollution/Pol_15.jpg'

# Load and preprocess the image
processed_image = load_and_preprocess_image(img_path)

# Define the directory to save the model
model_dir = 'models'

# Create the directory if it doesn't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Use the tf.saved_model.save() method to save the model in SavedModel format
tf.saved_model.save(model, model_dir)

print(f"Model saved in the directory: {model_dir}")


# Make a prediction
predictions = model.predict(processed_image)

# Convert predictions to binary outcome
predicted_class = 'Smoke' if predictions[0] > 0.5 else 'No Smoke'
print(f"The image was predicted as: {predicted_class}")
