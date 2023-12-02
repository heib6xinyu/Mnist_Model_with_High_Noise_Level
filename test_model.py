import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model


def add_gaussian_noise(images, mean=0, std=0):
    noise = np.random.normal(mean, std, images.shape)
    noisy_images = images + noise
    noisy_images = np.clip(noisy_images, 0, 1)
    return noisy_images

X_train, X_test, y_train, y_test = joblib.load('mnist_dataset.pkl')



# Load the model
model = load_model('./mnist_cnn_model.h5')
#autoencoder = load_model('./denoising_autoencoder.h5')


# Create a dictionary to hold datasets with different noise levels
noisy_datasets = {}

for i in range(10):
    std_dev = i  # Standard deviation for Gaussian noise
    noisy_datasets[f'N(0,{std_dev})'] = add_gaussian_noise(X_test, std=std_dev)

# Evaluate the model on each noisy dataset
for noise_level, noisy_data in noisy_datasets.items():
#    cleaned_data = autoencoder.predict(noisy_data)

    predictions = model.predict(noisy_data)
    predicted_classes = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)  # Convert y_test from one-hot if needed
    accuracy = accuracy_score(y_true, predicted_classes)
    print(f"Accuracy on {noise_level}: {accuracy * 100:.2f}%")