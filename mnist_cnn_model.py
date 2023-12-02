import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam



def add_gaussian_noise(images, mean=0, std=0.5):
    noise = np.random.normal(mean, std, images.shape)
    noisy_images = images + noise
    noisy_images = np.clip(noisy_images, 0, 1)
    return noisy_images




# Load and preprocess the MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')


# Normalize the data
X = X / 255.0

# Reshape the data to include channel dimension (1 for grayscale)
X = X.values.reshape(-1, 28, 28, 1)

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## Create noisy images for training the autoencoder
#X_train_noisy = add_gaussian_noise(X_train, std=0.5)
#X_test_noisy = add_gaussian_noise(X_test, std=0.5)

## Define the autoencoder architecture
#autoencoder = models.Sequential([
#    # Encoder
#    layers.Input(shape=(28, 28, 1)),
#    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#    layers.MaxPooling2D((2, 2), padding='same'),
#    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#    layers.MaxPooling2D((2, 2), padding='same'),

#    # Decoder
#    layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same'),
#    layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same'),
#    layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
#])

## Compile the autoencoder
#autoencoder.compile(optimizer='adam', loss='mean_squared_error')

#autoencoder.fit(X_train_noisy, X_train, epochs=10, batch_size=128,
#                validation_data=(X_test_noisy, X_test))

#autoencoder.save('denoising_autoencoder.h5')

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train.astype(int), 10)
y_test = to_categorical(y_test.astype(int), 10)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'),
    layers.BatchNormalization(),
 
    layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'),
    layers.Dropout(0.25),
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'),
    layers.BatchNormalization(),
    layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', activation='relu', data_format='channels_last'),
    layers.BatchNormalization(),

    layers.MaxPooling2D(pool_size=(2, 2), padding='valid', strides=2),
    layers.Dropout(0.25),
    layers.Flatten(),

    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    layers.Dense(1024, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax'),

])



# Optimizer
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

# Compile the model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])





# Train the model with data augmentation
num_epochs = 2
for epoch in range(num_epochs):
    # Add random Gaussian noise
    noise_std = np.random.uniform(0, 1)
    X_train_noisy = add_gaussian_noise(X_train, std=noise_std)
    
    # Train the model on noisy data
    model.fit(X_train_noisy, y_train, batch_size=64, epochs=1, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# Save the model
model.save('mnist_cnn_model.h5')
print("Model saved as mnist_cnn_model.h5")
