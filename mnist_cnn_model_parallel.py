import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import numpy as np
import timeit
import joblib
import os
import multiprocessing

def add_gaussian_noise(images, mean=0, std=0.5):
    noise = np.random.normal(mean, std, images.shape)
    noisy_images = images + noise
    noisy_images = np.clip(noisy_images, 0, 1)
    return noisy_images

def train_cnn_with_noise(noise_std):
        # Create a new model for each process to avoid conflicts
        model_copy = models.clone_model(model)
        model_copy.build((None, 28, 28, 1))  # Build the model to compile it
        model_copy.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Add random Gaussian noise
        X_train_noisy = add_gaussian_noise(X_train, std=noise_std)

        # Train the model on noisy data
        model_copy.fit(X_train_noisy, y_train, batch_size=64, epochs=1, validation_data=(X_test, y_test))

        # Evaluate and save the model
        test_loss, test_acc = model_copy.evaluate(X_test, y_test, verbose=0)
        model_copy.save(f'mnist_cnn_model_{int(noise_std * 10)}.h5')
        print(f"Model with noise_std={noise_std:.1f} - Test accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    start_time = timeit.default_timer()

    ## Check GPU availability
    #physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #if len(physical_devices) > 0:
    #    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # Load and preprocess the MNIST dataset
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')

    # Normalize the data
    X = X / 255.0
    X = X.values.reshape(-1, 28, 28, 1)

    # Split the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert labels to categorical (one-hot encoding)
    y_train = tf.keras.utils.to_categorical(y_train.astype(int), 10)
    y_test = tf.keras.utils.to_categorical(y_test.astype(int), 10)
    joblib.dump((X_train, X_test, y_train, y_test), 'mnist_dataset.pkl')

    # Create noisy images for training the autoencoder
    X_train_noisy = add_gaussian_noise(X_train, std=0.5)
    X_test_noisy = add_gaussian_noise(X_test, std=0.5)

    # Define the autoencoder architecture
    autoencoder = models.Sequential([
        # Encoder
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),

        # Decoder
        layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same'),
        layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same'),
        layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    ])

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train the autoencoder
    autoencoder.fit(X_train_noisy, X_train, epochs=10, batch_size=128, validation_data=(X_test_noisy, X_test))

    # Save the trained autoencoder model
    autoencoder.save('denoising_autoencoder.h5')

    # Build the CNN model
    model = models.Sequential([
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last', input_shape=(28,28,1)),
        layers.BatchNormalization(),
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'),
        layers.BatchNormalization(),
 
        layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'),
        layers.Dropout(0.25),
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'),
        layers.BatchNormalization(),
        layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu', data_format='channels_last'),
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

    # Compile the CNN model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Define the number of processes for parallel computing
    num_processes = os.cpu_count()

    
    # Create a multiprocessing pool
    pool = multiprocessing.Pool(processes=num_processes)

    # Train CNN models with different levels of noise in parallel
    noise_std_values = np.linspace(0, 1, num=11)
    pool.map(train_cnn_with_noise, noise_std_values)
    # Save the trained MNIST classification model
    model.save('mnist_cnn_model.h5')
    print("MNIST Model saved as mnist_cnn_model.h5")
    # Close the pool
    pool.close()
    pool.join()

    elapsed = timeit.default_timer() - start_time
    print(f"Execution time: {elapsed} seconds")
