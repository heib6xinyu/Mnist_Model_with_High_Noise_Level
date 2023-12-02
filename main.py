import sys
import numpy as np
import idx2numpy
from svmutil import svm_read_problem

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Check if the correct number of command-line arguments are provided
if len(sys.argv) != 3:
    print("Usage: python predict_svm.py input_file output_file")
    sys.exit(1)

# Load the input data in libsvm format
input_file = sys.argv[1]
output_file = sys.argv[2]

# Load libsvm data and convert it to a NumPy array
X_test, y_test = svm_read_problem(input_file)

# Convert data to a NumPy array
X_test = np.array(X_test)

# Load the model
model = load_model('./mnist_cnn_model.h5')
prediction_results = []

# Perform predictions using the loaded model
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)  # Convert y_test from one-hot if needed
accuracy = accuracy_score(y_true, predicted_classes)
print(f"Accuracy : {accuracy * 100:.2f}%")
# Assuming 'predictions' contains your prediction results as probabilities or one-hot encoded values

# Write the prediction results to the output file
with open(output_file, 'w') as f:
    for prediction in predictions:
        f.write(f"{prediction}\n")

print(f"Predictions saved to {output_file}")