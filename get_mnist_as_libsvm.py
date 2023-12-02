from sklearn.datasets import fetch_openml
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import train_test_split

import pandas as pd

# Fetch the MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=True, parser='auto')

# Convert target to numeric
y = y.astype(float)

# Normalize the data
X /= 255.0

# Ensure all features are numeric
X = X.apply(pd.to_numeric)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to libsvm format and save
dump_svmlight_file(X_test, y_test, "mnist.libsvm")
