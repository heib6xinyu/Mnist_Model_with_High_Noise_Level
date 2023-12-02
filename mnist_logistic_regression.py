import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

import timeit

def load_and_preprocess_data():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')

    # Normalize the data
    X /= 255.0
    return X, y

def train_logistic_regression(X_train, y_train):
    # Create the Logistic Regression model
    model = LogisticRegression(max_iter=1000, tol=0.1, solver='lbfgs', multi_class='auto')

    # Train the model
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

def save_model(model, filename):
    joblib.dump(model, filename)

def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    joblib.dump((X, y), 'mnist_data.pkl')


    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the logistic regression model
    model = train_logistic_regression(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Save the model
    save_model(model, 'mnist_logistic_regression.pkl')

if __name__ == "__main__":
    start_time = timeit.default_timer()

    main()
    elapsed = timeit.default_timer() - start_time
    print(f"Execution time: {elapsed} seconds")
