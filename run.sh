#!/bin/bash
# Run mnist_cnn_model.py
echo "Running mnist_cnn_model.py..."
python mnist_cnn_model.py

# Tests

echo "I am not sure what is the file type of test dataset, so I write a script to make 10 test dataset with different noise level and test them"
python test_model.py