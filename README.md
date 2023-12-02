# MNIST Digit Recognition with Convolutional Neural Networks

This project is an implementation of a Convolutional Neural Network (CNN) for recognizing handwritten digits from the MNIST dataset. The CNN is trained to classify images of handwritten digits into the corresponding digit classes (0-9).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Prerequisites

Before running the project, make sure you have the following prerequisites installed:

- [Python](https://www.python.org/) (3.6 or higher)
- [Conda](https://docs.conda.io/en/latest/) (for managing dependencies)
- [Git](https://git-scm.com/) (for version control)

## Getting Started

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/heib6xinyu/Mnist_Model_with_High_Noise_Level.git
   ```

2. Change to the project directory:

   ```bash
   cd mnist-cnn
   ```

3. Create a Conda environment (optional but recommended):

   ```bash
   conda create --name mnist-cnn python=3.8
   ```

4. Activate the Conda environment:

   ```bash
   conda activate mnist-cnn
   ```

5. Install project dependencies:

   ```bash
   ./compile.sh
   ```

## Usage

To train and evaluate the CNN model on the MNIST dataset, you can run the following command:

```bash
python mnist_cnn_model.py
```

This will train the model, evaluate its accuracy, and save the trained model as `mnist_cnn_model.h5`.

```bash
python test_model.py
```

This file creates 10 test data, one with 0 noise, other with 9 level of gaussian noise. Then test the model on the 10 test data.

## Dependencies

The project depends on the following Python libraries and packages:

- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Joblib](https://joblib.readthedocs.io/)

All dependencies are listed in the `compile.sh` file and can be installed using Conda.

