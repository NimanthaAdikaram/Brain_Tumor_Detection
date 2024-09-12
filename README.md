# Deep Learning Brain Tumor Detection Model

This project demonstrates a deep learning pipeline for image classification using TensorFlow, OpenCV, and Matplotlib. The model is trained on a dataset of images, using a Convolutional Neural Network (CNN) to classify images into one of four classes.

## Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Setup and Installation](#setup-and-installation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project focuses on building a CNN model to classify images into four categories. The pipeline involves:

1. Loading and preprocessing the data.
2. Splitting the data into training, validation, and test sets.
3. Building a deep learning model using TensorFlow and Keras.
4. Training and evaluating the model.

## Dependencies

Make sure you have the following dependencies installed:

- Python 3.7+
- TensorFlow
- OpenCV
- Matplotlib
- NumPy

## Dataset

The dataset consists of 3,224 images belonging to 4 classes. Ensure that the images are organized in a folder structure.

## Setup and Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/NimanthaAdikaram/Brain_Tumor_Detection
   cd repo-name

2. **Install the required dependencies:**

   ```bash
   pip install tensorflow opencv-python matplotlib

2. **Run the code to train the model:**
Ensure you have the dataset folder named data in your working directory.

## Model Architecture

The CNN model has the following architecture:

Convolutional layers with ReLU activation and Max Pooling.
Flatten layer to convert 2D features to 1D.
Dense layers with Dropout to prevent overfitting.
Softmax activation for multiclass classification.

## Training

The model is trained with 70% of the data, validated with 20%, and tested with the remaining 10%.
The data is normalized by scaling pixel values to a range of 0-1.

## Results
Display your model's accuracy, loss graphs, and any other relevant metrics.
Include visualizations of sample predictions, confusion matrices, or other evaluation metrics.

## Usage

1. To train the model, execute:
   ```bash
   python train.py
2. Update the paths in the code if your dataset directory differs.
3. For testing the model, use:

## Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
1. Create a new branch (git checkout -b feature-branch).
2. Commit your changes (git commit -m 'Add new feature').
3. Push to the branch (git push origin feature-branch).
4. Open a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

Just copy and paste this directly into your README file!

![TU](https://github.com/user-attachments/assets/d851d70f-4eb8-4a06-8ae9-ee0a272f901a)
