# Brain Tumor Image Classification using CNN

This project builds a Convolutional Neural Network (CNN) model using TensorFlow/Keras to classify brain MRI images into four categories:

- `category1_tumor`
- `category2_tumor`
- `category3_tumor`
- `no_tumor`

The workflow is implemented in the Jupyter Notebook `Brain_Jupyter.ipynb`. It includes dataset loading, preprocessing, model building, training, performance visualization, evaluation, single-image prediction, and model saving.


## Project Overview

The main objective of this project is to classify brain MRI images using a deep learning image classification model. The model is trained on images stored in a directory-based dataset and uses a CNN architecture with convolutional layers, max pooling, dense layers, dropout, and a softmax output layer.

The notebook follows this workflow:

1. Install and import dependencies
2. Load the image dataset
3. Normalize image pixel values
4. Split the dataset into training, validation, and testing sets
5. Build a CNN model
6. Train the model
7. Plot training and validation performance
8. Evaluate the model using precision, recall, and accuracy
9. Test the model on a single image
10. Save the trained model

---

## Dependencies

Install the required Python packages:

```bash
pip install tensorflow opencv-python matplotlib numpy
```

Main libraries used:

```python
import tensorflow as tf
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
```



## Dataset Structure

The dataset should be arranged in a folder named `data`, where each class has its own subfolder.

Example structure:

```text
project-folder/
│
├── Brain_Jupyter.ipynb
├── README.md
├── data/
│   ├── category1_tumor/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   │
│   ├── category2_tumor/
│   │   └── ...
│   │
│   ├── category3_tumor/
│   │   └── ...
│   │
│   └── no_tumor/
│       └── ...
│
├── logs/
└── models/
```

The dataset is loaded using:

```python
data = tf.keras.utils.image_dataset_from_directory(
    'data',
    label_mode='categorical'
)
```

The model expects images resized to `256 x 256 x 3`.



## Data Preprocessing

Image pixel values are normalized from the range `0–255` to `0–1`:

```python
data = data.map(lambda x, y: (x / 255, y))
```

The dataset is then split as follows:

```python
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2)
test_size = int(len(data) * 0.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)
```

Dataset split:

| Dataset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 20% |
| Testing | 10% |

---

## Model Architecture

The CNN model is built using Keras `Sequential` API.

```python
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
```

The final dense layer has 4 output neurons because the dataset has 4 classes.


## Model Compilation

The model is compiled using the Adam optimizer and categorical cross-entropy loss:

```python
model.compile(
    optimizer='adam',
    loss=tf.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)
```


## Training

The model is trained for 20 epochs:

```python
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(
    train,
    epochs=20,
    validation_data=val,
    callbacks=[tensorboard_callback]
)
```

TensorBoard logs are saved in the `logs` directory.

To open TensorBoard:

```bash
tensorboard --logdir logs
```


## Training Results

The notebook obtained the following final training performance:

| Metric | Value |
|---|---:|
| Final training accuracy | 97.23% |
| Final validation accuracy | 95.31% |
| Final training loss | 0.0669 |
| Final validation loss | 0.3158 |

The notebook also plots:

- Training loss vs validation loss
- Training accuracy vs validation accuracy



## Evaluation Results

The model was evaluated using precision, recall, and categorical accuracy.

```python
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

pre = Precision()
re = Recall()
acc = CategoricalAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')
```

Obtained test results:

| Metric | Value |
|---|---:|
| Precision | 0.9338 |
| Recall | 0.9250 |
| Accuracy | 0.9281 |



## Single Image Prediction

A test image is loaded using OpenCV, converted from BGR to RGB, resized to `256 x 256`, normalized, and passed into the trained model.

```python
img = cv2.cvtColor(cv2.imread('nt_img (158).jpg'), cv2.COLOR_BGR2RGB)
resize = tf.image.resize(img, (256, 256))

yhat = model.predict(np.expand_dims(resize / 255, axis=0))
predicted_class_index = np.argmax(yhat)
```

Class mapping used in the notebook:

```python
if predicted_class_index == 0:
    print('category1_tumor')
elif predicted_class_index == 1:
    print('category2_tumor')
elif predicted_class_index == 2:
    print('category3_tumor')
else:
    print('no_tumor')
```

Example prediction from the notebook:

```text
no_tumor
```



## Save and Load the Model

Create the `models` folder before saving:

```python
os.makedirs('models', exist_ok=True)
model.save(os.path.join('models', 'Braintumor.h5'))
```

To load the saved model:

```python
from tensorflow.keras.models import load_model

new_model = load_model('models/Braintumor.h5')
```



## Important Notes

The notebook contains two small points that should be corrected for cleaner use:

1. During training, images are scaled using `/255`, so inference images should also be scaled using `/255`, not `/256`.

   Recommended:

   ```python
   yhat = model.predict(np.expand_dims(resize / 255, axis=0))
   ```

2. The model is saved as:

   ```python
   models/Braintumor.h5
   ```

   Therefore, it should be loaded using the same path:

   ```python
   load_model('models/Braintumor.h5')
   ```


## How to Run

1. Clone or download the project folder.
2. Place the dataset inside the `data` directory using class-wise folders.
3. Install dependencies:

   ```bash
   pip install tensorflow opencv-python matplotlib numpy
   ```

4. Open the notebook:

   ```bash
   jupyter notebook Brain_Jupyter.ipynb
   ```

5. Run all cells sequentially.
6. Check training graphs, evaluation metrics, and single-image prediction.
7. Save the trained model inside the `models` directory.


## Limitations

- The model is trained on a custom dataset, so performance depends strongly on dataset quality and size.
- The notebook uses a basic CNN architecture and does not include transfer learning.
- The test split is created from the dataset pipeline and may vary depending on dataset loading order.
- This project is intended for educational and research purposes only. It should not be used as a medical diagnosis system without proper clinical validation.


## Future Improvements

Possible improvements include:

- Add data augmentation to improve generalization
- Use transfer learning models such as VGG16, ResNet50, EfficientNet, or MobileNetV2
- Add confusion matrix and classification report
- Save class names automatically from `data.class_names`
- Use a separate fixed test dataset
- Add early stopping and model checkpoint callbacks
- Convert the model to TensorFlow Lite for deployment
- Build a simple web or desktop interface for prediction



