#This will include various data preprocessing, which can be reused for other files.

import numpy as np
from tensorflow.keras.preprocessing import image

def load_image(img_path, target_size=(150, 150)):
    """Load and preprocess an image for prediction."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def mock_train_data():
    """Returns mock training data (features and labels)"""
    # Randomly generated data 
    X_train = np.random.random((10, 150, 150, 3))
    y_train = np.random.randint(0, 2, (10,))
    return X_train, y_train
