#This file is used to predict the surface irregularities using a trained model.


from tensorflow.keras.models import load_model
from utils import load_image

# Load pre-trained model into model
model = load_model('models/model.h5')

# Load image
img_path = '/path/to/your/image.jpg'
# Process image
img_array = load_image(img_path)

# Predict the class where an image belongs
prediction = model.predict(img_array)
if prediction < 0.5:
    print("No Irregularity Detected")
else:
    print("Irregularity Detected")
