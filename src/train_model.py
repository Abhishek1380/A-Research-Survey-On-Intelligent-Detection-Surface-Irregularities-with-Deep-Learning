#This file is for traing the model. Dataset will to loaded to build CNN and train the model. 

from cnn_model import build_model
from utils import mock_train_data

# Load generated data
X_train, y_train = mock_train_data()

# Build the model
model = build_model()

# Model Training
history = model.fit(
    X_train,
    y_train,
    epochs=5,  
    batch_size=2
)


model.save('models/model.h5')
# print("Model saved to 'models/model.h5'")
