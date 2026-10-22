"""Loads the trained digit model and provides"""
# Importing required libs
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image

# Loading model
model = load_model("digit_model.h5")


# Preparing and pre-processing the image
def preprocess_img(img_path):
    """Open an image, resize it to 224x224, scale to 0-1 and reshape to (1, 224, 224, 3)"""
    op_img = Image.open(img_path).convert("RGB") #Force 3 colour channels
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, 224, 224, 3)
    return img_reshape


# Predicting function
def predict_result(predict):
    """Return predicted digit (Index of the highest scored model)"""
    pred = model.predict(predict)
    return np.argmax(pred[0], axis=-1)
