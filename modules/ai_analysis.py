import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/tumor_detection.h5")

def analyze_dicom(image):
    """ Run AI analysis on a DICOM image slice. """
    image = cv2.resize(image, (128, 128)) / 255.0
    image = np.expand_dims(image, axis=[0, -1])
    prediction = model.predict(image)
    
    if prediction > 0.5:
        return "Possible Tumor Detected"
    return "No Tumor Detected"
