import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.h5")

def inspect_model():
    if not os.path.exists(model_path):
        return
    model = load_model(model_path)
    print(f"Model type: {type(model)}")
    model.summary()

if __name__ == "__main__":
    inspect_model()
