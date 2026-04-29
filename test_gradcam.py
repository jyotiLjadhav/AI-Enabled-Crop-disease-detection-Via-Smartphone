import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image as PILImage
import matplotlib.cm as cm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.h5")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    found = False
    for layer in model.layers:
        if found:
            x = layer(x)
        if layer.name == last_conv_layer_name:
            found = True
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        
        preds = classifier_model(last_conv_layer_output)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    if grads is None:
        print("Gradients are None!")
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    return heatmap

def test_gradcam():
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print("Loading model...")
    model = load_model(model_path)
    last_conv_layer_name = next(
        (layer.name for layer in reversed(model.layers) if 'conv' in layer.name),
        None,
    )
    print(f"Last conv layer: {last_conv_layer_name}")

    img_array = np.random.random((1, 225, 225, 3)).astype('float32')
    
    try:
        print("Generating heatmap...")
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=0)
        if heatmap is not None:
            print("Heatmap generated successfully!")
            print(f"Heatmap shape: {heatmap.shape}")
            print(f"Heatmap max value: {np.max(heatmap)}")
        else:
            print("Heatmap generation failed")
    except Exception as e:
        print(f"Error in Grad-CAM: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gradcam()
