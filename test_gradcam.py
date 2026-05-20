import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.h5")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    input_shape = img_array.shape[1:]
    inputs = tf.keras.Input(shape=input_shape, name="gradcam_input")

    x = inputs
    conv_output = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_output = x

    if conv_output is None:
        raise ValueError(f"Layer not found in forward graph: {last_conv_layer_name}")

    grad_model = tf.keras.Model(inputs, [conv_output, x])

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        tape.watch(conv_outputs)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        print("Gradients are None!")
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()

def test_gradcam():
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print("Loading model...")
    model = load_model(model_path)
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
    if last_conv_layer_name is None:
        last_conv_layer_name = next((layer.name for layer in reversed(model.layers) if "conv" in layer.name), None)
    print(f"Last conv layer: {last_conv_layer_name}")

    if last_conv_layer_name is None:
        print("No convolutional layer found; cannot generate Grad-CAM heatmap.")
        return

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    height = int(input_shape[1] or 224)
    width = int(input_shape[2] or 224)
    channels = int(input_shape[3] or 3)
    img_array = np.random.random((1, height, width, channels)).astype('float32')
    
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
