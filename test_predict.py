import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

model = load_model('model.h5')

def p(image_path):
    img = load_img(image_path, target_size=(225, 225))
    x = img_to_array(img).astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)[0]
    return pred

print('Healthy:', p('Dataset/Test/Test/Healthy/8ddaeccbf23485ab.jpg'))
print('Powdery:', p('Dataset/Test/Test/Powdery/8f2f6460fae2447d.jpg'))
print('Rust:', p('Dataset/Test/Test/Rust/82f49a4a7b9585f1.jpg'))
