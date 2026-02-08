import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

model = tf.keras.models.load_model('/models/cifar10_model.keras')

def predict_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)

            img = image.load_img(img_path, target_size=(75, 75))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array, verbose=0)
            score = tf.nn.softmax(predictions[0])
            class_idx = np.argmax(score)

            print(
                f"{filename} → {class_names[class_idx]} "
                f"({100 * np.max(score):.2f}% confidence)"
            )

predict_folder("/your_folder")
