import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from PIL import Image

img_width = 224
img_height = 224
batch_size = 64

def load_keras_model(model_filepath):

    return tf.keras.models.load_model(
        model_filepath,
        custom_objects={
            'KerasLayer': hub.KerasLayer
        }
    )

def process_image(np_image):
    
    image_tensor = tf.image.convert_image_dtype(np_image, dtype=tf.float32)
    
    return tf.image.resize(image_tensor, [img_height, img_width]).numpy()

def predict(image_path, model, top_k=1):
    
    im = Image.open(image_path)
    test_image = np.asarray(im)

    processed_test_image = process_image(test_image)
    
    image = tf.expand_dims(processed_test_image, 0)
    
    result = model.predict(image)
    
    result_dict = dict(enumerate(result[0]))

    sorted_list_by_value = sorted(result_dict, key=result_dict.__getitem__)
    
    sorted_list_by_value.reverse()
    
    probs = []
    classes = []
    
    for index in sorted_list_by_value[:top_k]:
        
        tmp_class = str(index + 1)
        tmp_prob = result_dict[index]
        
        probs.append(tmp_prob)
        classes.append(tmp_class)
    
    
    return probs, classes