from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import requests
import tensorflow as tf

# fname = "class.txt"
# with open(fname ,"r") as f:
#     class_labels = sorted(set([word for line in f for word in line.split()]))

class_labels = ['glioma', 'meningioma','notumor', 'pituitary']

def load_and_prep_image(filepath, image_size):
    img = tf.io.read_file(filepath) #read image

    img = tf.io.decode_image(img,channels=3) 
    # img = tf.image.decode_image(img) # decode the image to a tensor
    img = tf.image.resize(img, size = [image_size, image_size]) # resize the image
    img = img/255. # rescale the image
    return img

def classify_image(filepath, model_path, image_size=229, class_labels=class_labels):
    # loading trained model
    # trained_model=load_model(model_path)
    trained_model=load_model(model_path, compile=False)
    trained_model.compile(loss = "categorical_crossentropy" ,
                optimizer = tf.keras.optimizers.Adam(learning_rate= 0.0001), #--< When FineTuning u want to lower the LR by 10x
                metrics = ["accuracy"]
               )

    # Import the target image and preprocess it
    img = load_and_prep_image(filepath, image_size)

    prediction = trained_model.predict(tf.expand_dims(img, axis=0))
    index = np.argmax(prediction)

    class_name = class_labels[index]
    confidence_score = prediction[0][index]

    return {
        'class' : class_name,
        'score' : f'{confidence_score*100:02.2f}%'
    }

# google t m
def predict_class(filepath, model_path, image_size = 299):
    np.set_printoptions(suppress=True)

    model = load_model(model_path ,compile=False)
    model.compile(loss = "categorical_crossentropy" ,
                optimizer = tf.keras.optimizers.Adam(learning_rate= 0.0001), 
                metrics = ["accuracy"]
               )

    data = np.ndarray(shape=(1, image_size, image_size, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(filepath).convert("RGB")

    # resizing the image to be at least 299 X 299 and then cropping from the center
    size = (image_size, image_size)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_labels[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    # print("Class:", class_name[2:], end=" \n")
    # print("Confidence Score:", confidence_score)

    result = {
        "class" : class_name,
        "score" :f'{(confidence_score*100):2.2f}%'
    }

    return result

# for bytes
def prepare_image(image, image_size):
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.cast(image, tf.float32)
    image /= 255.0
    image = tf.image.resize(image, [image_size, image_size])

    image = np.expand_dims(image, axis=0)

    return image

def classify_using_bytes(image_bytes, model_path, image_size):
    model = load_model(model_path, compile=False)
    model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

    prediction = model.predict(prepare_image(image_bytes, image_size))
    index = np.argmax(prediction, axis=1)[0]

    class_name = class_labels[index]
    confidence_score = prediction[0][index]

    return {
        'class' : class_name,
        'score' : f'{confidence_score*100:02.2f}%'
    }

def classify_using_url(url, model_path, image_size=299):
    image_source = requests.get(url).content
    
    return classify_using_bytes(image_source, model_path, image_size)

if __name__ == '__main__':
    pth = "pituitary.jpg"
    print(predict_class(pth, 'brain_tumor_fl_model.h5', 224))
    print(predict_class(pth, 'brain_tumor_vgg16_epochs_10.h5', 224))
    # print(predict_class('COVID-8.png', 'diseases.h5'))
    # print(classify_using_url(
    #                         url = 'https://res.cloudinary.com/ddm2qblsr/image/upload/v1690679815/COVID-8_uqezzz.png', 
    #                         model_path='diseases.h5'))