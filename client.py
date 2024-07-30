import flwr as fl
import sys, os
import pickle as pkl
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential

vgg = VGG16(include_top=False, weights='imagenet', input_shape=[224, 224, 3])
for layer in vgg.layers:
    layer.trainable = False

model = Sequential([
    vgg,
    GlobalAveragePooling2D(),
    Dense(200, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

train_dir = "processed_fl/client2"
test_dir = "processed_fl/test_data"

train_data_gen = ImageDataGenerator(rescale=1. / 225)
test_data_gen = ImageDataGenerator(rescale=1. / 225 )

train_data = train_data_gen.flow_from_directory(train_dir, target_size=(120,120), class_mode='categorical', batch_size=16)
test_data = test_data_gen.flow_from_directory(test_dir, target_size=(120,120), class_mode='categorical', batch_size=16)

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        print("client0 started..........")
        model.set_weights(parameters)
        r = model.fit(train_data, epochs=10, validation_data=test_data) 

        hist = r.history
        print("Fit history : " ,hist)
        return model.get_weights(), sum([len(fs) for rt, fl, fs in os.walk(train_dir)]), {} 

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(test_data) 
        print("Eval accuracy : ", accuracy)
        return loss, sum([len(fs) for rt, fl, fs in os.walk(test_dir)]), {"accuracy": accuracy} 


fl.client.start_numpy_client(
        server_address="localhost:"+str(sys.argv[1]), 
        client=FlowerClient()
)
