import flwr as fl
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential

test_dir = "processed_fl/test_data"

test_data_gen = ImageDataGenerator(rescale=1. / 225 )

test_data = test_data_gen.flow_from_directory(test_dir, target_size=(120,120), class_mode='categorical', batch_size=16)
   
def get_model():
    
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
    return model

def weighted_average(metrics):
    
    
    
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_evaluate_fn():
   
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config
    ):
        model = get_model()  
        model.set_weights(parameters)  
        
        loss, accuracy = model.evaluate(test_data, verbose=0)
        model.save("brain_tumaor_fl_model.h5")
        return loss, {"accuracy": accuracy}

    return evaluate



strategy = fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,  
        evaluate_fn=get_evaluate_fn(),  
    )


fl.server.start_server(
    server_address = 'localhost:'+str(sys.argv[1]) , 
    config=fl.server.ServerConfig(num_rounds=1) ,
    strategy = strategy
)
