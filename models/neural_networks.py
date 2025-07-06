import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential             # type: ignore
from tensorflow.keras.layers import Dense, Input           # type: ignore




# A simple Neural Model
def neural_model(input_dimension):
    model = Sequential(
        [  
            Input(shape=(input_dimension,)),
            Dense(18,activation='relu'),
            Dense(13,activation='relu'),
            Dense(8,activation ='relu'),
            Dense(1,activation = 'sigmoid')
        ]
    )
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )
    return model


def train_neural_network(model, X_train, y_train):
    history = model.fit(
        X_train,
        y_train,
        epochs=60,
        validation_split = 0.25
    )
    return history