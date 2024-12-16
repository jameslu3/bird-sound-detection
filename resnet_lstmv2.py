import tensorflow as tf
from tensorflow import keras
from keras import regularizers
import keras_cv

def create_resnet_lstm_model(CFG):
    # Create an input layer for the model
    inp = keras.layers.Input(shape=(CFG.time_steps, CFG.img_size[0], CFG.img_size[1] // CFG.time_steps, 3))
    #inp = keras.layers.Input(shape=(None, None, 3))
    
    # Load the ResNet50 backbone
    backbone = keras_cv.models.ResNet50Backbone.from_preset(
        CFG.preset,
    )

    # Stack custom layers on top of the backbone
    x = keras.layers.TimeDistributed(backbone)(inp, training=False)
    x = keras.layers.TimeDistributed(keras.layers.GlobalAveragePooling2D())(x)

    # Add an LSTM layer
    x = keras.layers.LSTM(64, return_sequences=True, dropout=0.2, kernel_regularizer=regularizers.l2(0.01))(x)
    x = keras.layers.LSTM(64, return_sequences=False, dropout=0.2, kernel_regularizer=regularizers.l2(0.01))(x)

    x = keras.layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = keras.layers.Dropout(0.2)(x)  # Add dropout for regularization

    out = keras.layers.Dense(CFG.num_classes, activation="softmax")(x)

    # Build the model
    model = keras.models.Model(inputs=inp, outputs=out)

    # Compile the model
    model.compile(
        optimizer="adam",
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
        metrics=[keras.metrics.AUC(name='auc')],
    )

    # Display the model summary
    model.summary()
    return model 