import tensorflow as tf
from tensorflow import keras
import keras_cv

def create_efficientnet_lstm_model(CFG):
    # Create an input layer for the model
    inp = keras.layers.Input(shape=(CFG.time_steps, CFG.img_size[0], CFG.img_size[1] // CFG.time_steps, 3))
    #inp = keras.layers.Input(shape=(None, None, 3))
    
    # Load the EfficientNet backbone
    backbone = keras_cv.models.EfficientNetB0Backbone.from_preset(
        CFG.preset,
    )

    # Stack custom layers on top of the backbone
    x = keras.layers.TimeDistributed(backbone)(inp, training=False)
    x = keras.layers.TimeDistributed(keras.layers.GlobalAveragePooling2D())(x)

    # Add an LSTM layer
    print("Shape before LSTM:", x.shape)
    x = keras.layers.LSTM(256, return_sequences=False)(x)

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