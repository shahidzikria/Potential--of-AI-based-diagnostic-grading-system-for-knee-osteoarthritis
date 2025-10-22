import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Softmax, Concatenate, GlobalMaxPooling2D, Dropout
from tensorflow.keras.applications import DenseNet201, MobileNetV2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy

def build_model(input_shape=(224, 224, 3), num_classes=5):
    input_layer = Input(shape=input_shape)
    base1 = DenseNet201(include_top=False, weights='imagenet', input_tensor=input_layer)
    base2 = MobileNetV2(include_top=False, weights='imagenet', input_tensor=input_layer)

    base1.trainable = False
    base2.trainable = True

    out1 = base1.output
    out2 = base2.output
    concat = Concatenate()([out1, out2])
    gap = GlobalMaxPooling2D()(concat)
    x = Dense(786, activation="gelu")(gap)
    x = Dense(512, activation="gelu")(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="gelu")(x)
    x = Dense(64, activation="gelu")(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output)
    return model

def compile_model(model):
    model.compile(
        optimizer=SGD(learning_rate=0.01, momentum=0.05),
        loss=CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name='acc'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tfa.metrics.F1Score(num_classes=5)
        ]
    )
    return model

def train_model(model, X_train, y_train, X_val, y_val, batch_size=8, epochs=200):
    callback = ReduceLROnPlateau(monitor="val_loss", patience=3)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[callback]
    )
    return history
