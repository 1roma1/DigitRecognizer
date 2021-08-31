import tensorflow as tf

from app.models.dataset.data_preprocessing import preproc_mnist

batch_size = 128
epochs = 15

def get_net():
    input_shape = (28, 28, 1)
    num_classes = 10

    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    return model

x_train, y_train, x_test, y_test = preproc_mnist()

model = get_net()
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=batch_size, 
                    epochs=epochs, validation_split=0.1)

model.save('trained/cnn_model.h5')

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])