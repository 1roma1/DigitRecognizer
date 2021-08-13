import os
import pandas as pd
import matplotlib.pyplot as plt

from app.models.data.data_preprocessing import preproc_mnist
from model import get_net

batch_size = 128
epochs = 15

x_train, y_train, x_test, y_test = preproc_mnist()

model = get_net()
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=batch_size, 
                    epochs=epochs, validation_split=0.1)

model.save(os.path.join(ROOT_PATH, '../models/model.h5'))
print("Saving the models as model.h5")

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.savefig(os.path.join(ROOT_PATH, "../figures/learning_curves.png"), format="png", dpi=300)
plt.show()