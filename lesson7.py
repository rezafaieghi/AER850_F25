import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers


# For reproducability
np.random.seed(42)
keras.utils.set_random_seed(42)

# Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Split data into train, validation, and test subsets.
# The network will be trained on the train data.
# After each epoch, the network will be evaluated on the validation data.
# Validation data is used to monitor training.
# Once we finalize the model, we test it on the test data.

x_train, x_val = x_train[:50000], x_train[50000:]
y_train, y_val = y_train[:50000], y_train[50000:]

# The images in the dataset are grayscale with pixel values in [0, 255] (integers).
# We scale the data to [0,1] range. Generally, scaling data to [0,1] or [-1,1] 
#range is a good practice in training NNs.
x_train = (x_train / 255.0).astype("float32")
x_val   = (x_val   / 255.0).astype("float32")
x_test  = (x_test  / 255.0).astype("float32")

# Set batch size and epochs
BATCH_SIZE = 50000 
EPOCHS = 15


# Build a simple NN
# Flatten layer to flatten the 28x28 input images
# A densely connected hidden layer with 128 neurons
# A dropout layer that sets 20% of activation values of the previous latyer to
# zero, helpful to avoid over-fitting.
# A densely connected output layer with 10 neurons, one for each output class
mdl1 = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax")
        ])

# Set up the learning mechanism for the NN. Answer the following three questions:
# What optimization algorithm to use? This is the numerical method to solve the
# error backpropagation problem. Adam is by far the most common choice.
# (2) What loss function to minimize? Here the problem is a multi-class classification
# problem. Since the dataset labels are provided in a sparse format, we use
# sparse_categorical_crossentropy. We could also use categorical_crossentropy
# but that would require a bit of re-formatting in how the data labels are presented.
# (3) What metrics to monitor during training? This does not affect the training
# process. It is only for monitoring purposes.
mdl1.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
    )


# Early stopping monitors a chosen metric, e.g., validation accuracy while the
# is training. If that metric stops improving for several epochs, it automatically
# stops the training early before overfitting occurs.
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
    )

history1 = mdl1.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
    )

test_loss, test_acc = mdl1.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

plt.figure(figsize=(6,4))
plt.plot(history1.history["accuracy"], label="Train Acc")
plt.plot(history1.history["val_accuracy"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))
plt.plot(history1.history["loss"], label="Train Loss")
plt.plot(history1.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()
plt.grid(True)
plt.show()

pred_probs = mdl1.predict(x_test[:8])
pred_labels = np.argmax(pred_probs, axis=1)
print("Predicted labels:", pred_labels)
print("True labels     :", y_test[:8])


# Predict probabilities for first 8 test samples and print the results
pred_probs = mdl1.predict(x_test[:8])
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = y_test[:8]
print("Predicted labels:", pred_labels)
print("True labels     :", true_labels)

# Plot results for the first 8 test samples
plt.figure(figsize=(10, 3))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(x_test[i], cmap="gray")
    plt.axis("off")
    color = "green" if pred_labels[i] == true_labels[i] else "red"
    plt.title(f"P:{pred_labels[i]} / T:{true_labels[i]}", color=color)
plt.suptitle("Model Predictions on Test Samples", fontsize=14)
plt.tight_layout()
plt.show()



# # Build a deeper NN
# mdl2 = keras.Sequential([
#         layers.Flatten(input_shape=(28, 28)),
#         layers.Dense(256, activation="relu"),
#         layers.Dense(256, activation="relu"),
#         layers.Dense(128, activation="relu"),
#         layers.Dense(128, activation="relu"),
#         layers.Dense(64, activation="relu"),
#         layers.Dense(64, activation="relu"),
#         layers.Dense(32, activation="relu"),
#         layers.Dense(32, activation="relu"),
#         layers.Dropout(0.3),
#         layers.Dense(10, activation="softmax")
#         ])
# mdl2.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=1e-3),
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
#     )
# history2 = mdl2.fit(
#     x_train, y_train,
#     validation_data=(x_val, y_val),
#     epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     callbacks=[early_stop],
#     verbose=1
#     )
# test_loss, test_acc = mdl2.evaluate(x_test, y_test, verbose=0)
# print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

