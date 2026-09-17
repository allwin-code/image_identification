import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. CLASS NAMES
# ============================================================

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# ============================================================
# 2. LOAD CIFAR-10 DATASET
# ============================================================

print("\nLoading CIFAR-10 dataset...")

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.cifar10.load_data()
)

print("\nDataset loaded successfully!")

print("Training images shape:", x_train.shape)
print("Training labels shape:", y_train.shape)

print("Testing images shape:", x_test.shape)
print("Testing labels shape:", y_test.shape)


# ============================================================
# 3. NORMALIZE IMAGE PIXELS
# ============================================================

print("\nNormalizing images...")

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Normalization completed.")


# ============================================================
# 4. DISPLAY SAMPLE IMAGES
# ============================================================

print("\nDisplaying sample images...")

plt.figure(figsize=(10, 10))

for i in range(25):

    plt.subplot(5, 5, i + 1)

    plt.imshow(x_train[i])

    plt.title(class_names[y_train[i][0]])

    plt.axis("off")

plt.tight_layout()

plt.show()


# ============================================================
# 5. BUILD CNN MODEL
# ============================================================

print("\nBuilding CNN model...")

model = tf.keras.Sequential([

    # Input image: 32 x 32 RGB
    tf.keras.layers.Input(
        shape=(32, 32, 3)
    ),

    # First convolution layer
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu"
    ),

    # First pooling layer
    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # Second convolution layer
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    ),

    # Second pooling layer
    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # Convert feature maps into one-dimensional data
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    # Output layer
    # CIFAR-10 has 10 classes
    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])


# ============================================================
# 6. DISPLAY MODEL STRUCTURE
# ============================================================

print("\nCNN model created successfully!\n")

model.summary()


# ============================================================
# 7. COMPILE MODEL
# ============================================================

print("\nCompiling model...")

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)

print("Model compiled successfully.")


# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\n========================================")
print("STARTING CNN TRAINING")
print("========================================\n")

history = model.fit(

    x_train,

    y_train,

    epochs=10,

    batch_size=64,

    validation_split=0.1,

    verbose=1
)


# ============================================================
# 9. EVALUATE MODEL ON TEST DATA
# ============================================================

print("\n========================================")
print("TESTING THE TRAINED MODEL")
print("========================================\n")

test_loss, test_accuracy = model.evaluate(

    x_test,

    y_test,

    verbose=1
)


print("\nTest Loss:", test_loss)

print(
    "Test Accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)


# ============================================================
# 10. SAVE TRAINED MODEL
# ============================================================

print("\nSaving trained model...")

model.save("cnn_model.keras")

print("Model saved as: cnn_model.keras")


# ============================================================
# 11. CREATE ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Training and Validation Accuracy")

plt.legend()

plt.grid(True)

plt.savefig(
    "training_graph.png"
)

plt.show()


# ============================================================
# 12. CREATE LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training and Validation Loss")

plt.legend()

plt.grid(True)

plt.savefig(
    "loss_graph.png"
)

plt.show()


# ============================================================
# 13. TEST ONE IMAGE
# ============================================================

print("\n========================================")
print("TESTING ONE IMAGE")
print("========================================\n")


image_index = 0


# Get one test image
image = x_test[image_index]


# Get its real label
actual_label = y_test[image_index][0]


# Convert one image into a batch
image_batch = np.expand_dims(
    image,
    axis=0
)


# Ask the CNN to predict
prediction = model.predict(
    image_batch,
    verbose=0
)


# Find class with highest probability
predicted_label = np.argmax(
    prediction[0]
)


# Get confidence
confidence = (
    prediction[0][predicted_label] * 100
)


print(
    "Predicted class:",
    class_names[predicted_label]
)

print(
    "Actual class:",
    class_names[actual_label]
)

print(
    "Confidence:",
    round(confidence, 2),
    "%"
)


# Display prediction
plt.figure(figsize=(4, 4))

plt.imshow(image)

plt.title(
    "Predicted: "
    + class_names[predicted_label]
    + "\nActual: "
    + class_names[actual_label]
)

plt.axis("off")

plt.show()


# ============================================================
# 14. FINISHED
# ============================================================

print("\n========================================")
print("CNN PROJECT COMPLETED SUCCESSFULLY!")
print("========================================")

print("\nGenerated files:")

print("1. cnn_model.keras")
print("2. training_graph.png")
print("3. loss_graph.png")