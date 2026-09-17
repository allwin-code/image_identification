import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# CIFAR-10 class names
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


# Load trained CNN model
print("Loading trained CNN model...")

model = tf.keras.models.load_model("cnn_model.keras")

print("Model loaded successfully!")


# Ask user for image path
image_path = input("\nEnter the image path: ")


# Load image
image = tf.keras.utils.load_img(
    image_path,
    target_size=(32, 32)
)


# Convert image to NumPy array
image_array = tf.keras.utils.img_to_array(image)


# Normalize pixel values
image_array = image_array / 255.0


# Add batch dimension
image_batch = np.expand_dims(
    image_array,
    axis=0
)


# Predict
prediction = model.predict(
    image_batch,
    verbose=0
)


# Find highest probability
predicted_label = np.argmax(
    prediction[0]
)


# Get confidence
confidence = (
    prediction[0][predicted_label] * 100
)


# Display result
print("\n================================")
print("PREDICTION RESULT")
print("================================")

print(
    "Predicted class:",
    class_names[predicted_label]
)

print(
    "Confidence:",
    round(confidence, 2),
    "%"
)


# Display image
plt.imshow(image)

plt.title(
    "Predicted: "
    + class_names[predicted_label]
    + "\nConfidence: "
    + str(round(confidence, 2))
    + "%"
)

plt.axis("off")

plt.show()