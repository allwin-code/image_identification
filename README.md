# CNN Beginner Project

## Project Description
Image classification using a Convolutional Neural Network
with TensorFlow and CIFAR-10.

## Dataset
CIFAR-10
- 50,000 training images
- 10,000 testing images
- 10 classes
- Image size: 32×32×3

## CNN Architecture
Conv2D → MaxPooling
→ Conv2D → MaxPooling
→ Flatten → Dense → Softmax

## Training
Epochs: 10
Batch size: 64
Optimizer: Adam

## Result
Test Accuracy: 68.49%

## Prediction
The project includes predict.py, which can load
the trained model and predict a new image.
