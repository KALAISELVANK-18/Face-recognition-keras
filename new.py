from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import cv2
import os
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("transfer_learning_trained_face_cnn_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# capture frames from a camera
img = cv2.imread("client.jpg")
# Detects faces of different sizes in the input image
faces = face_cascade.detectMultiScale(img, 1.3, 5)

i = 0
for (x, y, w, h) in faces:
    bee = img[y:y + h, x:x + w]
    zoo = 'kar' + str(i+1) + '.jpg'
    i += 1

    cv2.imwrite(zoo,bee)

    # To draw a rectangle in a face
    cv2.rectangle(img, (x, y), (x + w, y + h), (250, 250, 2), 2)
    roi_gray = img[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open('kar' + str(i) + '.jpg').convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    imgv=np.resize(bee, size)

    image_array = np.asarray(image) #change to bee
    color_arr = np.dstack((imgv, imgv, imgv))

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)

    index = np.argmax(prediction)

    class_name = class_names[index]
    confidence_score = prediction[0][index]
    print("Class:", class_name[2:], end="")
    print(prediction)
    print("Confidence Score:", confidence_score)

#     # Display an image in a window
#     cv2.imshow('img', img)
#
#     # Wait for Esc key to stop
#     cv2.waitKey(2000)
#
# # De-allocate any associated memory usage
# cv2.destroyAllWindows()