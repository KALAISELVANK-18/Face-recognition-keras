from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import io
import numpy as np
from PIL import Image
import cv2
import numpy as np
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import cv2


app = FastAPI()

@app.post("/upload")
async def upload(image: UploadFile = File(...)):

    # Read the image file as bytes
    contents = await image.read()

    image = Image.open(io.BytesIO(contents))

    image = np.array(image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("client.jpg", gray)

    gray, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    gray = cv2.bitwise_not(img_bin)

    kernel = np.ones((2, 1), np.uint8)
    img = cv2.erode(gray, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)


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
    listi=[]
    print(faces)
    for (x, y, w, h) in faces:
        bee = img[y:y + h, x:x + w]
        zoo = 'img' + str(i + 1) + '.jpg'
        i += 1

        cv2.imwrite(zoo, bee)

        # To draw a rectangle in a face
        cv2.rectangle(img, (x, y), (x + w, y + h), (250, 250, 2), 2)


        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Replace this with the path to your image
        image = Image.open('img' + str(i) + '.jpg').convert("RGB")

        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)



        image_array = np.asarray(image)


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
        listi.append(str(class_name[2:]))
    # Return the result as a JSON response
    return JSONResponse({'result':listi})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)