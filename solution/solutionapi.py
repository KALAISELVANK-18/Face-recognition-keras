from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import io
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2


app = FastAPI()

@app.post("/shell")
async def upload(image: UploadFile = File(...)):

    # Read the image file as bytes
    contents = await image.read()

    # Convert bytes to a PIL Image
    pil_image = Image.open(io.BytesIO(contents))

    # Convert PIL Image to a NumPy array
    image_array = np.array(pil_image)

    # Save the NumPy array as an image file
    cv2.imwrite("solution.jpg", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))

    model = load_model("shell_model.h5", compile=False)

    # Load the labels
    class_names = open("labels2.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open("solution.jpg").convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
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

    # Print prediction and confidence score

    print(class_name[2:],confidence_score)
    return JSONResponse({'result':class_name[2:]})

@app.post("/bagasse")
async def upload(image: UploadFile = File(...)):

    # Read the image file as bytes
    contents = await image.read()

    # Convert bytes to a PIL Image
    pil_image = Image.open(io.BytesIO(contents))

    # Convert PIL Image to a NumPy array
    image_array = np.array(pil_image)

    # Save the NumPy array as an image file
    cv2.imwrite("solution.jpg", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))

    model = load_model("bagasse_model.h5", compile=False)

    # Load the labels
    class_names = open("labels3.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open("solution.jpg").convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
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

    # Print prediction and confidence score

    print(class_name[2:],confidence_score)
    return JSONResponse({'result':class_name[2:]})

@app.post("/husk")
async def upload(image: UploadFile = File(...)):

    # Read the image file as bytes
    contents = await image.read()

    # Convert bytes to a PIL Image
    pil_image = Image.open(io.BytesIO(contents))

    # Convert PIL Image to a NumPy array
    image_array = np.array(pil_image)

    # Save the NumPy array as an image file
    cv2.imwrite("solution.jpg", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))

    model = load_model("husk_model.h5", compile=False)

    # Load the labels
    class_names = open("labels4.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open("solution.jpg").convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
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

    # Print prediction and confidence score

    print(class_name[2:],confidence_score)
    return JSONResponse({'result':class_name[2:]})

@app.post("/saw")
async def upload(image: UploadFile = File(...)):

    # Read the image file as bytes
    contents = await image.read()

    # Convert bytes to a PIL Image
    pil_image = Image.open(io.BytesIO(contents))

    # Convert PIL Image to a NumPy array
    image_array = np.array(pil_image)

    # Save the NumPy array as an image file
    cv2.imwrite("solution.jpg", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))

    model = load_model("saw_model.h5", compile=False)

    # Load the labels
    class_names = open("labels5.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open("solution.jpg").convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
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

    # Print prediction and confidence score

    print(class_name[2:],confidence_score)
    return JSONResponse({'result':class_name[2:]})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)