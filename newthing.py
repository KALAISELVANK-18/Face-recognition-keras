from PIL import Image
import os

def resize_image(image_path, new_size):
    # Open the image using Pillow
    image = Image.open(image_path)

    # Resize the image
    resized_image = image.resize(new_size)

    # Get the directory and filename of the original image
    directory, filename = os.path.split(image_path)

    # Create a new filename for the resized image
    resized_filename = "resized_" + filename

    # Save the resized image in the same folder
    resized_image.save(os.path.join(directory, resized_filename))

    # Close the image
    resized_image.close()

# Set the path to the folder containing the images
folder_path = r"C:\Users\kalai\Desktop\fuel\husk"

# Set the new size for the resized images
new_size = (224, 224)

# Traverse through the folder and its subfolders
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg','.webp')):
            # Get the full path of the image file
            image_path = os.path.join(root, file)

            # Resize the image
            resize_image(image_path, new_size)