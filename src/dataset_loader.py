import os
import cv2
import numpy as np

def load_data(image_folder):
    images = []

    # Check if the directory exists
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"The directory {image_folder} does not exist.")

    # Iterate through all files in the specified folder
    for image_name in os.listdir(image_folder):
        # Construct the full image path
        img_path = os.path.join(image_folder, image_name)

        # Read the image
        img = cv2.imread(img_path)

        # Check if the image was successfully loaded
        if img is None:
            print(f"Warning: Unable to load image at {img_path}. Skipping...")
            continue

        # Resize the image to 128x128 pixels
        img = cv2.resize(img, (128, 128))

        # Normalize the image
        img = img / 255.0

        # Append the processed image to the list
        images.append(img)

    return np.array(images)

# Example usage of load_data function
if __name__ == "__main__":
    image_folder = "results/output_images"  # Change this to your image folder path
    try:
        images = load_data(image_folder)
        print(f"Loaded {images.shape[0]} images with shape {images.shape[1:]} each.")
    except Exception as e:
        print(str(e))
