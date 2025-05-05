from datetime import datetime
import logging
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


def preprocess_image(image):
    resized_image = pad_and_resize(image)

    # Return a normalised image
    return(img_to_array(resized_image) / 255.0)

start_time = time.strftime("%d%m%y_%H%M%S")
current_dir = os.getcwd()
images_dir = "FETAL_PLANES_ZENODO/Images/"
full_path = os.path.join(current_dir, images_dir)
print(f"Full path: {full_path}")

image_paths, labels = [], []
images = os.listdir(full_path)
for image in images:
    img_path = os.path.join(full_path, image)
    image_paths.append(img_path)
    labels.append(image)

df = pd.DataFrame({"image_path": image_paths, "label": labels})
print(f"Loaded {len(df)} images")

print("\nClass distribution:")
print(df["label"].value_counts())

sample_sizes = [Image.open(path).size for path in df["image_path"][:100]]  # Check first 100
print("\nSample image sizes:", set(sample_sizes))

def pad_and_resize(image, target_size=(224, 224)):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    
    # Get orig dimensions
    width, height = image.size
    target_ratio = 1.0  # Square for padding
    
    # Calc padding
    current_ratio = width / height
    if current_ratio > 1:
        new_height = int(width / target_ratio)
        padding = (new_height - height) // 2
        new_size = (width, new_height)
    else:
        new_width = int(height * target_ratio)
        padding = (new_width - width) // 2
        new_size = (new_width, height)
    
    padded = Image.new("RGB", new_size, (0, 0, 0))
    padded.paste(image, ((new_size[0] - width) // 2, (new_size[1] - height) // 2))
    
    return padded.resize(target_size)

datagen = ImageDataGenerator(
    preprocessing_function=lambda x: img_to_array(pad_and_resize(x)) / 255.0,
    rescale=None
)

# Create gen
generator = datagen.flow_from_dataframe(
    df,
    x_col="image_path",
    y_col="label",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

batch_images, batch_labels = next(generator)
print(f"Batch shape: {batch_images.shape}")  # Should be (32, 224, 224, 3)
print(f"Normalized range: Min {batch_images.min()}, Max {batch_images.max()}")
print(f"Labels shape: {batch_labels.shape}")
print(f"Labels: {batch_labels}")
