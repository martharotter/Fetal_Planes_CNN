import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

current_dir = os.getcwd()
images_dir = "FETAL_PLANES_ZENODO/Images/"
full_path = os.path.join(current_dir, images_dir)
print(f"Full path: {full_path}")

images = os.listdir(full_path)
image_paths = []
labels = []

for image in images:
    img_path = os.path.join(full_path, image)
    image_paths.append(img_path)
    labels.append(image)

# Create a DataFrame for easier metadata exploration
df = pd.DataFrame({"image_path": image_paths, "label": labels})
print(f"Loaded {len(df)} images.")

# Basic stats
print(df.head())
print("\nClass distribution:")
print(df["label"].value_counts())

# Check image extensions
df["extension"] = df["image_path"].apply(lambda x: os.path.splitext(x)[1])
print("\nFile extensions:")
print(df["extension"].value_counts())

image_paths, labels = [], []
images = os.listdir(full_path)
for image in images:
    img_path = os.path.join(full_path, image)
    image_paths.append(img_path)
    labels.append(image)

df = pd.DataFrame({"image_path": image_paths, "label": labels})
print(f"Loaded {len(df)} images")

# Explore metadata
print("\nClass distribution:")
print(df["label"].value_counts())

# Sample size check
sample_sizes = [Image.open(path).size for path in df["image_path"][:100]]  # Check first 100
print("\nSample image sizes:", set(sample_sizes))

# Visualize class distribution
plt.figure(figsize=(10, 6))
sns.countplot(x="label", data=df)
plt.title("Class Distribution")
plt.xticks(rotation=45)
plt.show()

# Pad and resize
def pad_and_resize(image, target_size=(224, 224)):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    
    # orig dimensions
    width, height = image.size
    target_ratio = 1.0 
    
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
    
    padded = Image.new("RGB", new_size, (0, 0, 0))  # Black padding
    padded.paste(image, ((new_size[0] - width) // 2, (new_size[1] - height) // 2))
    
    # Resize
    return padded.resize(target_size)

# Set up ImageDataGenerator for batch processing
datagen = ImageDataGenerator(
    preprocessing_function=lambda x: img_to_array(pad_and_resize(x)) / 255.0,  # Pad, resize, normalize
    rescale=None
)

# Create gen
generator = datagen.flow_from_dataframe(
    df,
    x_col="image_path",
    y_col="label",
    target_size=(224, 224),  # Final size after padding
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# Inspect batch
batch_images, batch_labels = next(generator)
print(f"Batch shape: {batch_images.shape}")  # Should be (32, 224, 224, 3)
print(f"Normalized range: Min {batch_images.min()}, Max {batch_images.max()}")


# Plot class distribution
plt.figure(figsize=(10, 6))
sns.countplot(x="label", data=df)
plt.title("Class Distribution")
plt.xticks(rotation=45)
plt.show()

# Check image sizes and break if different
reference_size = (647, 381)
all_same_size = True
not_reference_size = 0

for img_path in df["image_path"]:
    img = Image.open(img_path)
    current_size = img.size
    if current_size != reference_size:
        print(f"Found mismatch at {img_path}: {current_size} != {reference_size}")
        not_reference_size += 1
        all_same_size = False

if all_same_size:
    print(f"All images match size: {reference_size}")
else:
    print("Not all images are the same size. Must resize.")
    print(f"Number of images not matching size: {not_reference_size}")
    exit()

target_size = (224, 224)

# Load / resize all images
def load_and_resize_image(img_path, target_size):
    img = load_img(img_path)
    img = img.resize(target_size)
    return img_to_array(img)

def load_image(img_path):
    img = load_img(img_path)
    return img_to_array(img)

images = np.array([load_image(path) for path in df["image_path"]])
print(f"Images shape: {images.shape}")

images_normalized = images / 255.0
print(f"Processed images shape: {images_normalized.shape}")
print(f"Min: {images_normalized.min()}, Max: {images_normalized.max()}")