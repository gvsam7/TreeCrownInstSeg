"""
Author: Georgios Voulgaris
Date: 19/11/2024
Version: 1
Description: This script creates separate JSON files for the train, validation, and test sets. Each subset (train, val,
             test) will have its own annotations.json file containing the relevant annotations for that subset.

             Methodology:
             •  Splits the data into train, validation, and test sets using stratification.
             •  Copies the images to their respective directories (train/images, val/images, test/images).
             •  Creates separate JSON files (annotations.json) for each subset, containing only the annotations relevant
                to the images in that subset.

                 Structure:
                    COCO_Data/
                    └── images/
                    ├── data/
                    │   ├── train/
                    │   │   ├── images/
                    │   │   └── annotations/
                    │   │       └── annotations.json
                    │   ├── val/
                    │   │   ├── images/
                    │   │   └── annotations/
                    │   │       └── annotations.json
                    │   └── test/
                    │       ├── images/
                    │       └── annotations/
                    │           └── annotations.json
                    └── annotations.json
"""

import os
import json
import shutil
from sklearn.model_selection import train_test_split

# Paths
images_dir = "COCO_Data/images/"
annotations_file = os.path.join(images_dir, "annotations.json")
output_dir = "COCO_Data/images/data"

# Load Annotations
with open(annotations_file, 'r') as f:
    data = json.load(f)

# Extract image filenames and corresponding labels
image_files = [img["file_name"] for img in data["images"]]
image_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
valid_annotations = [ann for ann in data["annotations"] if ann["image_id"] in image_id_to_file]

# Create a mapping from image filenames to their labels
image_to_labels = {img["file_name"]: [] for img in data["images"]}
for ann in valid_annotations:
    image_file = image_id_to_file[ann["image_id"]]
    image_to_labels[image_file].append(ann["category_id"])

# Use the first label for stratification purposes
image_files = list(image_to_labels.keys())
labels = [labels[0] for labels in image_to_labels.values()]

# Ensure the lengths match
print(f"Number of images: {len(image_files)}")
print(f"Number of labels: {len(labels)}")

# Split data with stratification
train_files, test_files, train_labels, test_labels = train_test_split(
    image_files, labels, test_size=0.2, random_state=21, stratify=labels
    )
train_files, val_files, train_labels, val_labels = train_test_split(
    train_files, train_labels, test_size=0.2, random_state=21, stratify=train_labels
    )


# Function to copy files
def copy_files(file_list, subset_name):
    subset_dir = os.path.join(output_dir, subset_name)
    os.makedirs(subset_dir, exist_ok=True)
    os.makedirs(os.path.join(subset_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(subset_dir, 'annotations'), exist_ok=True)

    subset_annotations = {"info": data["info"], "licences": data["licences"], "images": [], "annotations": [],
                          "categories": data["categories"]}

    for file in file_list:
        # Copy image
        shutil.copy(os.path.join(images_dir, file), os.path.join(subset_dir, "images", file))
        # Copy corresponding annotations
        image_id = next(img["id"] for img in data["images"] if img["file_name"] == file)
        subset_annotations["images"].append(next(img for img in data["images"] if img["id"] == image_id))
        subset_annotations["annotations"].extend([ann for ann in data["annotations"] if ann["image_id"] == image_id])

        # Save annotations
        with open(os.path.join(subset_dir, "annotations", "annotations.json"), "w") as f:
            json.dump(subset_annotations, f, indent=4)


# Copy files to respective directories
copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

print("Files have been successfully separated into train, validation, and test sets with stratification.")