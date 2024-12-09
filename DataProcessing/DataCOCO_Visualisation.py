"""
Author: Georgios Voulgaris
Date: 08/11/2024
Description: This script allows the visualisation of the COCO formatted dataset (created by DataCOCO_Format script) for
             visual inspection of the images, the bounding boxes, individual tree ID, and instance segmentation masks.
"""

# requires pycocotools and fiftyone
import fiftyone as fo
import fiftyone.zoo as foz
import pathlib
import random
import json

# data_path = pathlib.Path("COCO_Data/images/")
# json_path = pathlib.Path("COCO_Data/images/annotations.json")

data_path = pathlib.Path("COCO_Data/images/data/val/images/")
json_path = pathlib.Path("COCO_Data/images/data/val/annotations/annotations.json")


"""
# Load COCO annotations from the JSON file
with open(json_path, 'r') as f:
    coco_data = json.load(f)

# Try loading the existing dataset (if it exists)
dataset_name = "coco_dataset"  # Replace with your dataset name
try:
    # Load the existing dataset
    dataset = fo.load_dataset(dataset_name)
    print(f"Loaded existing dataset: {dataset_name}")
except:
    # Create a new FiftyOne dataset if it doesn't exist
    print(f"Dataset {dataset_name} not found. Creating a new dataset.")
    dataset = fo.Dataset(name=dataset_name)

    # Loop through the annotations and create samples for the dataset
    for image_info in coco_data['images']:
        image_id = image_info['id']
        image_filename = image_info['file_name']

        # Get annotations for this image
        image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

        # If there are annotations for this image, create a sample
        if image_annotations:
            detections = []
            for ann in image_annotations:
                # Get the category name
                category_id = ann['category_id']
                category_name = next(
                    (category['name'] for category in coco_data['categories'] if category['id'] == category_id), None)

                # Convert segmentation to points (for simplicity assuming polygons)
                segmentation = ann.get('segmentation', [])
                if segmentation and isinstance(segmentation, list) and len(segmentation) > 0:
                    points = segmentation[0]  # Assuming the first polygon in the list is the relevant one
                    detections.append(
                        fo.Detection(
                            label=category_name,
                            bounding_box=ann['bbox'],  # COCO bbox is [x, y, width, height]
                            segmentation=points
                        )
                    )

            # Add the image with its annotations to the dataset
            sample = fo.Sample(filepath=str(data_path / image_filename), detections=detections)
            dataset.add_sample(sample)

    print(f"Dataset {dataset_name} created.")

# Debug: Check how many images have annotations
print(f"Total samples with annotations: {len(dataset)}")

# Get all images with annotations
images_with_annotations = [sample for sample in dataset if len(sample['detections']) > 0]

# Specify the number of samples you want (e.g, 10 random samples)
num_samples = 10

# Randomly sample 'num_samples' images with annotations
sampled_images = random.sample(images_with_annotations, num_samples)

# Debug: Check how many sampled images are selected
print(f"Total sampled images: {len(sampled_images)}")

# Create a new dataset with the sampled images
sampled_dataset = fo.Dataset(name="sampled_dataset")

# Add sampled images to the new dataset
for image in sampled_images:
    sampled_dataset.add_sample(image)

# Launch the app with the new sampled dataset
session = fo.launch_app(sampled_dataset, port=5010)

# Blocks execution until the App is closed
session.wait()

print(f"Displayed {len(sampled_images)} sampled images in the FiftyOne app.")

"""
# Load COCO formatted dataset
coco_dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=data_path,
    labels_path=json_path,
    include_id=True,
)

# Launch the app
session = fo.launch_app(coco_dataset, port=5151)

# Blocks execution until the App is closed
session.wait()

import json

# Load and validate JSON
with open("COCO_Data/images/annotations.json") as f:
    data = json.load(f)

print("Number of images:", len(data["images"]))
print("Number of annotations:", len(data["annotations"]))
print("Categories:", [cat["name"] for cat in data["categories"]])
