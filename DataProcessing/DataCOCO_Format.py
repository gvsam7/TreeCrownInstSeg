"""
Author: Georgios Voulgaris
Date: 05/11/2024
Version: 1
Description: This script reads a GeoJSON file containing a collection of tree crown polygons, each representing an
             individual tree, along with an orthomosaic image in TIFF format. It slices the orthomosaic into 768x768
             pixel images and converts the data into COCO format, making it compatible with Mask R-CNN for instance
             segmentation tasks.
"""

import geopandas as gpd
import rasterio
import numpy as np
from rasterio.windows import Window
import os
import json
from PIL import Image
from shapely.geometry import shape, box, MultiPolygon, Polygon
from rasterio.warp import transform_geom

# Define paths
dataset_json_path = 'TestCOCO.json'
orthomosaic_path = 'TestCOCO.tif'
output_dir = 'COCO_Data/images/'
coco_json_path = os.path.join(output_dir, 'annotations.json')

# Load the dataset from JSON
with open(dataset_json_path, 'r') as f:
    dataset = json.load(f)

# Initialise COCO format data structure
coco_data = {
    "info": {
        "description": "Wytham Tree Crown Dataser",
        "version": "1.0",
        "year": 2024,
        "contributor": "Georgios Voulgaris",
        "date created": "11-11-2024"
    },
    "licences": [],
    "images": [],
    "annotations": [],
    "categories": [],
}

# Define categories (update this with your actual species and IDs)
categories = {
    'sycamore': 1, 'beech': 2, 'ash': 3, 'oak': 4, 'hawthorn': 5,
    'birch': 6, 'unknown': 7, 'field maple': 8, 'sweet chestnut': 9,
    'lime spp.': 10, 'hazel': 11,
}

for name, category_id in categories.items():
    coco_data['categories'].append({
        "id": category_id,
        "name": name,
        "supercategory": "tree"
    })

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Create output directory for non-labeled images
non_labeled_dir = os.path.join(output_dir, 'non_labeled')
os.makedirs(non_labeled_dir, exist_ok=True)

# Set tile size and pixel size
tile_size = 768
pixel_size = 0.01219  # 0.01219 meters per pixel

# Read the orthomosaic and process it
with rasterio.open(orthomosaic_path) as src:
    height, width = src.height, src.width
    print(f"src height: {height}")
    print(f"src widht: {width}")
    transform = src.transform

    # Slice the orthomosaic and create COCO annotations
    image_id = 0
    annotation_id = 0

    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            window = Window(col, row, tile_size, tile_size)
            print(f"Window: {window}")
            window_transform = src.window_transform(window)
            print(f"Window_transform: {window_transform}")

            # Convert pixel window bounds to real-world coordinates
            window_bounds_real_world = box(
                *rasterio.transform.array_bounds(tile_size, tile_size, window_transform)
            )

            # Read the tile
            tile = src.read(window=window)
            tile_image = tile.transpose(1, 2, 0)  # Change to HWC format

            # Check if the tile contains any annotations
            has_annotations = False

            # Check for geometries that intersect with the real-world window bounds
            for feature in dataset["features"]:
                geometry = shape(feature['geometry'])  # Convert geometry to Shapely shape

                # Check for intersection in real-world coordinates
                if geometry.intersects(window_bounds_real_world):
                    has_annotations = True
                    break

            if has_annotations:
                # Save the tile
                tile_image_pil = Image.fromarray(tile_image)
                # image_filename = f'image_{image_id}.png'
                image_filename = f'image_{row}_{col}.png'
                tile_image_pil.save(os.path.join(output_dir, image_filename))

                # Add image info to COCO
                coco_data['images'].append({
                    "id": image_id,
                    "file_name": image_filename,
                    "width": tile_size,
                    "height": tile_size,
                    "pixel_size": pixel_size
                })

                # Create COCO annotations for the tile
                for feature in dataset["features"]:
                    geometry = shape(feature['geometry'])
                    if geometry.intersects(window_bounds_real_world):
                        intersection = geometry.intersection(window_bounds_real_world)

                        # Handle case for Polygon and MultiPolygon
                        if isinstance(intersection, Polygon):
                            # Single Polygon
                            coords = list(intersection.exterior.coords)
                        elif isinstance(intersection, MultiPolygon):
                            # Multiple Polygons, iterate through each one
                            coords = []
                            for poly in intersection:
                                coords.extend(list(poly.exterior.coords))

                        # Ensure that each coordinate has only two values (x, y), and handle cases with extra values
                        coords_2d = [(x, y) for (x, y, *rest) in coords]  # Only unpack the first two values

                        # Tile offsets in real-world coordinates
                        tile_x_offset_real = window_transform.c  # Top-left x-coordinate of the current tile
                        tile_y_offset_real = window_transform.f  # Top-left y-coordinate of the current tile

                        # Adjust segmentation coordinates to be relative to the tile and convert to pixel coordinates
                        coords_pixel = [
                            (
                                (x - tile_x_offset_real) / pixel_size,
                                (tile_y_offset_real - y) / pixel_size
                            )
                            for x, y in coords_2d
                        ]

                        # Convert the real-world coordinates to pixel coordinates
                        # coords_pixel = [(x / pixel_size, y / pixel_size) for x, y in coords_2d]

                        # Update the COCO data structure with pixel-based segmentation
                        coco_data['annotations'].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": categories[feature['properties']['Species']],
                            "Tree_ID": feature['properties']['Tree_ID'],
                            "segmentation": [np.array(coords_pixel).flatten().tolist()],
                            "area": intersection.area / (pixel_size ** 2),  # Correct area calculation in pixel space
                            "bbox": [
                                round((intersection.bounds[0] - tile_x_offset_real) / pixel_size),
                                round((tile_y_offset_real - intersection.bounds[3]) / pixel_size),  # Upper left y in pixels
                                round((intersection.bounds[2] - intersection.bounds[0]) / pixel_size),  # Width in pixels
                                round((intersection.bounds[3] - intersection.bounds[1]) / pixel_size)  # Height in pixels
                            ],
                            "iscrowd": 0,
                        })
                        annotation_id += 1

            else:
                # Save the non-labeled tile to the non-labeled directory only if it's 768x768
                if tile_image.shape[0] == tile_size and tile_image.shape[1] == tile_size:
                    tile_image_pil = Image.fromarray(tile_image)
                    non_labeled_filename = f'non_labeled_image_{row}_{col}.png'
                    tile_image_pil.save(os.path.join(non_labeled_dir, non_labeled_filename))

            image_id += 1  # Increment image ID only if an image was saved

# Save COCO annotations
with open(coco_json_path, 'w') as f:
    json.dump(coco_data, f, indent=4)

print(f"Processed {image_id} images and created annotations at {coco_json_path}.")



"""
# This is a working code that saves all sliced 768x768 images, did not check the COCO
import geopandas as gpd
import rasterio
import numpy as np
from rasterio.windows import Window
import os
import json
from PIL import Image
from shapely.geometry import shape, mapping, box

# Define paths
dataset_json_path = 'TestCOCO.json'  # Your input dataset in JSON format
orthomosaic_path = 'Orthomosaic.rgb.tif'
output_dir = 'COCO_Data/images/'
coco_json_path = os.path.join(output_dir, 'annotations.json')

# Load the dataset from JSON
with open(dataset_json_path, 'r') as f:
    dataset = json.load(f)

# Initialize COCO format data structure
coco_data = {
    "images": [],
    "annotations": [],
    "categories": [],
}

# Define categories (update this with your actual species and IDs)
categories = {
    'sycamore': 1, 'beech': 2, 'ash': 3, 'oak': 4, 'hawthorn': 5,
    'birch': 6, 'unknown': 7, 'field maple': 8, 'sweet chestnut': 9,
    'lime spp.': 10, 'hazel': 11,
}

for name, category_id in categories.items():
    coco_data['categories'].append({
        "id": category_id,
        "name": name,
        "supercategory": "tree"
    })

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Set tile size
tile_size = 768

# Read the orthomosaic and process it
with rasterio.open(orthomosaic_path) as src:
    height, width = src.height, src.width

    # Slice the orthomosaic and create COCO annotations
    image_id = 0
    annotation_id = 0

    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            window = Window(col, row, tile_size, tile_size)
            transform = src.window_transform(window)

            # Read the tile
            tile = src.read(window=window)
            tile_image = tile.transpose(1, 2, 0)  # Change to HWC format

            # Check if the tile contains non-zero values (not blank)
            if np.any(tile_image):  # Only process if the tile is not blank
                tile_image_pil = Image.fromarray(tile_image)

                # Save the tile
                image_filename = f'image_{image_id}.png'
                tile_image_pil.save(os.path.join(output_dir, image_filename))

                # Add image info to COCO
                coco_data['images'].append({
                    "id": image_id,
                    "file_name": image_filename,
                    "width": tile_size,
                    "height": tile_size,
                })

                # Check for geometries that intersect with the tile
                for feature in dataset["features"]:
                    geometry = shape(feature['geometry'])  # Convert geometry to Shapely shape
                    species = feature['properties']['Species']
                    tree_id = feature['properties']['Tree_ID']

                    # Get the bounds of the current window in the raster space
                    window_bounds = box(col, row, col + tile_size, row + tile_size)

                    # Check for intersection
                    if geometry.intersects(window_bounds):
                        # Create COCO annotation
                        intersection = geometry.intersection(window_bounds)

                        # Create COCO annotation
                        coco_data['annotations'].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": categories[species],  # Get category ID from species
                            "segmentation": [np.array(intersection.exterior.coords).flatten().tolist()] if intersection.geom_type == 'Polygon' else [],
                            "area": intersection.area,
                            "bbox": intersection.bounds,
                            "iscrowd": 0,
                        })

                        annotation_id += 1

                image_id += 1  # Increment image ID only if an image was saved

# Save COCO annotations
with open(coco_json_path, 'w') as f:
    json.dump(coco_data, f)

print(f"Processed {image_id} images and created annotations at {coco_json_path}.")"""
