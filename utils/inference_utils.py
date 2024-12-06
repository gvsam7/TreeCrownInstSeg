import os
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
import csv
import yaml
from skimage.measure import regionprops, label
from detectron2.utils.visualizer import ColorMode


def initialise_predictor(config_file, threshold):
    # Initialise the Detectron2 predictor with the given config file.
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.DEVICE = "cpu"
    # Set the threshold for inference
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # Apply threshold for inference
    cfg.MODEL.MASK_ON = True

    predictor = DefaultPredictor(cfg)
    return cfg, predictor


def run_inference(image_dir, output_dir, predictor, metadata):
    # Perform inference and save visualised results.
    os.makedirs(output_dir, exist_ok=True)
    for image_filename in os.listdir(image_dir):
        if not image_filename.lower().endswith(('.PNG', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue  # Skip non-image files

        image_path = os.path.join(image_dir, image_filename)
        new_im = cv2.imread(image_path)
        outputs = predictor(new_im)

        # Debugging to ensure metadata is correct
        print(f"Metadata type: {type(metadata)}")  # Debug line

        # Visualise results
        v = Visualizer(new_im[:, :, ::-1], metadata=metadata, scale=1.0, instance_mode=ColorMode.IMAGE_BW)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        output_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_result.png")
        cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

    print("Inference completed and visualised results saved.")


def export_results_to_csv(image_dir, output_csv_path, predictor, metadata):
    # Extract object-level information and save to a CSV file.
    with open(output_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header row in the CSV file
        csvwriter.writerow(["File Name", "Category_ID", "Tree_ID", "Area", "Segmentation", "BoundingBox", "Score"])

        for image_filename in os.listdir(image_dir):
            if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue

            print(f"Processing image: {image_filename}")
            image_path = os.path.join(image_dir, image_filename)
            new_img = cv2.imread(image_path)

            # Detailed debugging of the model's configuration
            try:
                outputs = predictor(new_img)
                instances = outputs["instances"]

                # Comprehensive output logging
                print(f"Instances for {image_filename}:")
                print(f"  - Number of instances: {len(instances)}")
                print(f"  - Available fields: {instances.get_fields()}")

                # Check specific fields
                print("Checking specific fields:")
                print(f"  - Has pred_masks: {instances.has('pred_masks')}")
                print(f"  - Has pred_boxes: {instances.has('pred_boxes')}")
                print(f"  - Has scores: {instances.has('scores')}")

                if len(instances) == 0:
                    print(f"No objects detected in {image_filename}.")
                    continue

                # Retrieve all relevant information
                if instances.has('pred_masks'):
                    masks = instances.pred_masks.to("cpu").numpy().astype(bool)
                else:
                    masks = None

                pred_boxes = instances.pred_boxes.tensor.to("cpu").numpy()
                pred_classes = instances.pred_classes.to("cpu").numpy()
                scores = instances.scores.to("cpu").numpy()

                for instance_idx, (box, class_label, score) in enumerate(zip(pred_boxes, pred_classes, scores)):
                    # Extract bounding box coordinates
                    y1, x1, y2, x2 = box

                    # Calculate area of bounding box
                    area = (y2 - y1) * (x2 - x1)

                    # Get class name
                    class_name = metadata.thing_classes[class_label]

                    # Bounding box and centroid strings
                    bounding_box_str = f"({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})"
                    centroid_str = f"({(x1 + x2) / 2:.2f}, {(y1 + y2) / 2:.2f})"

                    # Handle mask information if available
                    mask_area = None
                    if masks is not None and instance_idx < len(masks):
                        instance_mask = masks[instance_idx]
                        mask_area = instance_mask.sum()

                    csvwriter.writerow(
                        [image_filename, class_name, instance_idx + 1,
                         area, centroid_str, bounding_box_str, score]
                    )

                    # Additional detailed logging
                    print(f"Detected Object {instance_idx + 1} in {image_filename}:")
                    print(f"  - Class Name: {class_name}")
                    print(f"  - Score: {score}")
                    print(f"  - Bounding Box Area: {area}")
                    if mask_area is not None:
                        print(f"  - Mask Area: {mask_area}")
                    print(f"  - Centroid: {centroid_str}")
                    print(f"  - Bounding Box: {bounding_box_str}")

            except Exception as e:
                print(f"Error processing {image_filename}: {e}")
                # Optionally, print the full traceback
                import traceback
                traceback.print_exc()

        print("Results exported to CSV.")

"""
def export_results_to_csv(image_dir, output_csv_path, predictor, metadata):
    # Extract object-level information and save to a CSV file.
    with open(output_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header row in the CSV file
        csvwriter.writerow(["File Name", "Category_ID", "Tree_ID", "Area", "Segmentation", "BoundingBox"])

        for image_filename in os.listdir(image_dir):
            if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue

            print(f"Processing image: {image_filename}")
            image_path = os.path.join(image_dir, image_filename)
            new_img = cv2.imread(image_path)

            outputs = predictor(new_img)
            print(f"Outputs for {image_filename}: {outputs}")  # Debug outputs

            if len(outputs["instances"]) == 0:
                print(f"No objects detected in {image_filename}.")
                continue

            masks = outputs["instances"].pred_masks.to("cpu").numpy().astype(bool)
            class_labels = outputs["instances"].pred_classes.to("cpu").numpy()

            for instance_idx, instance_mask in enumerate(masks):
                labeled_mask = label(instance_mask)
                props = regionprops(labeled_mask)

                for prop in props:
                    area = prop.area
                    centroid_str = f"({prop.centroid[0]:.2f}, {prop.centroid[1]:.2f})"
                    bounding_box_str = f"({prop.bbox[0]}, {prop.bbox[1]}, {prop.bbox[2]}, {prop.bbox[3]})"

                    if instance_idx < len(class_labels):
                        class_label = class_labels[instance_idx]
                        class_name = metadata.thing_classes[class_label]
                    else:
                        class_name = "Unknown"

                    print(f"Detected Object {instance_idx + 1} in {image_filename}:")
                    print(f"  - Class Name: {class_name}")
                    print(f"  - Area: {area}")
                    print(f"  - Centroid: {centroid_str}")
                    print(f"  - Bounding Box: {bounding_box_str}")

                    csvwriter.writerow(
                        [image_filename, class_name, instance_idx + 1, area, centroid_str, bounding_box_str])

    print("Results exported to CSV.")"""