import os
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
import csv
import yaml
from skimage.measure import regionprops, label


def initialise_predictor(config_file):
    """Initialise the Detectron2 predictor with the given config file."""
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.DEVICE = "cpu"

    # Save the config to a YAML file
    config_yaml_path = "Detectron2/config.yaml"
    with open(config_yaml_path, 'w') as file:
        yaml.dump(cfg, file)

    predictor = DefaultPredictor(cfg)
    return cfg, predictor


def run_inference(image_dir, output_dir, predictor, metadata):
    """Perform inference and save visualized results."""
    os.makedirs(output_dir, exist_ok=True)
    for image_filename in os.listdir(image_dir):
        if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue  # Skip non-image files

        image_path = os.path.join(image_dir, image_filename)
        new_im = cv2.imread(image_path)
        outputs = predictor(new_im)

        # Debugging to ensure metadata is correct
        print(f"Metadata type: {type(metadata)}")  # Debug line

        # Visualise results
        v = Visualizer(new_im[:, :, ::-1], metadata=metadata)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        output_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_result.png")
        cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

    print("Inference completed and visualized results saved.")


def export_results_to_csv(image_dir, output_csv_path, predictor, metadata):
    """Extract object-level information and save to a CSV file."""
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

    print("Results exported to CSV.")