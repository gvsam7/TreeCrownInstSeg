import os
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
import csv
import yaml


def initialise_predictor(config_file):
    """Initialise the Detectron2 predictor with the given config file."""
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.DEVICE = "cpu"

    # Save the config to a YAML file
    config_yaml_path = "Detectron2/config.yaml"
    print(f"path: ", config_yaml_path)
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

        # Visualise results
        v = Visualizer(new_im[:, :, ::-1], metadata=metadata)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        output_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_result.png")
        cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

    print("Inference completed and visualized results saved.")


def export_results_to_csv(image_dir, output_csv_path, predictor):
    """Extract object-level information and save to a CSV file."""
    with open(output_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["File Name", "Category_ID", "Tree_ID", "Area", "Segmentation", "BoundingBox"])

        for image_filename in os.listdir(image_dir):
            if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue  # Skip non-image files

            image_path = os.path.join(image_dir, image_filename)
            new_img = cv2.imread(image_path)
            outputs = predictor(new_img)
            instances = outputs["instances"].to("cpu")

            for i in range(len(instances)):
                category_id = instances.pred_classes[i].item()
                bbox = instances.pred_boxes[i].tensor.numpy().tolist()
                area = instances.sizes[i].item()

                # Placeholder for Tree_ID extraction logic
                tree_id = "Unknown"

                csvwriter.writerow([image_filename, category_id, tree_id, area, bbox])

    print("Results exported to CSV.")
