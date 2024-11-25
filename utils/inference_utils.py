import os
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
# from skimage.measure import label, regionprops
from detectron2.config import get_cfg 
import csv


def run_inference(image_dir, config_file, output_dir):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)

    os.makedirs(output_dir, exist_ok=True)
    for image_filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_filename)
        new_im = cv2.imread(image_path)
        outputs = predictor(new_im)
        v = Visualizer(new_im[:, :, ::-1], metadata=cfg.DATASETS.TRAIN[0])
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        output_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_result.png")
        cv2.imwrite(output_path, out.get_image()[:, :, ::-1])


def export_results(image_dir, output_dir, output_csv_path):
    with open(output_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["File Name", "Class Name", "Object Number", "Area", "Centroid", "BoundingBox"])
        # Implement object-level information extraction
