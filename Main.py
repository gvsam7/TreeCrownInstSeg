""""
Author: Georgios Voulgaris
Date: 22/11/2024
Description: This repository uses Detectron2 to train a Mask R-CNN for segmenting individual tree crowns. It serves as a
             testing platform that will enable for further experimentation on various models and explore data fusion
             techniques for the task of individual tree instance segmentation.
"""

import os
from utils.data_utils import register_datasets, visualise_samples
from utils.train_utils import train_model
from utils.test_utils import evaluate_model
from utils.inference_utils import initialise_predictor, run_inference, export_results_to_csv
from utils.hyperparameters import arguments
import torch
from detectron2.data import MetadataCatalog, DatasetCatalog #added for test
import pandas as pd
import wandb

import yaml
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
import random
from detectron2.utils.visualizer import Visualizer
import cv2


def main():
    args = arguments()

    # Initialise wandb with more detailed configuration
    wandb.init(
        project="InstSeg",
        config={
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "num_classes": args.num_classes,
            "num_workers": args.num_workers,
            "ims_per_batch": args.ims_per_batch,
            "max_iter": args.max_iter,
            "batch_size": args.batch_size,
            "base_lr": args.base_lr,
            "threshold": args.threshold
        }
    )

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on the: {device}")

    # Step 1: Setup environment and datasets
    register_datasets()
    # visualise_samples()

    # Step 2: Train the model
    config_file = "detectron2/config.yaml"
    train_model(
        device=device,
        output_dir="outputs/results",
        num_classes=args.num_classes,
        train_dataset="my_dataset_train",
        test_dataset="my_dataset_test",
        num_workers=args.num_workers,
        ims_per_batch=args.ims_per_batch,
        max_iter=args.max_iter,  # 1000,
        batch_size=args.batch_size,
        base_lr=args.base_lr,
    )

    #  Testing colab style
    cfg = get_cfg()
    # Save the config to a YAML file
    config_yaml_path = "detectron2/config.yaml"
    with open(config_yaml_path, 'w') as file:
        yaml.dump(cfg, file)

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
    # predictor = DefaultPredictor(cfg)

    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = os.path.join("outputs/results", "model_final.pth")
    cfg.MODEL.DEVICE = "cpu"
    # Set the threshold for inference
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Apply threshold for inference
    cfg.MODEL.MASK_ON = True
    predictor = DefaultPredictor(cfg)

    from detectron2.data.datasets import register_coco_instances
    register_coco_instances("my_dataset_test", {}, "Data/test/annotations.json", "Data/test")

    test_metadata = MetadataCatalog.get("my_dataset_test")
    test_dataset_dicts = DatasetCatalog.get("my_dataset_test")

    for d in random.sample(test_dataset_dicts, 6):  # select number of images for display
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=test_metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2_imshow(out.get_image()[:, :, ::-1])
        cv2.imwrite("Data/test", out.get_image()[:, :, ::-1])

    """
    # Step 3: Initialise the predictor
    cfg, predictor = initialise_predictor(config_file, threshold=args.threshold)

    # Step 4: Evaluate the model
    test_dataset = "my_dataset_test"
    output_dir = "outputs/results"
    # Evaluate model and get results
    evaluation_results = evaluate_model(cfg, predictor, test_dataset, output_dir)
    # Log evaluation results
    wandb.log({"evaluation_results": evaluation_results})

    # Step 5: Run inference on the test set
    test_images_dir = "Data/test"
    output_dir = "outputs/results"
    # metadata = cfg.DATASETS.TRAIN[0]
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # Debugging to ensure metadata is correct
    print(f"Main Metadata type: {type(metadata)}")
    run_inference(test_images_dir, output_dir, predictor, metadata)"""

    # Step 6: Export results to a CSV
    ########## Test ##################################
    test_images_dir = "Data/test"
    output_dir = "outputs/results"
    # metadata = cfg.DATASETS.TRAIN[0]
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    ######################################################
    output_csv_path = os.path.join(output_dir, "output_objects.csv")
    export_results_to_csv(test_images_dir, output_csv_path, predictor, metadata)

    # Wandb logging
    # Log the CSV file
    wandb.save(output_csv_path)

    # inference_results_dir = os.path.join(output_dir, "inference_results")
    # Create the output directory if it doesn't exist
    # os.makedirs(inference_results_dir, exist_ok=True)
    """inference_images = []
    for img_file in os.listdir(output_dir):
        if img_file.endswith(('_result.png', '.jpg', '.jpeg')):
            inference_images.append(wandb.Image(
                os.path.join(output_dir, img_file),
                caption=f"Inference Result: {img_file}"
            ))

    # Log images to wandb
    wandb.log({"inference_results": inference_images})"""

    results_dir = "outputs/results"
    inference_results = {}

    for img_file in os.listdir(results_dir):
        if img_file.endswith(('_result.png', '.jpg', '.jpeg', '.png')):
            image_path = os.path.join(results_dir, img_file)
            inference_results[img_file] = wandb.Image(image_path)

    wandb.log({"inference_results": inference_results})

    # Optional: Log some summary statistics from the CSV
    csv_df = pd.read_csv(output_csv_path)
    wandb.log({
        "total_detected_objects": len(csv_df),
        "unique_classes": csv_df['Category_ID'].nunique(),
        "avg_object_area": csv_df['Area'].mean(),
        "max_object_area": csv_df['Area'].max(),
        "min_object_area": csv_df['Area'].min()
    })


if __name__ == "__main__":
    main()
