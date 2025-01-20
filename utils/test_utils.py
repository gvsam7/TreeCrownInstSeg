from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import os
import numpy as np
from detectron2.utils.visualizer import GenericMask
from pycocotools import mask as mask_utils
from detectron2.structures import Instances
from detectron2.engine import DefaultPredictor


def evaluate_model(cfg, predictor, test_dataset, output_dir):
    """
    Evaluate the model using COCO metrics.

    Args:
        cfg: Detectron2 configuration object
        predictor: Detectron2 predictor object.
        test_dataset (str): Name of the registered test dataset.
        output_dir (str): Directory to save evaluation outputs

    Returns:
        dict: Evaluation results.
    """

    print(f"Evaluating the model on the dataset: {test_dataset}")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialise the COCO evaluator and test data loader
    evaluator = COCOEvaluator(test_dataset, cfg, False, output_dir=output_dir)
    test_loader = build_detection_test_loader(cfg, test_dataset)

    # Run Evaluation
    evaluation_results = inference_on_dataset(predictor.model, test_loader, evaluator)

    print(f"Evaluation complete. Results: {evaluation_results}")
    return evaluation_results


def filtered_evaluate_model(cfg, predictor, test_dataset, output_dir, ground_truth_annotations, iou_threshold=0.5):
    """
    Evaluate the model using COCO metrics after filtering predictions.

    Args:
        cfg: Detectron2 configuration object.
        predictor: Detectron2 predictor object.
        test_dataset (str): Name of the registered test dataset.
        output_dir (str): Directory to save evaluation outputs.
        ground_truth_annotations (list): List of ground truth annotations (COCO format).
        iou_threshold (float): Minimum IoU required to filter predictions.

    Returns:
        dict: Filtered evaluation results.
    """

    print(f"Evaluating the model on the dataset: {test_dataset}")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialise the COCO evaluator and test data loader
    evaluator = COCOEvaluator(test_dataset, cfg, False, output_dir=output_dir)
    test_loader = build_detection_test_loader(cfg, test_dataset)

    # Run inference on test data and filter predictions
    print("Running inference and filtering predictions...")
    # predictions = []
    for inputs in test_loader:
        """outputs = predictor(inputs)  # Model predictions
        filtered_preds = filter_predictions(outputs, ground_truth_annotations, iou_threshold)
        predictions.append(filtered_preds)

    # Evaluate filtered predictions
    filtered_evaluation_results = inference_on_dataset(predictor.model, predictions, evaluator)"""

    # The following part is a more computationally efficient way in accordance to COCO standards
    # Get image ID for matching ground truth
    img_id = inputs[0]["image_id"].item()

    # Get relevant ground truth annotations for this image
    img_gt_anns = [ann for ann in ground_truth_annotations if ann["image_id"] == img_id]

    # Get predictions
    outputs = predictor(inputs)

    # Filter predictions based on ground truth for this image
    outputs["instances"] = filter_predictions(outputs["instances"], img_gt_anns, iou_threshold)

    # Process filtered predictions
    evaluator.process(inputs, outputs)

    filtered_evaluation_results = evaluator.evaluate()

    # This part was left as it was...
    print(f"Filtered evaluation complete. Results: {filtered_evaluation_results}")
    return filtered_evaluation_results


# Retain only the predictions that overlap sufficiently (based on the IoU threshold) with at least 1 labelled tree crown.
def filter_predictions(instances, ground_truth_annotations, iou_threshold=0.5):
    """
    Filter predictions based on IoU with ground truth.

    Args:
        instances (Instances): Predicted instances object from Detectron2.
        ground_truth_annotations (list): List of ground truth annotations (COCO format).
        iou_threshold (float): Minimum IoU required to consider a prediction valid.

    Returns:
        Instances: Filtered instances object containing valid predictions.
    """
    if len(instances) == 0:
        return instances

    pred_masks = instances.pred_masks.cpu().numpy()
    valid_indices = []

    # Convert ground truth to masks
    gt_masks = [
        GenericMask(gt["segmentation"], gt["height"], gt["width"]).mask
        for gt in ground_truth_annotations
    ]

    # Check each prediction against ground truth
    for idx, pred_mask in enumerate(pred_masks):
        ious = [
            mask_utils.iou([pred_mask.astype(np.uint8)], [gt.astype(np.uint8)], [False])[0]
            for gt in gt_masks
        ]
        if any(iou >= iou_threshold for iou in ious):
            valid_indices.append(idx)

    # Create new filtered instances
    return Instances(
        instances.image_size,
        pred_masks=instances.pred_masks[valid_indices],
        pred_boxes=instances.pred_boxes[valid_indices],
        scores=instances.scores[valid_indices],
        pred_classes=instances.pred_classes[valid_indices]
    )


# Retain only the predictions that overlap sufficiently (based on the IoU threshold) with at least 1 labelled tree crown.
# def filter_predictions(predictions, ground_truth_annotations, iou_threshold=0.5):
    """
    Filters model predictions to include only those overlapping with ground truth annotations.

    Args:
        predictions (dict): Model's predicted outputs. Should contain "instances" with masks or boxes.
        ground_truth_annotations (list): List of ground truth annotations (COCO format or binary masks).
        iou_threshold (float): Minimum IoU required to consider a prediction valid.

    Returns:
        filtered_predictions (list): List of valid predictions after filtering.
    """
#     valid_predictions = []
#     pred_masks = predictions["instances"].pred_masks.cpu().numpy()  # Predicted masks
#     gt_masks = [GenericMask(gt["segmentation"], gt["height"], gt["width"]).mask for gt in ground_truth_annotations]

#     for pred_mask in pred_masks:
        # Calculate IoU with all ground truth masks
#         ious = [mask_utils.iou([pred_mask.astype(np.uint8)], [gt_mask.astype(np.uint8)], [False])[0] for gt_mask in
#                 gt_masks]

        # Check if any IoU exceeds the threshold
#         if max(ious) >= iou_threshold:
#             valid_predictions.append(pred_mask)

#     return valid_predictions