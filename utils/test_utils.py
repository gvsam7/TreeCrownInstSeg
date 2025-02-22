from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import os
import numpy as np
from detectron2.utils.visualizer import GenericMask
from pycocotools import mask as mask_utils
from detectron2.structures import Instances
from detectron2.engine import DefaultPredictor
import numpy as np
import cv2


class COCOEvaluatorModified:
    def __init__(self, test_dataset):
        self.test_dataset = test_dataset
        self.filtered_predictions = []  # stored filtered predictions
        print(f"Evaluating the model on the dataset: {test_dataset}")

    def reset(self):
        # Reset stored predictions at the beginning of evaluation
        self.filtered_predictions = []

    def overlaps_with_annotations(self, pred_bbox, gt_bboxes):
        # Check if predicted bbox overlaps with any ground truth bbox
        for gt_bbox in gt_bboxes:
            # Intersection-over-Union (IoU) for bbox overlap
            ixmin = max(pred_bbox[0], gt_bbox[0])
            iymin = max(pred_bbox[1], gt_bbox[1])
            ixmax = min(pred_bbox[2], gt_bbox[2])
            iymax = min(pred_bbox[3], gt_bbox[3])

            iw = max(0, ixmax - ixmin)
            ih = max(0, iymax - iymin)
            intersection = iw * ih

            area_pred = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
            area_gt = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])

            union = area_pred + area_gt - intersection
            iou = intersection / union if union != 0 else 0

            # Assuming a threshold for overlap, for instance, 0.5
            if iou >= 0.5:
                return True
        return False

    def process(self, inputs, outputs):
        """Filter predictions that do not overlap with ground truth."""
        for input, output in zip(inputs, outputs):
            gt_bboxes = [ann["bbox"] for ann in input.get("annotations", [])]
            valid_predictions = []

            if "instances" in output:
                pred_boxes = output["instances"].pred_boxes.tensor.cpu().numpy()
                for i, pred_bbox in enumerate(pred_boxes):
                    if self.overlaps_with_annotations(pred_bbox, gt_bboxes):
                        valid_predictions.append(output["instances"][i])

                output["instances"] = output["instances"][valid_predictions]

            self.filtered_predictions.append(output)

    def evaluate(self):
        """Return filtered predictions for evaluation."""
        return self.filtered_predictions

    def evaluate_predictions(self, predictions):
        filtered_predictions = []

        for prediction in predictions:
            pred_bbox = prediction["bbox"]  # Assuming bbox is a [xmin, ymin, xmax, ymax] list
            # Ground truth bboxes in coco_data
            gt_bboxes = [entry["bbox"] for entry in self.coco_data["annotations"]]

            # Check overlap with ground truth
            if self.overlaps_with_annotations(pred_bbox, gt_bboxes):
                filtered_predictions.append(prediction)

        return filtered_predictions


class FixedCOCOEvaluator(COCOEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._predictions = []  # Initialise _predictions to avoid AttributeError


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


def filtered_evaluate_model2(cfg, predictor, test_dataset, output_dir, ground_truth_annotations, iou_threshold):
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

    print(f"Evaluating the model on the filtered dataset: {test_dataset}")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialise the COCO evaluator (use the modified version)
    evaluator = COCOEvaluatorModified(test_dataset)  # Using my modified evaluator class
    test_loader = build_detection_test_loader(cfg, test_dataset)

    # Run Evaluation - I might need to modify this part to accommodate the new evaluator
    # Since my evaluator has custom logic, it might need adjustments in how predictions are handled
    filtered_evaluation_results = inference_on_dataset(predictor.model, test_loader, evaluator)

    print(f"Evaluation complete. Results: {filtered_evaluation_results}")
    return filtered_evaluation_results


def filtered_evaluate_model(cfg, predictor, test_dataset, output_dir, ground_truth_annotations, iou_threshold):
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
    # evaluator = COCOEvaluator(test_dataset, cfg, False, output_dir=output_dir)
    evaluator = FixedCOCOEvaluator(test_dataset, cfg, False, output_dir=output_dir)
    test_loader = build_detection_test_loader(cfg, test_dataset)

    # Run inference on test data and filter predictions
    print("Running inference and OLD filtering predictions...")
    # predictions = []
    for inputs in test_loader:
        """outputs = predictor(inputs)  # Model predictions
        filtered_preds = filter_predictions(outputs, ground_truth_annotations, iou_threshold)
        predictions.append(filtered_preds)

    # Evaluate filtered predictions
    filtered_evaluation_results = inference_on_dataset(predictor.model, predictions, evaluator)"""

        # The following part is a more computationally efficient way in accordance to COCO standards
        # Process single image
        input_data = inputs[0]  # Extract the first image dictionary

        # Get the image ID for filtering ground truth annotations
        img_id = input_data["image_id"]
        img_gt_anns = [ann for ann in ground_truth_annotations if ann["image_id"] == img_id]

        # Load the original image using its file path
        original_image = cv2.imread(input_data["file_name"])
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Pass the full input_data dictionary to the predictor
        outputs = predictor(original_image)

        # Filter predictions based on ground truth
        # outputs["instances"] = filter_predictions(outputs["instances"], img_gt_anns, iou_threshold)
        # Ensure outputs contain predictions before filtering
        if "instances" in outputs and len(outputs["instances"]) > 0:
            outputs["instances"] = filter_predictions(outputs["instances"], img_gt_anns, iou_threshold)
        else:
            print("Warning: No predictions found in outputs!")

        # Process filtered predictions
        evaluator.process([input_data], [outputs])

    filtered_evaluation_results = evaluator.evaluate()

    # This part was left as it was...
    print(f"Filtered evaluation complete. Results: {filtered_evaluation_results}")
    return filtered_evaluation_results


def binary_mask_to_rle(mask):
    """
    Convert a binary mask to RLE format compatible with pycocotools.

    Args:
        mask (np.ndarray): Binary mask (height x width).

    Returns:
        dict: RLE-encoded mask.
    """
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle


# Retain only the predictions that overlap sufficiently (based on the IoU threshold) with at least 1 labelled tree crown.
def filter_predictions(instances, ground_truth_annotations, iou_threshold):
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
        print("No predictions available to filter.")
        return instances

    print(f"Number of predictions: {len(instances)}")

    pred_masks = instances.pred_masks.cpu().numpy()
    valid_indices = []

    # Extract image dimensions from the input
    image_height = instances.image_size[0]
    image_width = instances.image_size[1]

    # Convert ground truth to RLE masks
    gt_rle_masks = [
        binary_mask_to_rle(GenericMask(gt["segmentation"], image_height, image_width).mask)
        for gt in ground_truth_annotations
    ]

    # Create an `iscrowd` list with the same length as `gt_rle_masks`
    iscrowd = [0] * len(gt_rle_masks)  # All ground truth annotations are not "crowd"

    # Check each prediction against ground truth
    for idx, pred_mask in enumerate(pred_masks):
        # Convert prediction mask to RLE
        pred_rle = binary_mask_to_rle(pred_mask)
        print(f"Prediction {idx}: Mask shape {pred_mask.shape}")  # Debugging

        # Compute IoUs between this prediction and all ground truth masks
        ious = mask_utils.iou([pred_rle], gt_rle_masks, iscrowd)[0]
        print(f"Prediction {idx}: IoUs with ground truth = {ious}")  # Debugging

        # Check if the IoU with any ground truth exceeds the threshold
        if any(iou >= iou_threshold for iou in ious):
            print(f"Prediction {idx} passes IoU threshold with value: {max(ious)}")  # Debugging
            valid_indices.append(idx)
        else:
            print(f"Prediction {idx} does not meet IoU threshold.")  # Debugging

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