from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import os


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
