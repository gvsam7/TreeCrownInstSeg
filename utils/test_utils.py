from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import os
import torch
import traceback


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
    # evaluator = COCOEvaluator(test_dataset, cfg, False, output_dir=output_dir)
    test_loader = build_detection_test_loader(cfg, test_dataset)
    print(f"Test dataset size: {len(test_loader)}")

    # Validate prediction on a single sample (before full evaluation)
    try:
        # Take first batch and first image
        sample_batch = next(iter(test_loader))
        sample_image = sample_batch[0]['image']

        # Ensure image is in correct format for predictor
        # Typically, predictors expect numpy arrays or torch tensors in specific formats
        if isinstance(sample_image, torch.Tensor):
            # Convert to numpy if needed
            sample_image = sample_image.permute(1, 2, 0).numpy()

        # Test prediction on single image
        sample_prediction = predictor(sample_image)
        print(f"Sample prediction type: {type(sample_prediction)}")
        print(f"Sample prediction content: {sample_prediction}")
    except Exception as e:
        print(f"Error in sample prediction: {e}")
        traceback.print_exc()

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the COCO evaluator
    evaluator = COCOEvaluator(test_dataset, cfg, False, output_dir=output_dir)

    # Run Evaluation
    try:
        evaluation_results = inference_on_dataset(predictor.model, test_loader, evaluator)
        print(f"Evaluation complete. Results: {evaluation_results}")
        return evaluation_results
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None
