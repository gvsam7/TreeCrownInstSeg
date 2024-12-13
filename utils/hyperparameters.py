import argparse


# Hyperparameters
def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--ims_per_batch", type=int, default=2)
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument("--checkpoint_period", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--base_lr", type=float, default=25e-5)
    parser.add_argument("--threshold", type=float, default=0.1)

    return parser.parse_args()

