from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os


def train_model(config_file):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()