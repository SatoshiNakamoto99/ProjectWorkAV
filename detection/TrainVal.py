from utils import create_cross_validation_splits
from ultralytics import YOLO
import yaml
import argparse
import torch

from pathlib import Path

class TrainerYolo():
    """
    Class for training and validating a YOLO model.

    Args:
        cfg (dict): Configuration parameters for the trainer.

    Attributes:
        cfg (dict): Configuration parameters for the trainer.
        data (str): Path to the data directory.
        imgsz (int): Input image size.
        batch (int): Batch size.
        epochs (int): Number of training epochs.
        output (str): Path to the output directory.
        mode (str): Training mode ('train', 'test', or 'val').
        resume (bool): Whether to resume training from a checkpoint.
        model (YOLO): Pretrained YOLO model.

    Methods:
        train(): Trains the YOLO model.
        validate(): Validates the YOLO model.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.data = cfg['data']
        self.imgsz = cfg['img_size']
        self.batch = cfg['batch_size']
        self.epochs = cfg['epochs']
        self.output = cfg['output']
        self.mode = cfg['mode']
        self.resume = cfg['resume']

        # loading a pretrained YOLO model
        self.model = YOLO(cfg['model'])

    def train(self):
        """
        Trains the YOLO model.

        Returns:
            dict: Training results.
        """
        results = self.model.train(
            mode=self.mode,
            data=self.data,
            imgsz=self.imgsz,
            epochs=self.epochs,
            batch=self.batch,
            name=self.output,
            resume=self.resume,
        )
        torch.cuda.empty_cache()

    def validate(self):
        """
        Validates the YOLO model.

        Returns:
            dict: Validation results.
        """
        results = self.model.val(
            data=self.data,
            imgsz=self.imgsz,
            name=self.output,
        )

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-cfg", "--config", type=str, default="./config/yolov8x.yaml", help="path to model config file")
    args.add_argument("-m", "--mode", type=str, default="train", help="train or validate")
    args = args.parse_args()

    with open(args.config) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    trainer = TrainerYolo(cfg)

    if args.mode == "train":
        if cfg['cross_val']:
            ds_yamls = create_cross_validation_splits(Path(cfg['dataset_path']), Path(cfg['data']), ksplit=cfg['cross_val']['ksplit'], random_state=cfg['cross_val']['random_state'])
            #results = {}
            #k = 1
            for ds_yaml in ds_yamls:
                cfg['data']= ds_yaml
                trainer = TrainerYolo(cfg)
                trainer.train()
              #  results[k] = trainer.metrics
               # k += 1
        else:
            trainer.train()
            

    elif args.mode == "validate":
        trainer.validate()