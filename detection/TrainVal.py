from utils import create_cross_validation_splits
from ultralytics import YOLO
import yaml
import argparse

from pathlib import Path

class TrainerYolo():
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
        results = self.model.train(
            mode=self.mode,
            data=self.data,
            imgsz=self.imgsz,
            epochs=self.epochs,
            batch=self.batch,
            name=self.output,
            resume=self.resume,
        )

    def validate(self):
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
            
            for ds_yaml in ds_yamls:
                cfg['data']= ds_yaml
                trainer = TrainerYolo(cfg)
                trainer.train()
        else:
            trainer.train()
            pass

    elif args.mode == "validate":
        trainer.validate()