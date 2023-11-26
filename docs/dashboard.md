##  Training 🏋️‍♀️
###  Training a Model with Configuration File🚀

To train a model, you need to create a configuration file in the `config` folder. This file will contain all the necessary parameters for training. Here is an example of what this file might look like:

```yaml
# Verbose during prediction
verbose: False

img_size: 640
epochs: 200
batch_size: 16

# for training
mode: 'train' # 'train' or 'train resume'
data: "./datasets/CrowdHuman/yolo/dataset.yaml"
output: "yolov8n_crowdhuman"
model: weights/yolov8n.pt
resume: False
```

In this configuration file:

- `verbose`: Controls the verbosity of the training process.
- `img_size`: The size of the images for training.
- `epochs`: The number of epochs for training.
- `batch_size`: The size of the batches for training.
- `mode`: The mode of training, can be either 'train' or 'train resume'.
- `data`: The path to the dataset.
- `output`: The name of the output model.
- `model`: The path to the pretrained model.
- `resume`: Whether to resume training from a checkpoint.

The TrainerYolo class in the TrainVal.py script uses this configuration file to train or validate a model. It loads the configuration file, initializes a YOLO model with the specified parameters, and then either trains or validates the model based on the mode specified in the configuration file.

To use this script, you can run the following command:

```bash
 python ./detection/TrainVal.py -cfg ./config/yolov8n_mot_ch.yaml
```
### Cross Validation 🔄

Cross-validation is a potent preventive measure against overfitting. Here are the main advantages of cross-validation, each with its own set of benefits:

- **Model Validation 📊:** Cross-validation provides insights into how well the model performs on an independent dataset, helping estimate the accuracy of the model.

- **Bias-Variance Tradeoff ⚖️:** It aids in estimating the model's skill on new data. More folds during cross-validation can reduce bias-related errors, yielding a less biased model. However, this might introduce higher variability.

- **Hyperparameter Tuning 🛠️:** Cross-validation is highly useful in tuning a model's parameters. It helps find optimal parameters that result in the least validation error.

- **Model Selection 🏆:** It assists in selecting the model that best fits the data. Different models can be trained, and the one with the best performance on the validation set can be chosen.

- **Feature Selection 🤔:** Cross-validation helps identify which features contribute the most to predictions. This can be a valuable feature for dimensionality reduction.

The implementation of this strategy is done using a configuration file like this:

```yaml
# Verbose during prediction
verbose: False

img_size: 640
epochs: 50
batch_size: 16

# For training
# Cross-validation parameters
cross_val:
  ksplit: 5  # Change this value as needed
  random_state: 20  # Change this value as needed 

mode: 'train'
data: ".\\datasets\\all_data\\dataset.yaml"
dataset_path: ".\\datasets\\all_data\\"
output: "yolov8n_mot_ch"
model: weights/yolov8n.pt
resume: False

```
The TrainerYolo class in the TrainVal.py script uses this configuration file to check if the key `cross_val` is present. In this case, it initiates a cross-validation strategy.

To use this script, you can run the following command:

```bash
 python ./detection/TrainVal.py -cfg ./config/yolov8_Cross_Val_mot_ch.yaml
```

### Configuration Parameters📝

Note that this is the list of all parameters that you can set in a configuration file for training, which will consequently update the TrainVal class:

Key | Value | Description
--- | --- | ---
model | None | Path to the model file, e.g., `yolov8n.pt`, `yolov8n.yaml`
data | None | Path to the data file, e.g., `coco128.yaml`
epochs | 100 | Number of epochs to train for
patience | 50 | Epochs to wait for no observable improvement for early stopping of training
batch | 16 | Number of images per batch (-1 for AutoBatch)
imgsz | 640 | Size of input images as an integer
save | True | Save train checkpoints and predict results
save_period | -1 | Save checkpoint every x epochs (disabled if less than 1)
cache | False | `True/ram`, `disk`, or `False`. Use cache for data loading
device | None | Device to run on, e.g., `cuda device=0` or `device=0,1,2,3` or `device=cpu`
workers | 8 | Number of worker threads for data loading (per RANK if DDP)
project | None | Project name
name | None | Experiment name
exist_ok | False | Whether to overwrite an existing experiment
pretrained | True | `True` (bool) or a `str`. Whether to use a pretrained model (bool) or a model to load weights from (str)
optimizer | 'auto' | Optimizer to use, choices=[`SGD`, `Adam`, `Adamax`, `AdamW`, `NAdam`, `RAdam`, `RMSProp`, `auto`]
verbose | False | Whether to print verbose output
seed | 0 | Random seed for reproducibility
deterministic | True | Whether to enable deterministic mode
single_cls | False | Train multi-class data as single-class
rect | False | Rectangular training with each batch collated for minimum padding
cos_lr | False | Use cosine learning rate scheduler
close_mosaic | 10 | (int) Disable mosaic augmentation for final epochs (0 to disable)
resume | False | Resume training from the last checkpoint
amp | True | Automatic Mixed Precision (AMP) training, choices=[`True`, `False`]
fraction | 1.0 | Dataset fraction to train on (default is 1.0, all images in the train set)
profile | False | Profile ONNX and TensorRT speeds during training for loggers
freeze | None | (int or list, optional) Freeze the first n layers, or freeze a list of layer indices during training
lr0 | 0.01 | Initial learning rate (i.e., SGD=1E-2, Adam=1E-3)
lrf | 0.01 | Final learning rate (`lr0 * lrf`)
momentum | 0.937 | SGD momentum/Adam beta1
weight_decay | 0.0005 | Optimizer weight decay 5e-4
warmup_epochs | 3.0 | Warmup epochs (fractions are okay)
warmup_momentum | 0.8 | Warmup initial momentum
warmup_bias_lr | 0.1 | Warmup initial bias lr
box | 7.5 | Box loss gain
cls | 0.5 | Cls loss gain (scale with pixels)
dfl | 1.5 | DFL loss gain
pose | 12.0 | Pose loss gain (pose-only)
kobj | 2.0 | Keypoint obj loss gain (pose-only)
label_smoothing | 0.0 | Label smoothing (fraction)
nbs | 64 | Nominal batch size
overlap_mask | True | Masks should overlap during training (segment train only)
mask_ratio | 4 | Mask downsample ratio (segment train only)
dropout | 0.0 | Use dropout regularization (classify train only)
val | True | Validate/test during training
plots | False | Save plots and images during train/val

[Docs](https://docs.ultralytics.com/modes/train/)
## Validation ✅ 
###  Validation of a Model with Configuration File🎯

Validation is an essential part of training a model. It allows us to evaluate the model's performance on a separate dataset that was not used during training, which gives us a better understanding of how the model will perform on unseen data.

The `TrainerYolo` class in the `TrainVal.py` script also supports validation. The `validate` method of the `TrainerYolo` class performs validation on the model. It uses the same configuration file as the training phase, but the `mode` parameter in the configuration file should be set to 'validate'.

Here is how you can run the validation:

```bash
python ./detection/TrainVal.py -cfg ./config/yolov8n_mot_ch.yaml -m validate
```
### Configuration Parameters📝

Note that this is the list of all parameters that you can set in a configuration file for validation, which will consequently update the TrainVal class:

Key | Value | Description
--- | --- | ---
data | None | Path to the data file, e.g., `coco128.yaml`
imgsz | 640 | Size of input images as an integer
batch | 16 | Number of images per batch (-1 for AutoBatch)
save_json | False | Save results to JSON file
save_hybrid | False | Save hybrid version of labels (labels + additional predictions)
conf | 0.001 | Object confidence threshold for detection
iou | 0.6 | Intersection over union (IoU) threshold for NMS
max_det | 300 | Maximum number of detections per image
half | True | Use half precision (FP16)
device | None | Device to run on, e.g., `cuda device=0/1/2/3` or `device=cpu`
dnn | False | Use OpenCV DNN for ONNX inference
plots | False | Save plots and images during train/val
rect | False | Rectangular val with each batch collated for minimum padding
split | val | Dataset split to use for validation, e.g., 'val', 'test', or 'train'

[Docs](https://docs.ultralytics.com/modes/val/)

## Supported Pre-Trained Models 🤖

| Model | YAML | Size (pixels) | mAPval 50-95 | Speed CPU ONNX (ms) | Speed A100 TensorRT (ms) | Params (M) | FLOPs (B) |
|-------|------|---------------|--------------|---------------------|--------------------------|------------|-----------|
| yolov5nu.pt | yolov5n.yaml | 640 | 34.3 | 73.6 | 1.06 | 2.6 | 7.7 |
| yolov5su.pt | yolov5s.yaml | 640 | 43.0 | 120.7 | 1.27 | 9.1 | 24.0 |
| yolov5mu.pt | yolov5m.yaml | 640 | 49.0 | 233.9 | 1.86 | 25.1 | 64.2 |
| yolov5lu.pt | yolov5l.yaml | 640 | 52.2 | 408.4 | 2.50 | 53.2 | 135.0 |
| yolov5xu.pt | yolov5x.yaml | 640 | 53.2 | 763.2 | 3.81 | 97.2 | 246.4 |
| yolov5n6u.pt | yolov5n6.yaml | 1280 | 42.1 | 211.0 | 1.83 | 4.3 | 7.8 |
| yolov5s6u.pt | yolov5s6.yaml | 1280 | 48.6 | 422.6 | 2.34 | 15.3 | 24.6 |
| yolov5m6u.pt | yolov5m6.yaml | 1280 | 53.6 | 810.9 | 4.36 | 41.2 | 65.7 |
| yolov5l6u.pt | yolov5l6.yaml | 1280 | 55.7 | 1470.9 | 5.47 | 86.1 | 137.4 |
| yolov5x6u.pt | yolov5x6.yaml | 1280 | 56.8 | 2436.5 | 8.98 | 155.4 | 250.7 |
| YOLOv8n || 640 | 37.3 | 80.4 | 0.99 | 3.2 | 8.7 |
| YOLOv8s || 640 | 44.9 | 128.4 | 1.20 | 11.2 | 28.6 |
| YOLOv8m || 640 | 50.2 | 234.7 | 1.83 | 25.9 | 78.9 |
| YOLOv8l || 640 | 52.9 | 375.2 | 2.39 | 43.7 | 165.2 |
| YOLOv8x || 640 | 53.9 | 479.1 | 3.53 | 68.2 | 257.8 |

Please note that the availability of these models may vary, and some models may still be in experimental stages.

See also [Yolov6](https://docs.ultralytics.com/models/yolov6/#usage-examples).

More detail about [Yolov5](https://docs.ultralytics.com/models/yolov5/) and  [Yolov8](https://docs.ultralytics.com/models/yolov8/).