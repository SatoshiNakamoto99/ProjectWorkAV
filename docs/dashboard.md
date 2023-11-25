##  Training and Validation🏋️‍♀️
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
 python TrainVal.py --config config/your_config_file.yaml --mode train
```
#### 💡 Future Improvements

This current version for training a model is still naive and straightforward. In the near future, we plan to add the ability to perform K-Fold Cross Validation during training. This technique will help to ensure that our model does not overfit the training data and can generalize well to unseen data. Stay tuned for this exciting update!

###  Validation of a Model with Configuration File🎯

Validation is an essential part of training a model. It allows us to evaluate the model's performance on a separate dataset that was not used during training, which gives us a better understanding of how the model will perform on unseen data.

The `TrainerYolo` class in the `TrainVal.py` script also supports validation. The `validate` method of the `TrainerYolo` class performs validation on the model. It uses the same configuration file as the training phase, but the `mode` parameter in the configuration file should be set to 'validate'.

Here is how you can run the validation:

```bash
python TrainVal.py --config config/your_config_file.yaml --mode validate
```

### Supported Pre-Trained Models 🤖

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