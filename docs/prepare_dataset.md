# Dataset Setup 📚

## Download Dataset 📥

The first step in preparing your environment is to download the necessary datasets. The datasets required for this project are the MOT17 and CrowdHuman datasets. 

You can download these datasets from their official websites:

- [MOT17 Dataset](https://motchallenge.net/data/MOT17/)
- [CrowdHuman Dataset](https://www.crowdhuman.org/)

After downloading, the datasets should be organized in the following structure. Please note that within the `train` directory, the `Images` folder should contain the extracted contents of `CrowdHuman_train01.zip`, `CrowdHuman_train02.zip`, and `CrowdHuman_train03.zip`. Similarly,  within the `val` directory, the `Images`folder should contain the extracted contents of `CrowdHuman_val.zip`:


```plaintext
datasets
|
├── MOT17/
|   ├── train
|   ├── test
│   |
├── CrowdHuman
│   ├── annotation_train.odgt
│   ├── annotation_val.odgt
│   ├── train
│   │   ├── Images  // Contains the extracted contents of CrowdHuman_train01.zip, CrowdHuman_train02.zip, and CrowdHuman_train03.zip
│   │   ├── CrowdHuman_train01.zip
│   │   ├── CrowdHuman_train02.zip
│   │   ├── CrowdHuman_train03.zip
│   ├── val
│   │   ├── Images // Contains the extracted contents of CrowdHuman_val.zip
│   │   ├── CrowdHuman_val.zip 
```
## Prepare Dataset 🛠️

### Step 1 - Conversion to COCO Format Dataset 🔄

The initial step entails the transformation of the MOT17 and CrowdHuman datasets into the COCO format. This can be accomplished by executing the following commands:


```bash
 python ./datasets/crowdhuman_to_coco.py --data_path ./datasets/CrowdHuman/
```
```bash
 python ./datasets/mot17_to_coco.py --data_path datasets/MOT17
```

Subsequent to the conversion to the COCO format, the directory structure should appear as follows:

```plaintext
datasets
|
├── MOT17/
|   ├── annotations // coco format
|   |    ├── train_half.json
|   |    ├── val_half.json
|   |    ├── train.json
|   |    ├── test.json
|   ├── train
|   ├── test
│   |
├── CrowdHuman
|   ├── annotations // coco format
|   |    ├── train.json
|   |    ├── val.json
│   ├── annotation_train.odgt
│   ├── annotation_val.odgt
│   ├── train
│   │   ├── Images  // Contains the extracted contents of CrowdHuman_train01.zip, CrowdHuman_train02.zip, and CrowdHuman_train03.zip
│   │   ├── CrowdHuman_train01.zip
│   │   ├── CrowdHuman_train02.zip
│   │   ├── CrowdHuman_train03.zip
│   ├── val
│   │   ├── Images // Contains the extracted contents of CrowdHuman_val.zip
│   │   ├── CrowdHuman_val.zip 
```

### Step 2 - Conversion to YOLO Format Dataset 🔄
The second step entails the transformation of the MOT17 and CrowdHuman datasets into the COCO format. This can be accomplished by executing the following commands:
```bash
python .\datasets\coco_to_yolo.py --data_path .\datasets\CrowdHuman\ --split val train --dataset crowdhuman -np 0
```
```bash
 python ./datasets/coco_to_yolo.py --data_path ./datasets/MOT17 --image_path ./datasets/MOT17/train --split val_half train_half --dataset mot -np 0
```

Subsequent to the conversion to the YOLO format, the directory structure should appear as follows:

```plaintext
datasets
|
├── MOT17/
|   ├── annotations // coco format
|   |    ├── train_half.json
|   |    ├── val_half.json
|   |    ├── train.json
|   |    ├── test.json
|   ├── train
|   ├── test
|   ├── yolo
|   |    ├── train_half
|   |    ├── val_half
|   |    ├── dataset.yaml
│   |
├── CrowdHuman
|   ├── annotations // coco format
|   |    ├── train.json
|   |    ├── val.json
│   ├── annotation_train.odgt
│   ├── annotation_val.odgt
│   ├── train
│   │   ├── Images  // Contains the extracted contents of CrowdHuman_train01.zip, CrowdHuman_train02.zip, and CrowdHuman_train03.zip
│   │   ├── CrowdHuman_train01.zip
│   │   ├── CrowdHuman_train02.zip
│   │   ├── CrowdHuman_train03.zip
│   ├── val
│   │   ├── Images // Contains the extracted contents of CrowdHuman_val.zip
│   │   ├── CrowdHuman_val.zip 
|   ├── yolo
|   |    ├── train
|   |    ├── val
|   |    ├── dataset.yaml

```

### Step 3 - Mix the MOT17 with CrowdHuman to Create a New Dataset 🧩

In this step, we will merge the MOT17 and CrowdHuman datasets to create a new, combined dataset. This will provide a more diverse range of data for our model to train on, improving its ability to generalize to new data. This can be accomplished by executing the following command:

```bash
 python datasets/mix_mot_ch.py --paths datasets --out_path ./datasets/all_data
 
```
Subsequent to the mix, the directory structure should appear as follows:
```plaintext
datasets
├── all
|   ├── train
|   ├── val
|   ├── dataset.yaml
|
├── MOT17/
|   ├── annotations // coco format
|   |    ├── train_half.json
|   |    ├── val_half.json
|   |    ├── train.json
|   |    ├── test.json
|   ├── train
|   ├── test
|   ├── yolo
|   |    ├── train_half
|   |    ├── val_half
|   |    ├── dataset.yaml
│   |
├── CrowdHuman
|   ├── annotations // coco format
|   |    ├── train.json
|   |    ├── val.json
│   ├── annotation_train.odgt
│   ├── annotation_val.odgt
│   ├── train
│   │   ├── Images  // Contains the extracted contents of CrowdHuman_train01.zip, CrowdHuman_train02.zip, and CrowdHuman_train03.zip
│   │   ├── CrowdHuman_train01.zip
│   │   ├── CrowdHuman_train02.zip
│   │   ├── CrowdHuman_train03.zip
│   ├── val
│   │   ├── Images // Contains the extracted contents of CrowdHuman_val.zip
│   │   ├── CrowdHuman_val.zip 
|   ├── yolo
|   |    ├── train
|   |    ├── val
|   |    ├── dataset.yaml

```
## Conclusion 🎉

With all the pieces of the puzzle now in place, we are ready to embark on our adventure! The datasets are prepared, the formats are correct, and everything is set up for us to start training our model. Let's dive in and start our machine learning journey! 🚀

For more details on the next steps, please refer to the [dashboard guide](./dashboard.md).
