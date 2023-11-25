# Artificial Vision Project 📹
This project involves the development of a sophisticated video analysis software that performs real-time human detection, tracking, attribute recognition, and behavior analysis from input video streams. The software identifies individuals in a scene, tracks their movements, recognizes attributes such as gender, presence of a bag or hat, and clothing colors. It also analyzes behaviors based on movements and persistence within defined Regions of Interest (ROIs). 

The software integrates a speech synthesis module to announce each newly tracked individual in the video. The results of the analysis are recorded in a specific format for further evaluation against a ground truth to assess the accuracy of detection, tracking, attribute recognition, and behavior analysis. 

The software is designed to handle videos with a resolution of 1920x1080, recorded in conditions similar to a typical atrium setting. It accepts a configuration file that specifies the position of two rectangular ROIs, with coordinates relative to the image size. The software is expected to consider a person in a ROI if the center of the bounding box is inside the ROI.

## Getting Started 🚀
Instructions on how to get a copy of the project up and running on your local machine.

### Featurs 🎯
- **Human Detection:** Identifies and localizes humans within images or video frames.
- **Human Tracking:** Tracks detected humans across multiple frames to maintain their identity.
- **Evaluation Pedestrian Attribute:** This feature involves the analysis and recognition of specific attributes associated with each detected pedestrian. These attributes can include gender, presence of a bag or hat, and clothing colors. The system is designed to accurately identify and evaluate these attributes in real-time.
- **Real-time Processing:** Designed for real-time operation with live video streams.

### Installing🔧
A step-by-step guide on how to install the project.
#### 1. Clone the repository
```bash
git clone https://github.com/SatoshiNakamoto99/ProjectWork.git
cd ProjectWork
```
#### 2. Create a conda environment
```bash
conda create --name < your_env_name > python=3.11.5 
```
#### 3. Install dependencies 
```bash
conda activate 
pip3 install -r requirements.txt
```
#### 4. Install PyTorch 

PyTorch is an open-source machine learning library for Python, based on Torch, used for applications such as natural language processing and artificial neural networks.

To install PyTorch, you can use pip or conda commands. However, the command you use depends on your system configuration (OS, Python version, CUDA version if applicable). 

Here is a general example of how to install PyTorch using pip3 for Windows and CUDA ToolKit 12.1:

```bash
 pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
For a more specific installation command tailored to your system, please refer to the [official PyTorch website](https://pytorch.org/get-started/locally/). There, you can select your preferences and get the command that suits your needs. 

Note that if you don't have a GPU you can install a version fo CPU.

## Usage 📖
Instructions on how to use the project or its features are described in [Prepare_dataset.md](./docs/prepare_dataset.md) and [QuickStart.md](./docs/dashboard.md)

## Contributing 🤝
Contributions are welcome! Feel free to fork the repository, make improvements, and submit pull requests. Please follow the Contributing Guidelines.

## License 📜 
This project is licensed under the [MIT License] - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- This project was based on the work done in the [human_dettrack](https://github.com/vankhoa21991/human_dettrack) repository. We appreciate their contributions to the open-source community.

- Group Member:
    - Nocerino Antonio - [GitHub](https://github.com/SatoshiNakamoto99)
    - Spinelli Gianluigi - [GitHub](https://github.com/givnlvigi)
    - Sara 
    - Prisco