
import os
import sys

from torchvision.io.image import read_image, ImageReadMode
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
import torch 
from torchsummary import summary    
from torchvision import transforms
from PIL import Image
from attributes_recognition_module.src.MultiTaskNN import MultiTaskNN

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, ".."))
sys.path.append(project_path)

def draw_activation_map(model, input_tensor, image, filename):
    with SmoothGradCAMpp(model) as cam_extractor:
        # Retrieve the CAM by passing the model and the input tensor to the CAM extractor.
        output = model(input_tensor)
        activation_map = cam_extractor(class_idx=output.squeeze(0).argmax().item())
        result = overlay_mask(image, to_pil_image(activation_map[0].squeeze[0], mode='F'), alpha=0.5)
        plt.imshow(result); plt.axis('off'); plt.tight_layout()
        plt.savefig(filename)
data_transforms = transforms.Compose([
        transforms.Resize((96,288)),
        transforms.ToTensor()
    ])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    #print cuda memory usage
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    # print clear cache
    torch.cuda.empty_cache()
    #print cuda memory usage
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

model = torch.load(".\models\MultiTaskNN_ConvNeXt_v1_CBAM_lr_1e-05_wd_0.01_epochs_100\MultiTaskNN_ConvNeXt_v1_CBAM.pth")
image_path = ".\\par_dataset\\training_set\\0002_2_25027_160_75_118_274.jpg"
image = read_image(image_path, ImageReadMode.RGB)
input_tensor = data_transforms(image) if not isinstance(image, torch.Tensor) else image.unsqueeze(0).to(device)

draw_activation_map(model, input_tensor, image, "activation_map.jpg")
