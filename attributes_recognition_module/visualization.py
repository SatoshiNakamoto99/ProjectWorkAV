
import os
import sys


current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, ".."))
sys.path.append(project_path)

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
from captum.attr import LayerGradCam
from torchcam.methods import GradCAM


def draw_activation_map(model,input_tensor):
    if hasattr(model, "attention_module_upper_color"):
        target_layer = getattr(model, "attention_module_upper_color")
        with SmoothGradCAMpp(model,target_layer ) as cam_extractor:
            # Retrieve the CAM by passing the model and the input tensor to the CAM extractor.
            output = model(input_tensor)
            print(output)

            # (
            #     tensor([[6.2300e-07, 2.6075e-08, 2.6075e-08, 2.6075e-08, 2.6075e-08, 2.6075e-08, 3.9647e-03, 2.6075e-08, 9.9603e-01, 2.6075e-08, 2.6075e-08]], grad_fn=<SoftmaxBackward0>), 
            #     tensor([[1.5259e-11, 1.5259e-11, 1.5259e-11, 1.5259e-11, 1.5259e-11, 2.2317e-06, 3.1680e-07, 5.5753e-11, 1.0000e+00, 1.5259e-11, 1.5259e-11]], grad_fn=<SoftmaxBackward0>), 
            #     tensor([[1.8760e-09, 1.0000e+00]], grad_fn=<SoftmaxBackward0>), 
            #     tensor([[1.0000e+00, 1.5133e-13]], grad_fn=<SoftmaxBackward0>), 
            #     tensor([[1.0000e+00, 1.1128e-10]], grad_fn=<SoftmaxBackward0>)
            #     )

            class_idx=output[0].squeeze(0).argmax().item()
            print(f"class_idx = {class_idx}")
            print(type(class_idx))
            print(f'outputtttt: {output}')
            print(f"output = {output[0][0].squeeze(0)}")
            activation_map = cam_extractor(class_idx=class_idx, scores=output[0].squeeze(0))
    else:
        print("No attention module")
        #result = overlay_mask(image, to_pil_image(activation_map[0].squeeze[0], mode='F'), alpha=0.5)
        #plt.imshow(result); plt.axis('off'); plt.tight_layout()
  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

if torch.cuda.is_available():
    #print cuda memory usage
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    # print clear cache
    torch.cuda.empty_cache()
    #print cuda memory usage
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

model = MultiTaskNN(1024,device =  device).to(device)
model.load_state_dict(torch.load("attributes_recognition_module\\model\\MultiTaskNN_ConvNeXt_v1_CBAM.pth", map_location=device))
model.eval()

image_path = "attributes_recognition_module\\par_dataset\\training_set_reduced\\CAM30-2014-02-20-20140220171915-20140220172439-tarid43-frame1944-line2.jpg"
image = read_image(image_path, ImageReadMode.RGB)
data_transforms = transforms.Compose([transforms.Resize((96,288)), transforms.ToTensor()])
img = Image.open(f"{image_path}").convert("RGB")
#from torchsummary import summary
#summary(single_task_upper_color, input_size=(3, 288, 96))  # Sostituisci channels, height e width con le dimensioni del tuo input
# Apply transforms
img = data_transforms(img)
input = img.unsqueeze(0).to(device)

draw_activation_map( model, input)
 