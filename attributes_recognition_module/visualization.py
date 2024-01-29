
import os
import sys


current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, ".."))
sys.path.append(project_path)

from torchvision.io.image import read_image, ImageReadMode
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import XGradCAM
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
import torch 
from torchsummary import summary    
from torchvision import transforms
from PIL import Image
from attributes_recognition_module.src.MultiTaskNN import MultiTaskNN
from captum.attr import LayerGradCam
from torchcam.methods import GradCAM

dict_layer = {
        0: "attention_module_upper_color",
        
        1: "attention_module_lower_color",

        2: "attention_module_gender",

        3: "attention_module_bag",

        4: "attention_module_hat"
        
        }


def draw_activation_map(model,input_tensor,image, dict_layer = dict_layer):
        
    for key in dict_layer:
        layer_name = dict_layer[key]
        if hasattr(model, layer_name):
            target_layer = getattr(model, layer_name)
            with XGradCAM(model, target_layer=target_layer) as cam_extractor:
                # Retrieve the CAM by passing the model and the input tensor to the CAM extractor.
                output = model(input_tensor)
                class_idx=output[key].squeeze(0).argmax().item()
                output_final = output[key].squeeze(0)
                output_final = output_final.unsqueeze(0)
                activation_map = cam_extractor(class_idx=class_idx, scores=output_final) 
                print(activation_map)
                plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()
                result = overlay_mask(image, to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
                plt.imshow(result); 
                plt.axis('off'); 
                plt.tight_layout() 
                plt.title(f"Task ID: {key} - Pred: {class_idx}")
                plt.waitforbuttonpress()


        else:
            print(f"Layer {layer_name} not found in the model")

if __name__ == "__main__":
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
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

    image_path = "attributes_recognition_module\\par_dataset\\training_set\\0002_2_25027_160_75_118_274.jpg"
    image = read_image(image_path, ImageReadMode.RGB)
    data_transforms = transforms.Compose([transforms.Resize((96,288)), transforms.ToTensor()])
    img = Image.open(f"{image_path}").convert("RGB")
    img1 = data_transforms(img)
    input = img1.unsqueeze(0).to(device)

    # gettattr del mio modello voglio visualizzare tutti i layer del modello e accedervi 
    # con getattr
    #summary(model, input_size=(3, 96, 288))

    

    draw_activation_map( model, input, img, dict_layer = dict_layer)
 