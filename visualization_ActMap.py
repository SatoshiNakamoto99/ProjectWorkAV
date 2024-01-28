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

def draw_activation_map(model, input):
    with torch.no_grad():
        # Passa l'input attraverso il modello
        outputs = model(input)

        # Estrai le attivazioni delle feature map da ogni output
        
    


    



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    # Stampa l'utilizzo della memoria GPU
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    # Svuota la cache GPU
    torch.cuda.empty_cache()
    # Stampa l'utilizzo della memoria GPU
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

# Carica il modello (assicurati che il percorso sia corretto)
model = MultiTaskNN(1024,device =  device).to(device)
model.load_state_dict(torch.load("attributes_recognition_module\\model\\MultiTaskNN_ConvNeXt_v1_CBAM.pth"))
model.eval()
image_path = "attributes_recognition_module\\par_dataset\\training_set\\0002_2_25027_160_75_118_274.jpg"
image = read_image(image_path, ImageReadMode.RGB)
data_transforms = transforms.Compose([transforms.Resize((96,288)), transforms.ToTensor()])
img = Image.open(f"{image_path}").convert("RGB")
#from torchsummary import summary
#summary(model, input_size=(3, 288, 96))  # Sostituisci channels, height e width con le dimensioni del tuo input

# Apply transforms
img = data_transforms(img)
input = img.unsqueeze(0).to(device)
draw_activation_map(model, input)
