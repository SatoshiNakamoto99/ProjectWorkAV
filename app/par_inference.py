import sys
import torch
import os
from PIL import Image
from torchvision import transforms

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, ".."))
sys.path.append(project_path)

from attributes_recognition_module.src.MultiTaskNN import MultiTaskNN

class LabelDict():
    def __init__(self) -> None:
        self.names_dict = {}
        self.create_dict()
        
    def create_dict(self):
        
        self.names_dict = {
            "upper_color" : {},
            "lower_color" : {},
            "gender" : {},
            "hat" : {},
            "bag" : {}
        }
        
        self.names_dict["upper_color"] = {
            0 : "black",
            1 : "blue",
            2 : "brown",
            3 : "gray",
            4 : "green",
            5 : "orange",
            6 : "pink",
            7 : "purple",
            8 : "red",
            9 : "white",
            10 : "yellow"
        }
        
        self.names_dict["lower_color"] = self.names_dict["upper_color"]
        
        self.names_dict["gender"] = {
            0 : "male",
            1 : "female"
        }
        
        self.names_dict["hat"] = {
            0 : "false",
            1 : "true"
        }
        
        self.names_dict["bag"] = {
            0 : "false",
            1 : "true"
        }
        
        
    def __getitem__(self, key):
        return self.names_dict[key]
        

class PARModuleInference():
    def __init__(self, model_path) -> None:
        self.model_path = model_path
        self.labels_dict = LabelDict()

        self.data_transforms = transforms.Compose([transforms.Resize((96,288)), transforms.ToTensor()])
        
        # Initialize model
        convnext_version = "v1"
        am_type = "CBAM"
        device = torch.device('cpu') # for index cuda debugging
        self.model = MultiTaskNN(1024, convnext_version, am_type, device).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()    
              
    def prediction(self, image):
        
        image_pil = Image.fromarray(image).convert("RGB")
        input = self.data_transforms(image_pil)
        input = input.unsqueeze(0)
        
        preds = self.model(input) 
        results = self.convert_model_outputs(preds)
        return results    
            
    def convert_model_outputs(self, preds):
        
        results = {}
        
        preds_upper_color, preds_lower_color, preds_gender, preds_bag, preds_hat = preds
        
        uc_confidence = torch.max(preds_upper_color).item()
        uc_index = torch.argmax(preds_upper_color).item()
        uc_string = self.labels_dict["upper_color"][uc_index]
        
        lc_confidence = torch.max(preds_lower_color).item()
        lc_index = torch.argmax(preds_lower_color).item()
        lc_string = self.labels_dict["lower_color"][lc_index]
        
        gender_confidence= torch.max(preds_gender).item()
        gender_index = torch.argmax(preds_gender).item()
        gender_string = self.labels_dict["gender"][gender_index]
        
        bag_confidence = torch.max(preds_bag).item()
        bag_index = torch.argmax(preds_bag).item()
        bag_string = self.labels_dict["bag"][bag_index]
        
        hat_confidence = torch.max(preds_hat).item()
        hat_index = torch.argmax(preds_hat).item()
        hat_string = self.labels_dict["hat"][hat_index]
        
        # write in results dict
        results["upper_color"] = [uc_string, uc_confidence]
        results["lower_color"] = [lc_string, lc_confidence]
        results["gender"] = [gender_string, gender_confidence]
        results["bag"] = [bag_string, bag_confidence]
        results["hat"] = [hat_string, hat_confidence]
        
        return results
     
        
 

# example of usage
if __name__ == "__main__":
    pass
    