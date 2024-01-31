import sys
import torch
from torch import nn
import os
from PIL import Image
from torchvision import transforms
from time import time
import csv
import pandas as pd
import json

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
        
        # self.data_transforms = transforms.Compose([transforms.Resize((144,90)), transforms.ToTensor()])
        self.data_transforms = transforms.Compose([transforms.Resize((192,64)), transforms.ToTensor()])
        
        # Initialize model
        convnext_version = "v1"
        am_type = "CBAM"
        #select device
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu') # for index cuda debugging
        #device = torch.device('cpu') # for index cuda debugging
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
        
        uc_index = torch.argmax(preds_upper_color).item()
        uc_string = self.labels_dict["upper_color"][uc_index]
        
        lc_index = torch.argmax(preds_lower_color).item()
        lc_string = self.labels_dict["lower_color"][lc_index]
        
        gender_index = torch.argmax(preds_gender).item()
        gender_string = self.labels_dict["gender"][gender_index]
        
        bag_index = torch.argmax(preds_bag).item()
        bag_string = self.labels_dict["bag"][bag_index]
        
        hat_index = torch.argmax(preds_hat).item()
        hat_string = self.labels_dict["hat"][hat_index]
        
        # write in results dict
        results["upper_color"] = uc_string
        results["lower_color"] = lc_string
        results["gender"] = gender_string
        results["bag"] = bag_string
        results["hat"] = hat_string
        
        return results
        
    # def print_results(results):
        
    #     print(results)
    
    # def write_results_on_image():
    #     pass    
  
    # def write_on_csv(self, results):
        
    #     # rf = pd.DataFrame(results)
    #     # rf.to_csv(f'{self.results_folder_name}/{self.csv_name}')
        
    #     results_txt = json.dumps(results, indent = 4)
    #     with open(f"{self.results_folder_name}/results.txt", "w") as f:
    #         f.write(results_txt)
        
            
        # def load_image(self):
    #     pass
    
    # def image_reader(self):
    #     pass
    #     # for 
        
    # def image_reader(self):
    #     images_dir = [d for d in os.listdir(self.images_path)]
    #     for image_dir in images_dir:
            
    #         # Load image
    #         path_name = f"{self.images_path}\\{image_dir}\\image.jpg"
            
    #         if os.path.isfile(path_name):
    #             img = Image.open(path_name).convert("RGB")
    #             # Apply transforms
    #             img = self.data_transforms(img)
    #         else:
    #             print(f"IMAGE {path_name} not found, skipping ... ")
            
    #         yield img, image_dir
            
    # def image_reader_generic(self):
        
    #     for image_dir in os.listdir(self.images_path):
            
    #         path_name = f"{self.images_path}\\{image_dir}"
        
    #         if os.path.isfile(path_name):
    #             img = Image.open(path_name).convert("RGB")
    #             # Apply transforms
    #             img = self.data_transforms(img)
    #         else:
    #             print(f"IMAGE {path_name} not found, skipping ... ")
            
    #         yield img, path_name

    # def inference(self):
        
    #     people = []
        
    #     for input, input_name in self.image_reader_generic():
    #         input = input.unsqueeze(0)  # add batch dimension
    #         preds = self.model(input)        
    #         results = self.convert_model_outputs(preds)            
    #         people.append(results)
            
    #     return people
            
            
            
        
        
        
        
        
        
    
        
 

# example of usage
if __name__ == "__main__":
    pass
    
    # model_path = "attributes_recognition_module/model/MultiTaskNN_ConvNeXt_v1_CBAM.pth"
    # # images_path = "test_colors_on_PAR_dataset\\resultsPAR"
    # images_path = "C:\Users\gianl\Desktop\uni\secondo_anno_AI\artificial_vision\PAR_Project_ForAV\progetto\par_dataset"
    
    # inference_model = PARModuleInference(model_path=model_path, images_path=images_path)
    # inference_model.inference()    

    

    
    
    
    
