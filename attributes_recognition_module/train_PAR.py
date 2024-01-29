# set progetto as Python source root


import argparse
from torch.utils.data import  DataLoader
from torchvision import transforms
import torch

import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, ".."))
sys.path.append(project_path)

from attributes_recognition_module.src.MultiTaskNN import MultiTaskNN
from attributes_recognition_module.src.Trainer import Trainer
from attributes_recognition_module.src.CustomImageDataset import CustomImageDataset
from attributes_recognition_module.src.CustomSempler import CustomSampler
from attributes_recognition_module.src.utils import create_path

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
                    prog='train script',
                    description='train the multitask NN on PAR task')
    
    parser.add_argument('--reduced', action='store_true', help='train with a reduced size dataset')
    opt = parser.parse_args()
    
    if (opt.reduced == True):
        print("It will be used a reduced dataset")
    # Load the dataset here and return the data and labels as a list of tuples
        train_annotation_file = "attributes_recognition_module\\par_dataset\\training_set_reduced.txt"
        validation_annotation_file = "attributes_recognition_module\\par_dataset\\validation_set_reduced.txt"
        train_img_dir = "attributes_recognition_module\\par_dataset\\training_set_reduced\\"
        validation_img_dir = "attributes_recognition_module\\par_dataset\\validation_set_reduced\\"
    else:
        print("It will be used the entire dataset")
        train_annotation_file = "attributes_recognition_module\\par_dataset\\training_set.txt"
        validation_annotation_file = "attributes_recognition_module\\par_dataset\\validation_set.txt"
        train_img_dir = "attributes_recognition_module\\par_dataset\\training_set\\"
        validation_img_dir = "attributes_recognition_module\\par_dataset\\validation_set\\"

    data_transforms_val = transforms.Compose([transforms.Resize((96,288)), transforms.ToTensor()])
    data_trasfporms_train = transforms.Compose([transforms.Resize((96, 288)), transforms.ToTensor()])
    # Load the dataset if do you want to see the output of the dataset saved  you casn use verbose=True
   
   
    train_dataset = CustomImageDataset(train_annotation_file, train_img_dir, data_trasfporms_train)
    val_dataset = CustomImageDataset(validation_annotation_file, validation_img_dir, data_transforms_val)
    
    batch_size = 64

    if (opt.reduced == True):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        custom_sampler = CustomSampler(train_dataset, batch_size)
        train_dataloader = DataLoader(train_dataset,  sampler=custom_sampler, shuffle=False)
       
    
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)


    if torch.cuda.is_available():
        #print cuda memory usage
        print(torch.cuda.memory_summary(device=None, abbreviated=False))
        # print clear cache
        torch.cuda.empty_cache()
        #print cuda memory usage
        print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # Initialize model
    convnext_version = "v1"
    am_type = "CBAM"
    #select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu') # for index cuda debugging
    model = MultiTaskNN(1024, convnext_version, am_type, device).to(device)


    # Initialize Trainer
    start_lr = 1e-5
    end_lr = 1e-7
    weight_decay = 1e-2
    num_epochs = 64
    model_name = "MultiTaskNN_ConvNeXt_"+ convnext_version+"_"+am_type
    exp_name = model_name+"_lr_"+str(start_lr)+"_wd_"+str(weight_decay)+"_epochs_"+str(num_epochs)+"_batch_size_"+str(batch_size)+"_v1"
    model_dir = "models/"+exp_name+"/"
    create_path(model_dir)
    model_save_path = model_dir + model_name + ".pth"
    patience = 5
    # if model  dir non esiste crealo 
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=end_lr)

    trainer = Trainer(model, train_dataloader, val_dataloader, model.compute_loss, optimizer, scheduler, num_epochs, device,model_dir, model_name, exp_name, patience = patience )
   
    # Train and validate the model
    trainer.train()
    #trainer.validate()