import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

from attributes_recognition_module.utils import calculate_metrics, visualize_metrics, save_metrics_to_csv
from attributes_recognition_module.gradnorm import gradnorm


#weights = torch.tensor([0.35,0.35,0.1,0.1,0.1])
# weghts = [0.35, 0.35, 0.1, 0.1, 0.1]

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, model_dir,model_name, exp_name, alpha=0.12, lr2=5e-4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.device = device
        self.model_dir = model_dir
        self.model_name = model_name
        self.best_val_loss = float('inf')
        self.exp_name = exp_name
        self.alpha  = alpha
        self.lr2 = lr2

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_task = 5
        for inputs, labels in tqdm(self.train_loader, desc="Training"):
            inputs, labels = inputs.to(self.device), [label.to(self.device) for label in labels]
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            #loss_tensor = torch.zeros(5)
            
            #for index, el in enumerate(loss):
            #    loss_tensor[index] = el
                
            
            # loss è una lista di 5 elementi
            # devo chiamare backward su ogni elemento della lista
            # e poi fare la step
            
            #weighted_loss = torch.dot(weights, loss_tensor)
            #weighted_loss.backward()
             #### METTERE GRAD NORM ###
            # convert a list to tensor

            log_weights, log_loss = gradnorm( self.iters, torch.tensor(loss), self.model.get_last_shared_layer(), self.alpha,self.lr2 , self.optimizer, log = False)
            
            
            total_loss += sum(log_loss).item() 
            #sum(loss).backward()

            #self.optimizer.step()
            
            
               
        total_loss/=num_task
        total_loss/=len(self.train_loader)
        return total_loss 
      

            

    def validate(self):
        self.model.eval()
        all_labels = [] # lista dove vengono conservate tutte le label
        all_predictions = []    # lista dove vengono convevate tutte le predizioni

        task_names = ["Upper Color", "Lower Color", "Gender", "Bag", "Hat"]


        
        task_metrics = {task: {"Loss": 0.0, "Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0} for task in task_names}
        overall_metrics = {"Loss": 0.0, "Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0}

        task_all_labels = {task: [] for task in task_names}
        task_all_predictions = {task: [] for task in task_names}

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validation"):
                inputs, labels = inputs.to(self.device), [label.to(self.device) for label in labels]
                outputs = self.model(inputs)    
                
                losses = self.criterion(outputs, labels) # loss della validation (campioni in mnumero del batch)

                for task, loss in zip(task_names, losses):  # coppia (task_hat, loss_hat)
                    task_metrics[task]["Loss"] += loss.item()

                for task, (label, output) in zip(task_names, zip(labels, outputs)): # for va 5 volte
                    task_all_labels[task].extend(label.cpu().numpy())
                    task_all_predictions[task].extend(torch.argmax(output, dim=1).cpu().numpy())
                    #print(f"{task} = > Prediction {task_all_predictions[task]} Label {task_all_labels[task]} ")
                    all_labels.extend(label.cpu().numpy())
                    all_predictions.extend(torch.argmax(output, dim=1).cpu().numpy())

            for task in task_names:
                task_metrics[task]["Loss"] /= len(self.val_loader)  # loss/n_batch (381)
                task_metrics[task]["Accuracy"], task_metrics[task]["Precision"], task_metrics[task]["Recall"] = calculate_metrics(task_all_labels[task], task_all_predictions[task])
                overall_metrics["Loss"] += task_metrics[task]["Loss"]
                #print (f"{task} => {task_metrics[task]}")
        overall_metrics["Loss"] /= len(task_names)
        overall_metrics["Accuracy"], overall_metrics["Precision"], overall_metrics["Recall"] = calculate_metrics(all_labels, all_predictions)
        #print(f"Overall Metrics: {overall_metrics}")
        
        return  overall_metrics, task_metrics, task_names
        

    def save_model(self):
        torch.save(self.model.state_dict(),  self.model_dir + self.model_name + ".pth")

    def train(self):
        self.iters = 0
        for epoch in range(self.num_epochs):
            self.iters = epoch
            train_loss = self.train_epoch()
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Training Loss: {train_loss:.4f}")

            overall_metrics, task_metrics, task_names = self.validate()
            #print(f"Epoch {epoch + 1}/{self.num_epochs}, Validation Loss: {val_loss:.4f}")
            visualize_metrics(epoch + 1, task_names, task_metrics, overall_metrics)
            matrics_filename = self.model_dir+ "metrics_"+self.exp_name+".csv"
            save_metrics_to_csv(epoch + 1, task_names, task_metrics, overall_metrics, matrics_filename)
            if overall_metrics["Loss"] < self.best_val_loss:
                print("Validation loss improved. Saving the model.")
                self.best_val_loss = overall_metrics["Loss"]
                print(f"Best Validation Loss: {self.best_val_loss:.4f}")
                self.save_model()
            else:
                print("Validation loss did not improve.")
                
                
            self.scheduler.step()
    

# Example usage:

""" # Set the path where you want to save the model
model_save_path = "path/to/save/model.pth"

# Initialize Trainer
optimizer = torch.optim.Adam(self.model.parameters(), lr=self.start_lr, weight_decay=self.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs, eta_min=self.end_lr)

trainer = Trainer(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs, device, model_save_path)

# Train and validate the model
trainer.train() """
