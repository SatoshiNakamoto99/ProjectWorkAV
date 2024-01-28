import argparse
from itertools import cycle
from torch.utils.data import  DataLoader
from torchvision import transforms
import torch
from attributes_recognition_module.src.MultiTaskNN import MultiTaskNN
from attributes_recognition_module.src.Trainer import Trainer
from attributes_recognition_module.src.CustomImageDataset import CustomImageDataset
from attributes_recognition_module.src.utils import create_path
import numpy as np
from torch.utils.data.sampler import Sampler
import random

""" i = 0
        #i<len(self.dataset) // self.batch_size
        while len(self.total_index) < len(self.dataset):
            batch_indices_set = set()
            for task_idx in range(self.num_task):
                while True:
                    #try:
                        # Cerca di ottenere l'indice successivo dall'iteratore dell'attività corrente
                        if task_idx != 0 and task_idx !=  1:
                            try:
                                idx = next(self.task_iterators[task_idx][0])
                                added = idx not in batch_indices_set
                                batch_indices_set.add(idx)
                                if added:
                                    self.total_index.add(idx)
                                    break
                            except StopIteration:
                                task_list = self.dataset.get_task_iterator(task_idx)
                                task_list = random.sample(task_list, len(task_list))
                                self.task_iterators[task_idx][0] = iter(task_list)
                                continue

                        else:
                            j = 1
                            while j < 12:
                                #print(j)
                                try:
                                    idx = next(self.task_iterators[task_idx][j-1])
                                    added = idx not in batch_indices_set
                                    batch_indices_set.add(idx)
                                    if added:
                                        self.total_index.add(idx)
                                        j = j + 1
                                except StopIteration:
                                    task_list = self.dataset.get_task_iterator(task_idx)[j]
                                    task_list = random.sample(task_list, len(task_list))
                                    self.task_iterators[task_idx][j-1] = iter(task_list)
                                    continue
                            break
                        # Aggiunge l'indice all'insieme
                        
                        
                        # Se l'aggiunta ha avuto successo, esci dal ciclo while
                        
                        
                    
            
            for _ in range (self.batch_size - self.num_task-10-10):
                while True:
                    #idx = np.random.randint(0, len(self.dataset))
                    idx = torch.randint(0, len(self.dataset), (1,)).item()
                    added = idx not in self.total_index
                    
                    if added:
                        self.total_index.add(idx)
                        batch_indices_set.add(idx)
                        break 
                    if len(self.total_index) == len(self.dataset):
                        break
                if len(self.total_index) == len(self.dataset):
                        break
            i = i + 1
            print(len(self.total_index))
            print(len(batch_indices_set))
            yield list(batch_indices_set) """


""" if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
                    prog='train script',
                    description='train the multitask NN on PAR task')
    
    parser.add_argument('--reduced', action='store_true', help='train with a reduced size dataset')
    opt = parser.parse_args()
    
    if (opt.reduced == True):
        print("It will be used a reduced dataset")
    # Load the dataset here and return the data and labels as a list of tuples
        train_annotation_file = "par_dataset\\training_set_reduced.txt"
        validation_annotation_file = "par_dataset\\validation_set_reduced.txt"
        train_img_dir = "par_dataset\\training_set_reduced\\"
        validation_img_dir = "par_dataset\\validation_set_reduced\\"
    else:
        print("It will be used the entire dataset")
        train_annotation_file = "par_dataset\\training_set.txt"
        validation_annotation_file = "par_dataset\\validation_set.txt"
        train_img_dir = "par_dataset\\training_set\\"
        validation_img_dir = "par_dataset\\validation_set\\"

    data_transforms_val = transforms.Compose([transforms.Resize((96,288)), transforms.ToTensor()])
    data_trasfporms_train = transforms.Compose([transforms.Resize((96, 288)), transforms.ToTensor()])
    # Load the dataset if do you want to see the output of the dataset saved  you casn use verbose=True
    train_dataset = CustomImageDataset(train_annotation_file, train_img_dir, data_trasfporms_train)
    val_dataset = CustomImageDataset(validation_annotation_file, validation_img_dir, data_transforms_val)
    
    batch_size = 64
 """
class CustomSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_task = dataset.get_num_task()
        self.task_iterators = {}
        self.total_index = []
        # crea total_index come una lista contenenete tutti gli indici del dataset
        
        #print("task_iterators: ", len(self.task_iterators))

    def do_iteratoors(self):
        task_iterators = {}
        for task_idx in range(self.num_task):
            task_iterators[task_idx] = []
            

            if task_idx != 0 and task_idx !=  1:
                task_list = self.dataset.get_task_iterator(task_idx)
                task_list = random.sample(task_list, len(task_list))
                task_iterators[task_idx].append(iter(task_list))
            else:
                #for da 1 a 11
                task_Nclass = self.dataset.get_task_iterator(task_idx)
                for i in range(1, 12):
                    task_Nclass[i] = random.sample(task_Nclass[i], len(task_Nclass[i]))
                    task_iterators[task_idx].append(iter(task_Nclass[i]))
        return task_iterators
    
    

    def __iter__(self):
        self.total_index = list(range(len(self.dataset)))
        self.task_iterators = self.do_iteratoors()
        # itera finche non ho finito total_index
        while len(self.total_index) > 0:
            batch_indices_set = set()
            # itera per ogni task
            for task_idx in range(self.num_task):
                while True:
                    if task_idx != 0 and task_idx !=  1:
                        try:
                            # Cerca di ottenere l'indice successivo dall'iteratore dell'attività corrente
                            idx = next(self.task_iterators[task_idx][0])
                            # verifica se l'indice è già stato aggiunto al batch
                            added = idx not in batch_indices_set
                            # Aggiunge l'indice all'insieme
                            if added:
                                batch_indices_set.add(idx)
                                # se l'indice è presente in total_index, lo rimuove
                                if idx in self.total_index:
                                    self.total_index.remove(idx)
                                # se l'aggiunta ha avuto successo, esci dal ciclo while
                                break
                        except StopIteration:
                            # se l'iteratore è finito, ricrea l'iteratore
                            task_list = self.dataset.get_task_iterator(task_idx)
                            task_list = random.sample(task_list, len(task_list))
                            self.task_iterators[task_idx][0] = iter(task_list)
                            continue 
                    else:
                        j = 1
                        while j < 12:
                            #print(j)
                            try:
                                idx = next(self.task_iterators[task_idx][j-1])
                                added = idx not in batch_indices_set
                                if added:
                                    #print("aggiunto")
                                    batch_indices_set.add(idx)
                                    if idx in self.total_index:
                                        self.total_index.remove(idx)
                                    j = j + 1
                            except StopIteration:
                                task_list = self.dataset.get_task_iterator(task_idx)[j]
                                task_list = random.sample(task_list, len(task_list))
                                self.task_iterators[task_idx][j-1] = iter(task_list)
                                continue
                        break 

            #Aggiungi altri indici randomici
            num_remain_elem =  self.batch_size - self.num_task - 10 - 10
            for _ in range (num_remain_elem):
                if len(self.total_index) > 0:
                    # estari a caso un elemento da total_index e aggiungilo al batch
                    idx = random.choice(self.total_index)
                    batch_indices_set.add(idx)
                    # rimuovi l'indice da total_index
                    self.total_index.remove(idx)
                else:
                    break
            #print(len(list(batch_indices_set)))
            #print(len(self.total_index))
            yield list(batch_indices_set)           
                    

        

    def __len__(self):
        return len(self.dataset) // self.batch_size

    

""" if __name__ == '__main__':
    # Creazione del DataLoader con il CustomSampler
    batch_size = 64
    custom_sampler = CustomSampler(train_dataset, batch_size)
    train_dataloader = DataLoader(train_dataset,  sampler=custom_sampler, shuffle=False) 
    
    i = 0
    for batch_indices in train_dataloader:
        
        # batch_indices è una lista di indici per il batch corrente
        # squeeze() rimuove le dimensioni unitarie dal tensore
        batch_indices[0] = batch_indices[0].squeeze()
        batch_indices[1] = [l.squeeze() for l in batch_indices[1]]
        print(batch_indices) """
    

