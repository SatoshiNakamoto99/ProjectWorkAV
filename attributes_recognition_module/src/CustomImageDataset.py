from itertools import cycle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, annotation_file, img_dir, data_transforms, verbose=False, num_task=5):
        self.annotation_file = annotation_file
        self.img_dir = img_dir
        self.data_transforms = data_transforms
        self.num_task = num_task

        # Load and process your dataset here
        self.data, self.task_iterators = self.load_dataset(verbose=verbose)

    def load_dataset(self, verbose):
        # Load the dataset here and return the data and labels as a list of tuples
        data = []
        task_iterators = {}
        with open(self.annotation_file, "r") as f:
            for line in f:
                img_path, label_str = line.split(",",1)
                label = [int(l) for l in label_str.split(",")]
                # verifica che img_path esiste in img_dir
                # se non esiste, non aggiungere a data
                if os.path.isfile(f"{self.img_dir}\\{img_path}"):
                    data.append((img_path, label))  # append di immagine + label
                    # Store iterators for each task
                    
                    # Memorizza gli iteratori per ogni compito
                    for task_idx, task_label in enumerate(label):
                        if task_label != -1:
                            if task_idx != 0 and task_idx != 1:
                                
                                    if task_idx not in task_iterators:
                                        task_iterators[task_idx] = []
                                    task_iterators[task_idx].append(len(data) - 1)

                                # Se il compito è il primo o il secondo, memorizza anche per colore
                            else:
                                    if task_idx not in task_iterators:
                                        task_iterators[task_idx] = {}
                                    if task_label not in task_iterators[task_idx]:
                                        task_iterators[task_idx][task_label] = []
                                    task_iterators[task_idx][task_label].append(len(data) - 1)
            
                else:
                    print(f"Image {img_path} not found in {self.img_dir}. Skipping...")
            # fai lo shuffle di ogni lista di task_iterators (per ogni task)
            """ for task_idx in task_iterators:
                if task_idx != 0 and task_idx != 1:
                    task_iterators[task_idx] = cycle(task_iterators[task_idx])
                else:
                    for color in task_iterators[task_idx]:
                        task_iterators[task_idx][color] = cycle(task_iterators[task_idx][color])    """       

        if verbose:
            print(f"Loaded {len(data)} samples")
            print("Loaded dataset successfully:")
            for img_path, label in data:
                print(f"Image: {img_path}, Label: {label}")
        return data, task_iterators


    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            # Se idx è una lista, restituisci una lista di campioni
            # voglio che restituisci una tensore di immagini e n tensori di label (una per ogni task) tensor([ 0,  0, -1,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0, -1,  1,  0,  0,
            # 1,  0,  0,  0,  0,  0,  0, -1,  0, -1, -1,  0,  0,  0, -1,  0,  0,  0,
            # 0,  0,  0,  0,  0,  1,  0, -1,  1, -1,  0,  0,  0,  0,  1, -1,  0,  0,
            # 0,  1,  0, -1,  0,  0,  1,  0, -1,  0])]] la cui size è pari alla dimensione di idx
            # imgs è un tesore di immagini
            # labels è una lista di tensori di label
            # labels[0] è un tensore di label per il primo task
            # labels[1] è un tensore di label per il secondo task
            # ...
            # labels[n] è un tensore di label per l'n-esimo task
            imgs = []
            labels = [[] for _ in range(self.num_task)]
            for i in idx:
                img, label = self.load_sample(i)
                imgs.append(img)
                for task_idx, task_label in enumerate(label):
                    labels[task_idx].append(task_label)
            
            return torch.stack(imgs), [torch.tensor(l) for l in labels]
            
        else:
            # Altrimenti, restituisci un singolo campione
            return self.load_sample(idx)

    def load_sample(self, idx):
        img_path, label = self.data[idx]

        # Load image
        img = Image.open(f"{self.img_dir}\\{img_path}").convert("RGB")

        # Apply transforms
        img = self.data_transforms(img)
        return img, label


    
    def get_task_iterator(self, task_idx):
        # Return an iterator for a specific task
        # Return a list of indices for a specific task
        return self.task_iterators.get(task_idx, [])
    
    def get_num_task(self):
        return self.num_task
# if __name__ == '__main__':
#     # Example usage used for testin It's not mandatory to use this but it's recommended for testing
#     annotation_file = "C:\\VSCode_Project\\ArtificialVision\\Code\\PAR_Project_ForAV\\progetto\\par_dataset\\training_set.txt"
#     img_dir = "C:\\VSCode_Project\\ArtificialVision\\Code\\PAR_Project_ForAV\\progetto\\par_dataset\\training_set\\"
#     data_transforms_val = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
#     data_trasfporms_train = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
#     # Load the dataset if do you want to see the output of the dataset saved  you casn use verbose=True
#     train_dataset = CustomImageDataset(annotation_file, img_dir, data_trasfporms_train, verbose=True)
#     #val_dataset = CustomImageDataset(annotation_file, img_dir, data_transforms_val)

#     train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
#     #val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)

#     for batch in train_dataloader:
#         images, labels = batch
#         print(f"Batch loaded successfully. Number of images: {len(images)}, Number of labels: {len(labels)}")
#         break  # Stop after the first batch for brevity
