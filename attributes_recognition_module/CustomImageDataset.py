import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, annotation_file, img_dir, data_transforms, verbose=False):
        self.annotation_file = annotation_file
        self.img_dir = img_dir
        self.data_transforms = data_transforms

        # Load and process your dataset here
        self.data = self.load_dataset(verbose=verbose)

    def load_dataset(self, verbose):
        # Load the dataset here and return the data and labels as a list of tuples
        data = []
        with open(self.annotation_file, "r") as f:
            for line in f:
                img_path, label_str = line.split(",",1)
                label = [int(l) for l in label_str.split(",")]
                # verifica che img_path esiste in img_dir
                # se non esiste, non aggiungere a data
                if os.path.isfile(f"{self.img_dir}\\{img_path}"):
                    data.append((img_path, label))  # append di immagine + label
                else:
                    print(f"Image {img_path} not found in {self.img_dir}. Skipping...")
                        

        if verbose:
            print(f"Loaded {len(data)} samples")
            print("Loaded dataset successfully:")
            for img_path, label in data:
                print(f"Image: {img_path}, Label: {label}")
        return data


    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Return the data sample at the given index

        img_path, label = self.data[idx]

        # Load image
        img = Image.open(f"{self.img_dir}\\{img_path}").convert("RGB")

        # Apply transforms
        img = self.data_transforms(img)

        return img, label

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
