import os
import shutil


def find(name, path, train = True):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
        else:
            if train == True:
                print("file not found in train")
            else:
                print("file not found in validation")
                
def create_reduced_image_path(image_path):
    folder_name_split = image_path.split("/")
    folder_name = folder_name_split[0]
    file_name = folder_name_split[1]
    folder_reduced_name = folder_name + "_reduced"
    
    image_reduced_path = os.path.join(folder_reduced_name, file_name)
    
    return image_reduced_path
    
                

def reduce_dataset(image_dir, label_dir, max_samples = 10, train = True):
    
    image_reduced_dir = image_dir.split("/")[0]
    image_reduced_dir = image_reduced_dir + "_reduced/"
    
    label_dir_no_extension = label_dir.split(".")[0]
    label_dir_reduced = label_dir_no_extension + "_reduced" + ".txt"
    
    
    label_file = open(label_dir, 'r')
    label_file_reduced = open(label_dir_reduced, 'w')
    
    
    os.mkdir(image_reduced_dir)
    
    for i in range(max_samples):
        line = label_file.readline()
        image_name = line.split(",")[0]
        image_path = find(image_name, image_dir, train = train)
        
        if image_path is None:
            print(f"{image_path} not found. Skipping ...")
            continue
        # move image_path and line
        # print(f"{image_path}\n{line}")
        
        image_reduced_path = create_reduced_image_path(image_path)
        
        shutil.copy(image_path, image_reduced_path) # copia l'immagine trovata da image_dir a image_dir_reduced
        label_file_reduced.write(line)
        
    label_file.close()
    label_file_reduced.close()


if __name__ == '__main__':
    ''' nota: questo script deve essere inserito nella cartella par_dataset/ per operare.
        Dati in input il training e il validation set, genera in base alla scelta dell'utente
        un dataset sottinsieme del principale.'''
    
    train_dir = "training_set/"
    train_label_dir = "training_set.txt"
    val_dir = "validation_set/"
    val_label_dir = "validation_set.txt"
    
    reduce_dataset(train_dir, train_label_dir, max_samples = 10, train = True)
    reduce_dataset(val_dir, val_label_dir, max_samples = 10, train = False)
    

