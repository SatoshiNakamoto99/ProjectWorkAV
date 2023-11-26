
import yaml

import shutil
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

def create_cross_validation_splits(dataset_path, yaml_file, ksplit=5, random_state=20):
    save_path = Path(dataset_path / f'K={ksplit}_rnd_state={random_state}-Fold_Cross-val')
    if not save_path.exists():
        labels = sorted(dataset_path.rglob("*labels/*.txt"))

        with open(yaml_file, 'r', encoding="utf8") as y:
            classes = yaml.safe_load(y)['names']
        cls_idx = sorted(classes)

        indx = [l.stem for l in labels]
        labels_df = pd.DataFrame([], columns=cls_idx, index=indx)

        for label in labels:
            lbl_counter = Counter()

            with open(label, 'r') as lf:
                lines = lf.readlines()

            for l in lines:
                lbl_counter[int(l.split(' ')[0])] += 1

            labels_df.loc[label.stem] = lbl_counter

        labels_df = labels_df.fillna(0.0)

        kf = KFold(n_splits=ksplit, shuffle=True, random_state=random_state)
        kfolds = list(kf.split(labels_df))

        folds = [f'split_{n}' for n in range(1, ksplit + 1)]
        folds_df = pd.DataFrame(index=indx, columns=folds)

        for idx, (train, val) in enumerate(kfolds, start=1):
            folds_df[f'split_{idx}'].loc[labels_df.iloc[train].index] = 'train'
            folds_df[f'split_{idx}'].loc[labels_df.iloc[val].index] = 'val'

        fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

        for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
            train_totals = labels_df.iloc[train_indices].sum()
            val_totals = labels_df.iloc[val_indices].sum()

            ratio = val_totals / (train_totals + 1E-7)
            fold_lbl_distrb.loc[f'split_{n}'] = ratio

        supported_extensions = ['.jpg', '.jpeg', '.png']
        images = []

        for ext in supported_extensions:
            images.extend(sorted(dataset_path.rglob(f"*images/*{ext}")))
            #.rglob("*labels/*.txt")

        
        #save_path.mkdir(parents=True, exist_ok=True)
        
        
        print(f"Creating directory {save_path}")
        save_path.mkdir(parents=True, exist_ok=True)
        
        ds_yamls = []

        for split in folds_df.columns:
            split_dir = save_path / split
            split_dir.mkdir(parents=True, exist_ok=True)
            (split_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
            (split_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
            (split_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
            (split_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)

            dataset_yaml = split_dir / f'{split}_dataset.yaml'
            ds_yamls.append(dataset_yaml)

            with open(dataset_yaml, 'w') as ds_y:
                yaml.safe_dump({
                    'path': split_dir.as_posix(),
                    'train': 'train',
                    'val': 'val',
                    'names': list(classes)  # Utilizza cls_idx al posto di classes
                }, ds_y)
        
        print("Copying images and labels to new directory...")
        for image, label in tqdm(zip(images, labels)):
            for split, k_split in folds_df.loc[image.stem].items():
                img_to_path = save_path / split / k_split / 'images'
                lbl_to_path = save_path / split / k_split / 'labels'
                
                # Se il percorso non esiste, esegui la copia
                shutil.copy(image, img_to_path / image.name)
                shutil.copy(label, lbl_to_path / label.name)
        folds_df.to_csv(save_path / "kfold_datasplit.csv", index_label="filename")
        fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv", index_label="split")
        print("Done!")
        return ds_yamls
    else:
        print(f"Dataset already exists at {save_path}")
        return [ds_yaml for ds_yaml in save_path.rglob("*dataset.yaml")]

