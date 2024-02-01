#!/bin/bash

python_executable="/usr/bin/python3" 

# Percorso della cartella contenente le sottocartelle
base_folder='datasets/MOT17/train'

# Lista di cartelle di origine e metodi di tracciamento
source_folders=( "$base_folder"/* )
tracking_methods=('botsort' 'bytetrack')

# Altri parametri fissi
fixed_params=(
    "--yolo-model" "models/best_Kfod5_100epoc_base.pt"
    "--imgsz" "640"  
    "--conf" "0.5"
    "--iou" "0.7"
    "--device" "cpu"  
    # "--show",
    # "--save",
    "--classes" "0"  
    # "--project" "runs/track",
    # "--name" "exp",
    # "--exist-ok",
    # "--half",
    # "--vid-stride" "1",
    # "--show-labels",
    # "--show-conf",
    # "--save-txt",
    #"--save-id-crops"
    "--save-mot"
    # "--per-class",
    # "--verbose",
    # "--vid_stride" "1",
)

# Ciclo esterno per iterare attraverso tutti i tracking_method
for tracking_method in "${tracking_methods[@]}"; do
    echo "---------------------------------- ${tracking_method^^} ----------------------------------"
    # Ciclo interno per iterare attraverso tutte le source_folder
    for source_folder in "${source_folders[@]}"; do
        # Esegui lo script principale con gli argomenti specificati
        "$python_executable" trackers/save_mot_challenge_results.py \
            --source "$source_folder" \
            --tracking-method "$tracking_method" \
            "${fixed_params[@]}"
    done
done