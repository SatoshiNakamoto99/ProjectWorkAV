import sys
import os
import subprocess
from pathlib import Path

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, ".."))
sys.path.append(project_path)

def main():
    fixed_params = [
        "python", "trackers/save_mot_challenge_results.py",
        "--yolo-model", "models/yolov8n",
        "--tracking-method", "bytetrack",
        "--source", "data/video",
        "--imgsz", "640",  # Replace with your specific imgsz values
        "--conf", "0.5",
        "--iou", "0.7",
        "--device", "cpu",  # Replace with your specific CUDA device
        # "--show",
        # "--save",
        "--classes", "0",  # Replace with your specific class values
        # "--project", "runs/track",
        # "--name", "exp",
        # "--exist-ok",
        # "--half",
        # "--vid-stride", "1",
        # "--show-labels",
        # "--show-conf",
        # "--save-txt",
        "--save-id-crops",
        "--save-mot",
        # "--per-class",
        # "--verbose",
        # "--vid_stride", "1",
    ]

    # Percorso della cartella contenente le sottocartelle
    base_folder = 'datasets/MOT17/train'
    trackers = ['botsort','byterack']

    # Ottenere tutte le cartelle nella cartella principale
    subfolders = [base_folder + '/' + f.name for f in os.scandir(base_folder) if f.is_dir()]
    

    for tracker in trackers:
        custom_args = fixed_params.copy()
        custom_args[fixed_params.index('--tracking-method')+1] = tracker
        print('------------------------ ' + tracker.upper() +  ' ------------------------------')

        for source in subfolders:
            print('Processing ' + Path(source).name + ' with ' + tracker)
            custom_args[fixed_params.index('--source')+1] = str(source)

            # Esegui lo script principale con gli argomenti personalizzati utilizzando subprocess
            subprocess.run(custom_args)

if __name__ == "__main__":
    main()
