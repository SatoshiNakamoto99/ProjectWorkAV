import os
import sys

from ultralytics.utils import LOGGER
LOGGER.setLevel("WARNING")  # Puoi impostare anche su "ERROR" o "CRITICAL" per stampare solo gli avvisi critici


current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, ".."))
sys.path.append(project_path)

from trackers.test_track import run
from pathlib import Path

from my_yolo import MyYOLO

import torch
from app.settings import VID_FORMATS

from ultralytics.utils.plotting import save_one_box
from trackers.utils.utils import write_mot_results

from tqdm import tqdm


@torch.no_grad()
def run(options):
    yolo_model = options.get('yolo-model', 'models/yolov8n')
    tracking_method = options.get('tracking-method', 'botsort')
    source = options.get('source', '0')
    imgsz = options.get('imgsz', [640])
    conf = options.get('conf', 0.5)
    iou = options.get('iou', 0.7)
    device = options.get('device', '')
    classes = options.get('classes', [0])
    save_id_crops = options.get('save-id-crops', False)
    save_mot = options.get('save-mot', False)

    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    yolo = MyYOLO(
        yolo_model if 'yolov8' in str(yolo_model) else 'yolov8n.pt',
    )

    subfolder = source
    source = source + '/img1'

    # Ottieni una lista di tutti i file nella cartella
    images = os.listdir(source)

    for img in tqdm(images,desc='       Processing ' + Path(subfolder).name):    
        frame = source + '/' + img
        results = yolo.track(
            source=frame,
            conf=conf,
            iou=iou,
            stream=True,
            device=device,
            classes=classes,
            imgsz=imgsz,
        )

        # store custom args in predictor
        yolo.predictor.custom_args = options

        for frame_idx, r in enumerate(results):

            if r.boxes.data.shape[1] == 7:

                if 'MOT16' or 'MOT17' or 'MOT20' in source:
                    p = Path('data/trackers/mot_challenge/MOT17-train/' + yolo.predictor.args.tracker.split('.')[0] + '/' +  (Path(subfolder).name + '.txt'))
                    yolo.predictor.mot_txt_path = p

                if save_mot:
                    write_mot_results(
                        yolo.predictor.mot_txt_path,
                        r,
                        frame_idx,
                    )

                if save_id_crops:
                    for d in r.boxes:
                        #print('args.save_id_crops', d.data)
                        save_one_box(
                            d.xyxy,
                            r.orig_img.copy(),
                            file=(
                                yolo.predictor.save_dir / 'crops' /
                                str(int(d.cls.cpu().numpy().item())) /
                                str(int(d.id.cpu().numpy().item())) / f'{frame_idx}.jpg'
                            ),
                            BGR=True
                        )

    if save_mot:
        print(f'MOT results saved to {yolo.predictor.mot_txt_path}')
    


if __name__ == "__main__":

    # Parametri fissi
    fixed_params = {
        'yolo-model': 'models/yolov8n',
        'tracking-method': 'botsort',
        'imgsz': [640],
        'conf': 0.5,
        'iou': 0.7,
        'device': '',
        'classes': [0],
        'save-id-crops': True,
        'save-mot': True
    }

    # Percorso della cartella contenente le sottocartelle
    base_folder = 'datasets/MOT17/train'

    # Ottenere tutte le cartelle nella cartella principale
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]

    for source in tqdm(subfolders, desc="Processing MOT17 dataset"):
        # Aggiungi source al dizionario
        custom_args = fixed_params.copy()
        custom_args['source'] = source

        # Chiamata alla funzione di tracking con parametri fissi e source variabile
        run(custom_args)