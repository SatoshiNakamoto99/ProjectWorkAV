import torch

import os
import sys

from ultralytics.utils import LOGGER
LOGGER.setLevel("WARNING")  # Puoi impostare anche su "ERROR" o "CRITICAL" per stampare solo gli avvisi critici
import argparse

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, ".."))
sys.path.append(project_path)

from pathlib import Path

from my_yolo import MyYOLO
from app.settings import CONFIG


#from ultralytics.utils.plotting import save_one_box
from trackers.utils.utils import write_mot_results

from tqdm import tqdm


@torch.no_grad()
def run(args):
    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    yolo = MyYOLO(
        args.yolo_model if 'yolov8' in str(args.yolo_model) else 'yolov8n.pt',
    )

    subfolder = args.source
    args.source = args.source + '/img1'

    # Ottieni una lista di tutti i file nella cartella
    images = os.listdir(args.source)
    frame_idx=0

    for img in tqdm(images,desc='Processing ' + Path(subfolder).name + ' with tracker ' + args.tracking_method):    
        
        frame = args.source + '/' + img
        results = yolo.track(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            show=args.show,
            stream=True,
            device=args.device,
            show_conf=args.show_conf,
            save_txt=args.save_txt,
            show_labels=args.show_labels,       
            save=args.save,
            verbose=args.verbose,
            exist_ok=args.exist_ok,
            project=args.project,
            name=args.name,
            classes=args.classes,
            imgsz=args.imgsz,
            vid_stride=args.vid_stride,
            line_width=args.line_width,
            tracker = CONFIG / (str(args.tracking_method) + '.yaml'),
            persist=True
        )

        # store custom args in predictor
        #yolo.predictor.custom_args = args

        for key, value in args.__dict__.items():
            if hasattr(yolo.predictor.args,key):
                setattr(yolo.predictor.args,key,value)

        for _, r in enumerate(results):

            if r.boxes.data.shape[1] == 7:

                if 'MOT16' or 'MOT17' or 'MOT20' in args.source:
                    p = Path('data/trackers/mot_challenge/MOT17-train/' + args.tracking_method + '/data/' +  (Path(subfolder).name + '.txt'))
                    yolo.predictor.mot_txt_path = p

                if args.save_mot:
                    write_mot_results(
                        yolo.predictor.mot_txt_path,
                        r,
                        frame_idx,
                    )

                # if args.save_id_crops:
                #     for d in r.boxes:
                #         save_one_box(
                #             d.xyxy,
                #             r.orig_img.copy(),
                #             file=(
                #                 yolo.predictor.save_dir / 'crops' /
                #                 str(int(d.cls.cpu().numpy().item())) /
                #                 str(int(d.id.cpu().numpy().item())) / f'{frame_idx}.jpg'
                #             ),
                #             BGR=True
                #         )
        frame_idx = frame_idx+1

    if args.save_mot:
        print(f'MOT results saved to {yolo.predictor.mot_txt_path}')
    


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default='models/yolov8n',
                        help='yolo model path')
    # parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
    #                     help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='botsort',
                        help='botsort, bytetrack')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default= 'runs/track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--save-mot', action='store_true',
                        help='save tracking results in a single txt file')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--vid_stride', default=1, type=int,
                        help='video frame-rate stride')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)