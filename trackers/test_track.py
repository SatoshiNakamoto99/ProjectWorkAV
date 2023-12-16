import argparse
import sys
import os

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, ".."))
sys.path.append(project_path)
from my_yolo import MyYOLO

import torch
from pathlib import Path
from app.settings import VID_FORMATS

from ultralytics.utils.plotting import save_one_box
from trackers.utils.utils import write_mot_results

@torch.no_grad()
def run(args):
    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    yolo = MyYOLO(
        args.yolo_model if 'yolov8' in str(args.yolo_model) else 'yolov8n.pt',
    )
    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        stream=True,
        device=args.device,
        classes=args.classes,
        imgsz=args.imgsz,
    )

    # store custom args in predictor
    yolo.predictor.custom_args = args

    for frame_idx, r in enumerate(results):

        if r.boxes.data.shape[1] == 7:

            if yolo.predictor.source_type.webcam or args.source.endswith(VID_FORMATS):
                p = yolo.predictor.save_dir / 'mot' / (args.source + '.txt')
                yolo.predictor.mot_txt_path = p
            elif 'MOT16' or 'MOT17' or 'MOT20' in args.source:
                p = yolo.predictor.save_dir / 'mot' / (Path(args.source).parent.name + '.txt')
                yolo.predictor.mot_txt_path = p

            if args.save_mot:
                write_mot_results(
                    yolo.predictor.mot_txt_path,
                    r,
                    frame_idx,
                )

            if args.save_id_crops:
                for d in r.boxes:
                    print('args.save_id_crops', d.data)
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

    if args.save_mot:
        print(f'MOT results saved to {yolo.predictor.mot_txt_path}')
    

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default='models/yolov8n',
                        help='yolo model path')
    parser.add_argument('--tracking-method', type=str, default='botsort',
                        help='botsort, bytetrack, deepocsort')
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
    parser.add_argument('--classes', nargs='+', type=int, default=[0],
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--save-mot', action='store_true',
                        help='save tracking results in a single txt file')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)




# def run(args):

#     # Open the video file
#     cap = cv2.VideoCapture(args.source)

#     # Store the track history
#     track_history = defaultdict(lambda: [])

#     # Loop through the video frames
#     while cap.isOpened():
#         # Read a frame from the video
#         success, frame = cap.read()

#         if success:
#             # Run YOLOv8 tracking on the frame, persisting tracks between frames
#             model = YOLO(
#                 args.yolo_model if 'yolov8' in str(args.yolo_model) else 'yolov8n.pt',
#             )
#             results = model.track(
#                 source=frame,
#                 conf=args.conf,
#                 iou=args.iou,
#                 device=args.device,
#                 classes=args.classes,
#                 imgsz=args.imgsz,
#             )

#             # Get the boxes and track IDs
#             boxes = results[0].boxes.xywh.cpu()
#             track_ids = results[0].boxes.id.int().cpu().tolist()

#             # Visualize the results on the frame
#             annotated_frame = results[0].plot()

#             # Plot the tracks
#             for box, track_id in zip(boxes, track_ids):
#                 x, y, w, h = box
#                 track = track_history[track_id]
#                 track.append((float(x), float(y)))  # x, y center point
#                 if len(track) > 30:  # retain 90 tracks for 90 frames
#                     track.pop(0)

#                 # Draw the tracking lines
#                 points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
#                 cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

#                 # Draw the bounding box in a different color (red)
#                 color = (0, 255, 0)
#                 cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, 3)

#             # Display the annotated frame
#             cv2.imshow("YOLOv8 Tracking", annotated_frame)

#             # Break the loop if 'q' is pressed
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break
#         else:
#             # Break the loop if the end of the video is reached
#             break

#     # Release the video capture object and close the display window
#     cap.release()
#     cv2.destroyAllWindows()