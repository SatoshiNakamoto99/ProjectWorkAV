from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt

from my_yolo import MyYOLO
from ultralytics import YOLO

from math import ceil
from tracking.loaders import LoadVideoStream
import cv2
# Set the path to your video file
video_path = 'data/video_prisco_tagliato.mp4' 
# Load the YOLOv8 model
detection_model = MyYOLO('models/yolov8n.pt')
tracking_model = MyYOLO('models/yolov8n.pt')

# Create a video stream loader
stream_loader = LoadVideoStream(source=video_path, fps_out=3)

# Store the track history
track_history = defaultdict(lambda: [])

# Create two OpenCV windows for displaying the annotated frames
cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
cv2.namedWindow("YOLOv8 Tracking", cv2.WINDOW_NORMAL)

# Specifica le dimensioni desiderate per le finestre
window_width, window_height = 700, 500  # Puoi regolare queste dimensioni come preferisci

# Imposta le dimensioni delle finestre
cv2.resizeWindow("YOLOv8 Inference", window_width, window_height)
cv2.resizeWindow("YOLOv8 Tracking", window_width, window_height)

# Move the windows to specific positions on the screen (adjust coordinates as needed)
cv2.moveWindow("YOLOv8 Inference", 50, 100)
cv2.moveWindow("YOLOv8 Tracking", 750, 100)

try:
    for source, images, _, _ in stream_loader:
        frame = images[0]

        detection_results = detection_model.predict(frame, conf=0.5, classes=[0], device='cpu')
        tracking_results = tracking_model.track(frame, conf=0.5, persist=True, classes=[0], device='cpu', tracker="config/bytetrack.yaml")

        # Visualize the results on the frame
        detection_annotated_frame = detection_results[0].plot()
        tracking_annotated_frame = tracking_results[0].plot()


        # Get the boxes and track IDs
        boxes = tracking_results[0].boxes.xywh.cpu()
        track_ids = tracking_results[0].boxes.id.int().cpu().tolist()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(tracking_annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)


        # Display the annotated frame 
        cv2.imshow("YOLOv8 Inference", detection_annotated_frame)
        cv2.imshow("YOLOv8 Tracking", tracking_annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Keyboard interrupt. Stopping the stream.")
finally:
    stream_loader.close()
    cv2.destroyAllWindows()