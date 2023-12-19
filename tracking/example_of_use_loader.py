
from math import ceil
from loaders import LoadVideoStream
import cv2
import os
import sys

from ultralytics.utils import LOGGER
LOGGER.setLevel("WARNING")  # Puoi impostare anche su "ERROR" o "CRITICAL" per stampare solo gli avvisi critici

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, ".."))
sys.path.append(project_path)
from my_yolo import MyYOLO
import time

# Set the path to your video file
video_path = 'data/video_prisco_tagliato.mp4' 
model = MyYOLO('models/yolov8n.pt')

# Create a video stream loader
stream_loader = LoadVideoStream(source=video_path, fps_out=3)

try:
    
    frame_count = 0
    # Iterate through the frames in the video stream
    start_time = time.time()
    for sources, images, _, _ in stream_loader:
        # Process the frames as needed
        # print(f"Received frames {frame_count} from source: {sources}")
        # cv2.imshow('frame',images)
        tracking_results = model.track(images, conf=0.5, persist=True, classes=[0], device='cpu', tracker="bytetrack.yaml")
        # Add your processing logic here
        frame_count += 1
        # Update last_frame with the latest received frame
        #last_frame = images[0]
    end_time = time.time()

except KeyboardInterrupt:
    # Handle keyboard interrupt (e.g., press Ctrl+C to stop the loop)
    print("Keyboard interrupt. Stopping the stream.")

finally:
    stream_loader.close()
    print("Processing time: ", end_time-start_time)