from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt

from my_yolo import MyYOLO
from ultralytics import YOLO

from concurrent.futures import ThreadPoolExecutor

#######################################################################################


from math import ceil
from tracking.loaders import LoadVideoStream
import cv2
# Set the path to your video file
video_path = 'data/video_prisco_tagliato.mp4' 
# Load the YOLOv8 model
model = MyYOLO('models/yolov8n.pt')

# Create a video stream loader
stream_loader = LoadVideoStream(source=video_path, fps_out=1)

# Store the track history
track_history = defaultdict(lambda: [])

# Create a figure with one row and two columns
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Function to perform inference and tracking on a frame
def infer_and_track(frame):
    # Perform inference
    detection_results = model.predict(frame, conf=0.5, classes=[0], device='cpu')

    # Run tracking on the frame, persisting tracks between frames
    tracking_results = model.track(frame, conf=0.5, persist=True, classes=[0], device='cpu', tracker="bytetrack.yaml")

    return frame, detection_results, tracking_results

try:
    with ThreadPoolExecutor(max_workers=2) as executor:
        for source, images, _, _ in stream_loader:
            frame = images[0]

            # Submit the inference and tracking tasks to the thread pool
            future = executor.submit(infer_and_track, frame)

            # Wait for the tasks to complete
            frame, detection_results, tracking_results = future.result()

            # Visualize the results on the frame
            detection_annotated_frame = detection_results[0].plot()

            print('---------------------------')
            print(tracking_results[0].boxes.conf.unsqueeze(1).to('cpu'))
            print('\n\n\n')


            # Display the annotated frame
            axs[0].imshow(cv2.cvtColor(detection_annotated_frame, cv2.COLOR_BGR2RGB))
            axs[0].set_title('YOLOv8 Inference')
            axs[0].axis('off')

            # Get the boxes and track IDs
            boxes = tracking_results[0].boxes.xywh.cpu()
            track_ids = tracking_results[0].boxes.id.int().cpu().tolist()

            # Visualize the tracking_results on the frame
            tracking_annotated_frame = tracking_results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(tracking_annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Display the annotated frame for tracking
            axs[1].imshow(cv2.cvtColor(tracking_annotated_frame, cv2.COLOR_BGR2RGB))
            axs[1].set_title('YOLOv8 Tracking')
            axs[1].axis('off')

            # Adjust spacing between subplots
            plt.subplots_adjust(wspace=0.4)

            # Pause to display the frames
            plt.pause(0.01)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

except KeyboardInterrupt:
    print("Keyboard interrupt. Stopping the stream.")
finally:
    stream_loader.close()
    cv2.destroyAllWindows()


#######################################################################################

# # Load the YOLOv8 model
# model = MyYOLO('models/yolov8n.pt')
# # Open the video file
# video_path = "data/video_prisco_tagliato.mp4"
# cap = cv2.VideoCapture(video_path)

# # Store the track history
# track_history = defaultdict(lambda: [])

# # Create a figure with one row and two columns
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if success:

#         detection_results = model.predict(frame, conf=0.5, classes=[0], device='cpu')

#         # Visualize the results on the frame
#         detection_annotated_frame = detection_results[0].plot()

#         # Display the annotated frame
#         axs[0].imshow(cv2.cvtColor(detection_annotated_frame, cv2.COLOR_BGR2RGB))
#         axs[0].set_title('YOLOv8 Inference')
#         axs[0].axis('off')

#         # Run YOLOv8 tracking on the frame, persisting tracks between frames
#         tracking_results = model.track(frame, conf=0.5, persist=True, classes=[0], device='cpu', tracker="bytetrack.yaml")

#         # Get the boxes and track IDs
#         boxes = tracking_results[0].boxes.xywh.cpu()
#         track_ids = tracking_results[0].boxes.id.int().cpu().tolist()

#         # Visualize the tracking_results on the frame
#         tracking_annotated_frame = tracking_results[0].plot()

#         # Plot the tracks
#         for box, track_id in zip(boxes, track_ids):
#             x, y, w, h = box
#             track = track_history[track_id]
#             track.append((float(x), float(y)))  # x, y center point
#             if len(track) > 30:  # retain 90 tracks for 90 frames
#                 track.pop(0)

#             # Draw the tracking lines
#             points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
#             cv2.polylines(tracking_annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

#         # Display the annotated frame for tracking
#         axs[1].imshow(cv2.cvtColor(tracking_annotated_frame, cv2.COLOR_BGR2RGB))
#         axs[1].set_title('YOLOv8 Tracking')
#         axs[1].axis('off')

#         # Adjust spacing between subplots
#         plt.subplots_adjust(wspace=0.4)

#         # Pause to display the frames
#         plt.pause(0.01)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()