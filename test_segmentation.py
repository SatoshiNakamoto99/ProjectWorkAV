from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

m = YOLO('models/yolov8n-seg.pt')

img = 'data/test_image.jpg'
video = 'data/video_prisco_tagliato.mp4'

source_path = video
source_name = Path(source_path).name.split('.')[0]

frame_id = 0
track_history = defaultdict(lambda: {'max_height': 0, 'max_height_frame': None, 'track': []})

cap = cv2.VideoCapture(source_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        res = m.track(frame, conf=0.3, persist=True, classes=0)

        # iterate detection results
        for r in res:
            img = np.copy(r.orig_img)

            # iterate each object contour
            for _, c in enumerate(r):
                b_mask = np.zeros(img.shape[:2], np.uint8)

                # Create contour mask
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

                # Retrieve bounding box coordinates
                x, y, w, h = c.boxes.xywh.cpu().tolist()[0]

                # Crop the region of interest (ROI) based on the bounding box
                roi = img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]

                # Extract pixels from the original image based on the mask
                masked_pixels = cv2.bitwise_and(roi, roi, mask=b_mask[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)])

                # Retrieve and print the track ID
                track_id = c.boxes.id.int().cpu().tolist()[0]

                # Update max height for the track_id
                if h > track_history[track_id]['max_height']:
                    track_history[track_id]['max_height'] = h
                    track_history[track_id]['max_height_frame'] = masked_pixels

                # Append the track history
                track_history[track_id]['track'].append((float(x), float(y)))

        frame_id += 1

        # Visualize the results on the frame
        annotated_frame = res[0].plot()

        # Plot the tracks
        for track_id, history in track_history.items():
            track = history['track']
            if track:
                # Draw the tracking lines
                points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Save the image with the maximum height bounding box for each track_id
        for track_id, history in track_history.items():
            max_height_frame = history['max_height_frame']

            if max_height_frame is not None:
                # Define the output folder and file path
                output_folder = 'resultssss' / Path(source_name) / str(track_id)
                output_folder.mkdir(parents=True, exist_ok=True)

                output_file_path = output_folder / 'max_height_image.jpg'
                cv2.imwrite(str(output_file_path), max_height_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
