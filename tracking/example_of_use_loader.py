
from math import ceil
from loaders import LoadVideoStream
import cv2
# Set the path to your video file
video_path = 'C:\\VSCode_Project\\ArtificialVision\\Code\\ProjectWork\\data\\video_prisco_tagliato.mp4' 

# Create a video stream loader
stream_loader = LoadVideoStream(source=video_path)

try:
    
    frame_count = 0
    # Iterate through the frames in the video stream
    for sources, images, _, _ in stream_loader:
        # Process the frames as needed
        print(f"Received frames {frame_count} from source: {sources}")
        # Add your processing logic here
        frame_count += 1
        # Update last_frame with the latest received frame
        #last_frame = images[0]

except KeyboardInterrupt:
    # Handle keyboard interrupt (e.g., press Ctrl+C to stop the loop)
    print("Keyboard interrupt. Stopping the stream.")

finally:
    stream_loader.close()