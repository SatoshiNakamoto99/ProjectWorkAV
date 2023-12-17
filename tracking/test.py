import time
from loaders import LoadVideoStream
import cv2

# Crea un oggetto LoadVideoStream con fps_out=5
stream_loader = LoadVideoStream(source='data/video_prisco_tagliato.mp4', fps_out=5)
frame_count = 0

try:
    for source, images, _, _ in stream_loader:
        for frame in images:
            # Process the frames as needed
            print(f"Received frames {frame_count} from source: {source}")
            # Add your processing logic here

            # Visualizza il frame utilizzando cv2.imshow()
            cv2.imshow('Frame', frame)

            # Attendi per 30 millisecondi e controlla se l'utente preme 'q' per interrompere
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                raise KeyboardInterrupt

            frame_count += 1

except KeyboardInterrupt:
    # Handle keyboard interrupt (e.g., press 'q' to stop the loop)
    print("Keyboard interrupt. Stopping the stream.")

finally:
    stream_loader.close()
    # Chiudi tutte le finestre aperte
    cv2.destroyAllWindows()
