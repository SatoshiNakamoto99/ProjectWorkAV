from pathlib import Path
from ultralytics import YOLO
import streamlit as st
import cv2
import settings as settings
from pathlib import Path


def load_model(model_path):
    """
    Load a YOLO model from the given model_path.

    Args:
        model_path (str): The path to the YOLO model file.

    Returns:
        YOLO: The loaded YOLO model.

    Raises:
        AssertionError: If the model file does not exist.
    """
    assert Path(model_path).exists(), 'Model file does not exist.'
    model = YOLO(model_path)
    return model



def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected frames on the video frame.

    Parameters:
    conf (float): Confidence threshold for object detection.
    model (Model): YOLOv8 model for object detection.
    st_frame (Streamlit): Streamlit object for displaying the video frame.
    image (numpy.ndarray): Input image for object detection.

    Returns:
    None
    """
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Predict the objects in the image using the YOLOv8 model
    res = model.predict(image, conf=conf, classes=[0], device='cpu')

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )



def play_webcam(conf, model):
    """
    Plays the webcam video and detects objects in real-time.

    Parameters:
    conf (object): The configuration object.
    model (object): The object detection model.

    Returns:
    None
    """
    
    source_webcam = settings.WEBCAM_PATH
    
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_stored_video(conf, model):
    """
    Plays a stored video and detects objects in real-time.

    Args:
        conf (object): Configuration object.
        model (object): Object detection model.

    Returns:
        None
    """
    
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    # Read video file as bytes
    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    
    # Display the video if bytes are available
    if video_bytes:
        st.video(video_bytes)

    # Detect video objects when button is clicked
    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()

            # Get the frame rate of the video
            fps = int(vid_cap.get(cv2.CAP_PROP_FPS))

            # Specify the desired frame rate (e.g., process every other frame)
            desired_frame_rate = fps 

            frame_count = 0
            #process_frame = False

            while (vid_cap.isOpened()):
                success, image = vid_cap.read()

                if success:
                    #if process_frame:
                        #frame_count += 1

                    # Process frames based on the desired frame rate
                    if frame_count % desired_frame_rate == 0:
                        _display_detected_frames(conf, model, st_frame, image)
                    #process_frame = not process_frame
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error(f"An error occurred: {e}")