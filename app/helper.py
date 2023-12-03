from pathlib import Path
from ultralytics import YOLO
import streamlit as st
import cv2
import settings as settings
from pathlib import Path
import yaml
import PIL

import yaml

def load_roi_config(config_file):
    """
    Load the ROI (Region of Interest) configuration from a file.

    Args:
        config_file (file): The file object containing the ROI configuration.

    Returns:
        dict: The ROI configuration as a dictionary.

    Raises:
        None

    """
    if config_file is not None:
        # Leggi i dati del file UploadedFile come stringa
        config_data = config_file.read()
        
        # Analizza i dati YAML
        roi_config = yaml.safe_load(config_data)
        
        return roi_config
    else:
        # Gestisci il caso in cui config_file sia None
        return None
        
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

def _display_tracking_frame(conf, model, st_frame, frame):
    """
    Display the tracking results on the frame.

    Parameters:
    conf (float): Confidence threshold for detection.
    model (Model): The tracking model.
    st_frame (Streamlit.image): Streamlit image object to display the frame.
    frame (numpy.ndarray): The input frame.

    Returns:
    None
    """
    # Run model on tracking frame, persisting detection IDs
    results = model.track(frame, conf=conf, persist=True, classes=[0], device='cpu', tracker="bytetrack.yaml")
    # visulize the results on the frame
    annoted_frame = results[0].plot()
    # Display the annotated frame
    st_frame.image(annoted_frame,
                   caption='Tracked Video',
                   channels="BGR",
                   use_column_width=True)
        
def _display_video(source_vid, conf, model, mode="Detection"):
    """
    Display video frames and perform object detection or tracking based on the specified mode.

    Args:
        source_vid (str): The path to the video file.
        conf (float): The confidence threshold for object detection or tracking.
        model: The object detection or tracking model.
        mode (str, optional): The mode of operation. Can be "Detection" or "Tracking". Defaults to "Detection".
    """
    vid_cap = cv2.VideoCapture(str(settings.VIDEOS_DICT.get(source_vid)))
    st_frame = st.empty()

    # Get the frame rate of the video
    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))

    # Specify the desired frame rate (e.g., process every other frame)
    desired_frame_rate = fps

    frame_count = 0

    while vid_cap.isOpened():
        success, image = vid_cap.read()

        if success:
            if frame_count % desired_frame_rate == 0:
                if mode == "Detection":
                    _display_detected_frames(conf, model, st_frame, image)
                elif mode == "Tracking":
                    _display_tracking_frame(conf, model, st_frame, image)
        else:
            vid_cap.release()
            break

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

def play_image(confidence, model):
    """
    Display and process images for object detection.

    Args:
        confidence (float): The confidence threshold for object detection.
        model: The object detection model.

    Returns:
        None
    """
    
    source_img = None
    source_img = st.sidebar.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image, caption="Default Image", use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(default_detected_image_path)
            st.image(default_detected_image, caption='Detected Image', use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                try:
                    res = model.predict(uploaded_image, conf=confidence, classes=[0])
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    st.image(res_plotted, caption='Detected Image', use_column_width=True)

                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    st.error("Error occurred during object detection.")
                    st.error(ex)
                    st.write("No image is uploaded yet!")

def play_stored_video(conf, model, mode, source_vid, roi_config = None):
    """
    Plays a stored video file.

    Parameters:
    - conf (object): Configuration object.
    - model (object): Model object.
    - mode (str): Mode of video playback.
    - source_vid (str): Name of the video file to be played.
    - roi_config (object, optional): Region of interest configuration object.

    Returns:
    None
    """
    
    # Read video file as bytes
    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    
    # Display the video if bytes are available
    if video_bytes:
        st.video(video_bytes)

    # Detect video objects when button is clicked
    
    try:
        _display_video(source_vid, conf, model, mode = mode)
    except Exception as e:
        st.sidebar.error(f"An error occurred: {e}")
    