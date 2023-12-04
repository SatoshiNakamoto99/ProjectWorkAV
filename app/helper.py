from pathlib import Path
from ultralytics import YOLO
import streamlit as st
import cv2
import settings as settings
from pathlib import Path
import yaml
import PIL

import yaml
import json
from math import ceil

RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
colors = {0:RED,1:BLUE,2:GREEN}


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

        # Convert the JSON string to a dictionary
        roi_config = json.loads(config_data)
        
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

def _display_tracking_frame(conf, model, st_frame, frame, rescaled_rois, cap, people):
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

    x_roi1,y_roi1,w_roi1,h_roi1 = rescaled_rois[0]
    x_roi2,y_roi2,w_roi2,h_roi2 = rescaled_rois[1]


    # Run model on tracking frame, persisting detection IDs
    results = model.track(frame, conf=conf, persist=True, classes=[0], device='cpu', tracker="bytetrack.yaml")

    # Get the boxes and track IDs
    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()

    # visulize the results on the frame
    annotated_frame = results[0].plot()
    # Display the annotated frame

    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box

        roi = get_roi_of_belonging(x,y,x_roi1,y_roi1,w_roi1,h_roi1,x_roi2,y_roi2,w_roi2,h_roi2)

        # Disegna il bounding box con il colore appropriato
        color = colors[roi]
        cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, 4)

        get_roi_passages_and_persistence(cap, people, track_id, roi)

    get_persitence_for_no_more_tracked_people(people,track_ids,cap)
    last_time=cap.get(cv2.CAP_PROP_POS_MSEC)


    st_frame.image(annotated_frame,
                   caption='Tracked Video',
                   channels="BGR",
                   use_column_width=True)
        
def _display_video(source_vid, conf, model, roi_config=None, mode="Detection"):
    """
    Display video frames and perform object detection or tracking based on the specified mode.

    Args:
        source_vid (str): The path to the video file.
        conf (float): The confidence threshold for object detection or tracking.
        model: The object detection or tracking model.
        mode (str, optional): The mode of operation. Can be "Detection" or "Tracking". Defaults to "Detection".
    """
    people = {}    
    last_time = 0
    skip_frames = 5
    frame_count = 0

    vid_cap = cv2.VideoCapture(str(settings.VIDEOS_DICT.get(source_vid)))
    # Store the track history
    st_frame = st.empty()

    # Get the frame rate of the video
    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))

    # Specify the desired frame rate (e.g., process every other frame)
    #desired_frame_rate = ceil(2*fps/3)
    #frame_count = 0
    #toggle = False
   

    while vid_cap.isOpened():
        success, image = vid_cap.read()
        rescaled_rois=get_rescaled_rois(vid_cap,roi_config)

        if success:
            #if toggle:
              #  if frame_count < desired_frame_rate :
                if frame_count % (skip_frames + 1) == 0:
             #       frame_count= frame_count+1
                    if mode == "Detection":
                        _display_detected_frames(conf, model, st_frame, image)
                    elif mode == "Tracking":
                        _display_tracking_frame(conf, model, st_frame, image, rescaled_rois, vid_cap, people)
           # toggle = not toggle
        else:
            update_persistence(people,last_time)
            save_tracking_results(people)
            vid_cap.release()
            break

def get_rescaled_rois(vid_cap,roi_config):
    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rois = roi_config.values()
    rescaled_rois = []

    for roi in rois:
        x = int(roi["x"] * width)
        y = int(roi["y"] * height)
        w = int(roi["w"] * width)
        h = int(roi["h"] * height)
        rescaled_rois.append((x,y,w,h))

    return rescaled_rois

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
        results = _display_video(source_vid, conf, model, roi_config, mode = mode)
        return results
    except Exception as e:
        st.sidebar.error(f"An error occurred: {e}")
    
######################

def get_roi_of_belonging(x,y,x_roi1,y_roi1,w_roi1,h_roi1,x_roi2,y_roi2,w_roi2,h_roi2):
    # Verifica se il punto (x, y) appartiene a ROI1
    if x_roi1 <= x <= x_roi1 + w_roi1 and y_roi1 <= y <= y_roi1 + h_roi1:
        roi = 1
    # Verifica se il punto (x, y) appartiene a ROI2
    elif x_roi2 <= x <= x_roi2 + w_roi2 and y_roi2 <= y <= y_roi2 + h_roi2:
        roi = 2
    # Se il punto non appartiene a nessuna ROI, assegna il colore no_roi
    else:
        roi = 0
    return roi


def get_roi_passages_and_persistence(cap, people, track_id, roi):
    if track_id not in people.keys():
        if roi==1:
            people[track_id] = {"roi1_passages":1,"roi1_persistence_time":0,"roi2_passages":0,"roi2_persistence_time":0,"prev_roi":roi,"start_persistence":0,"lost_tracking":False}
        elif roi==2:
            people[track_id] = {"roi1_passages":0,"roi1_persistence_time":0,"roi2_passages":1,"roi2_persistence_time":0,"prev_roi":roi,"start_persistence":0,"lost_tracking":False}
        else:
            people[track_id] = {"roi1_passages":0,"roi1_persistence_time":0,"roi2_passages":0,"roi2_persistence_time":0,"prev_roi":roi,"start_persistence":-1,"lost_tracking":False}
    else:
        if roi == 1 and (people[track_id]["prev_roi"] != 1 or people[track_id]["lost_tracking"]):
            people[track_id]["roi1_passages"] = people[track_id]["roi1_passages"] + 1
            people[track_id]["start_persistence"] = cap.get(cv2.CAP_PROP_POS_MSEC)
        elif roi == 2 and (people[track_id]["prev_roi"] != 2 or people[track_id]["lost_tracking"]):
            people[track_id]["roi2_passages"] = people[track_id]["roi2_passages"] + 1
            people[track_id]["start_persistence"] = cap.get(cv2.CAP_PROP_POS_MSEC)
        elif((roi != 1 and people[track_id]["prev_roi"] == 1) or (roi != 2 and people[track_id]["prev_roi"] == 2)):
            stop_persistence = cap.get(cv2.CAP_PROP_POS_MSEC)
            time_of_persistence=stop_persistence-people[track_id]["start_persistence"]
            if people[track_id]["prev_roi"] == 1 and people[track_id]["roi1_persistence_time"] != -1:
                people[track_id]["roi1_persistence_time"] = people[track_id]["roi1_persistence_time"]+time_of_persistence/1000.0
                people[track_id]["start_persistence"] = -1
            elif people[track_id]["prev_roi"] == 2 and people[track_id]["roi2_persistence_time"] != -1:
                people[track_id]["roi2_persistence_time"] = people[track_id]["roi2_persistence_time"]+time_of_persistence/1000.0
                people[track_id]["start_persistence"] = -1
        
    people[track_id]["prev_roi"] = roi
    people[track_id]["lost_tracking"]=False


def get_persitence_for_no_more_tracked_people(people,track_ids,cap):
    for track_id in people.keys():
        if track_id not in track_ids:
            people[track_id]["lost_tracking"]=True
            if people[track_id]["start_persistence"] != -1:
                stop_persistence = cap.get(cv2.CAP_PROP_POS_MSEC)
                print(stop_persistence)
                time_of_persistence=stop_persistence-people[track_id]["start_persistence"]
                if people[track_id]["prev_roi"] == 1:
                    people[track_id]["roi1_persistence_time"] = people[track_id]["roi1_persistence_time"]+time_of_persistence/1000.0
                elif people[track_id]["prev_roi"] == 2:
                    people[track_id]["roi2_persistence_time"] = people[track_id]["roi2_persistence_time"]+time_of_persistence/1000.0
                people[track_id]["start_persistence"] = -1


def update_persistence(people,last_time):
    for track_id in people.keys():
        if people[track_id]["start_persistence"] != -1:
            stop_persistence = last_time
            time_of_persistence=stop_persistence-people[track_id]["start_persistence"]
            if people[track_id]["prev_roi"] == 1:
                people[track_id]["roi1_persistence_time"] = people[track_id]["roi1_persistence_time"]+time_of_persistence/1000.0
                people[track_id]["start_persistence"] = -1
            elif people[track_id]["prev_roi"] == 2:
                people[track_id]["roi2_persistence_time"] = people[track_id]["roi2_persistence_time"]+time_of_persistence/1000.0
                people[track_id]["start_persistence"] = -1


def save_tracking_results(people):
    filtered_people = []
    for person_id, person in people.items():
        filtered_people.append({"id": person_id, "roi1_passages": person["roi1_passages"],
                                "roi1_persistence_time": ceil(person["roi1_persistence_time"]),
                                "roi2_passages": person["roi2_passages"],
                                "roi2_persistence_time": ceil(person["roi2_persistence_time"])})

    data = {"people": filtered_people}

    with open("results.json", 'w') as file:
        json.dump(data, file, indent=2)