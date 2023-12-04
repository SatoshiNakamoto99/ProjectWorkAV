# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st
import tkinter as tk
from tkinter import filedialog

# Local Modules
import settings as settings
import helper as helper

def main():
    # Setting page layout
    st.set_page_config(
        page_title="Artificial Vision",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Main page heading
    st.title("Project Artificial Vision")

    # Sidebar
    st.sidebar.header("ML Model Config")

    # Model Options
    model_type = st.sidebar.radio(
        "Select Task", ['Detection/Tracking'])

    confidence = float(st.sidebar.slider(
        "Select Model Confidence", 25, 100, 40)) / 100
    print(settings)
    if model_type == 'Detection/Tracking':
        model_path = Path(settings.DETECTION_MODEL)

    # Load Pre-trained ML Model
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        print(model_path)
        st.error(ex)

    st.sidebar.header("Image/Video Config")
    source_radio = st.sidebar.radio(
        "Select Source", settings.SOURCES_LIST)

    
    if source_radio == settings.IMAGE:
        helper.play_image(confidence=confidence, model=model)
    elif source_radio == settings.VIDEO:
        # select mode : Tracking or Detection
        mode = st.sidebar.radio("Select Mode", ['Detection', 'Tracking'])
        source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())
        if mode == 'Tracking':
            #config_path = st.sidebar.file_uploader("Select ROI Configuration File (YAML)", type=['yaml'])
            config_path = st.sidebar.file_uploader("Select ROI Configuration File (JSON)", type=['json'])
            if config_path:
                roi_config = helper.load_roi_config(config_path)
                st.sidebar.success("ROI Configuration Loaded!")
                #print(roi_config)
            else:
                roi_config = None
            if st.sidebar.button('Tracking Video Objects'):
                helper.play_stored_video(confidence, model, mode, source_vid, roi_config)
                st.sidebar.success("Results succesfully saved!")
        else:
            if st.sidebar.button('Detect Video Objects'):
                helper.play_stored_video(confidence, model, mode, source_vid)
            
        
       

    elif source_radio == settings.WEBCAM:
        helper.play_webcam(confidence, model)

    else:
        st.error("Please select a valid source type!")

if __name__=="__main__":
    main()