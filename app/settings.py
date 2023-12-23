from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent.parent

####  IMPORTANTE  ####

# Add the root path to the sys.path list if it is not already there

#if root_path not in sys.path:
#    sys.path.append(str(root_path))


####  IMPORTANTE  ####

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'
RTSP = 'RTSP'
YOUTUBE = 'YouTube'

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM]

# Images config
IMAGES_DIR = ROOT / 'data'
DEFAULT_IMAGE = IMAGES_DIR / 'test_image.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'test_image_pred.jpg'

# Videos config
VIDEO_DIR = ROOT / 'data'
VIDEO_1_PATH = VIDEO_DIR /'video_a_caso.mp4'
VIDEO_2_PATH = VIDEO_DIR / 'video_prisco.MOV'
VIDEO_3_PATH = VIDEO_DIR / 'video_prisco_tagliato.mp4'
VIDEOS_DICT = {
    'video_elite': VIDEO_1_PATH,
    'video_Atrio_Cues': VIDEO_2_PATH,
    'Video_Atrio_Cues_Tagliato': VIDEO_3_PATH
}

# ML Model config
MODEL_DIR = './models'
DETECTION_MODEL = Path(MODEL_DIR) / 'yolov8n-seg.pt'


# Webcam
WEBCAM_PATH = 0

# TRACKING
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'  # video suffixes
CONFIG = ROOT / 'config'
MY_TRACKER = CONFIG / 'botsort.yaml'