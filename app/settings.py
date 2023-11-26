

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
VIDEOS_DICT = {
    'video_elite': VIDEO_1_PATH,
}

# ML Model config
MODEL_DIR = '.'
DETECTION_MODEL = Path(MODEL_DIR) / 'yolov8n.pt'


# Webcam
WEBCAM_PATH = 0