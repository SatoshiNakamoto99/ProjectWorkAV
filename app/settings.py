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

# ML Model config
MODEL_DIR = './models'
DETECTION_MODEL = Path(MODEL_DIR) / 'best_Kfod5_100epoc_base.pt'
PAR_MODEL = Path("attributes_recognition_module/model/MultiTaskNN_ConvNeXt_v1_CBAM_64_192.pth")


# TRACKING
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'  # video suffixes
CONFIG = ROOT / 'config'
MY_TRACKER = CONFIG / 'botsort.yaml'