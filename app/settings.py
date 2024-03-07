from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent.parent

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# ML Model config
MODEL_DIR = './models'
DETECTION_MODEL = Path(MODEL_DIR) / 'best_Kfod5_100epoc_base.pt'
PAR_MODEL = Path("attributes_recognition_module/model/MultiTaskNN_ConvNeXt_v1_CBAM_after_exam.pth")


# TRACKING
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'  # video suffixes
CONFIG = ROOT / 'config'
MY_TRACKER = CONFIG / 'botsort.yaml'