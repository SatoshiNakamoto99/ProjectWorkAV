#python app/group07.py --video data/video_atrio_cues/video01.mp4 --configuration config.json --results results/video_atrio_cues/video01.json
#python app/group07.py --video data/video_atrio_cues/video02.mp4 --configuration config.json --results results/video_atrio_cues/video02.json
#python app/group07.py --video data/video_atrio_cues/video03.mp4 --configuration config.json --results results/video_atrio_cues/video03.json
#python app/group07.py --video data/video_atrio_cues/video04.mp4 --configuration config.json --results results/video_atrio_cues/video04.json
#python app/group07.py --video data/video_atrio_cues/video05.mp4 --configuration config.json --results results/video_atrio_cues/video05.json
#python app/group07.py --video data/video_atrio_cues/video06.mp4 --configuration config.json --results results/video_atrio_cues/video06.json
#python app/group07.py --video data/video_atrio_cues/video07.mp4 --configuration config.json --results results/video_atrio_cues/video07.json
#python app/group07.py --video data/video_atrio_cues/video08.mp4 --configuration config.json --results results/video_atrio_cues/video08.json
#python app/group07.py --video data/video_atrio_cues/video09.mp4 --configuration config.json --results results/video_atrio_cues/video09.json
#python app/group07.py --video data/video_atrio_cues/video10.mp4 --configuration config.json --results results/video_atrio_cues/video10.json
#python app/group07.py --video data/video_atrio_cues/video11.mp4 --configuration config.json --results results/video_atrio_cues/video11.json
#python app/group07.py --video data/video_atrio_cues/video12.mp4 --configuration config.json --results results/video_atrio_cues/video12.json
#python app/group07.py --video data/video_atrio_cues/video13.mp4 --configuration config.json --results results/video_atrio_cues/video13.json
#python app/group07.py --video data/video_atrio_cues/video14.mp4 --configuration config.json --results results/video_atrio_cues/video14.json
#python app/group07.py --video data/video_atrio_cues/video15.mp4 --configuration config.json --results results/video_atrio_cues/video15.json
#python app/group07.py --video data/video_atrio_cues/video16.mp4 --configuration config.json --results results/video_atrio_cues/video16.json
#python app/group07.py --video data/video_atrio_cues/video17.mp4 --configuration config.json --results results/video_atrio_cues/video17.json
#python app/group07.py --video data/video_atrio_cues/video18.mp4 --configuration config.json --results results/video_atrio_cues/video18.json
#python app/group07.py --video data/video_atrio_cues/video19.mp4 --configuration config.json --results results/video_atrio_cues/video19.json
#python app/group07.py --video data/video_atrio_cues/video20.mp4 --configuration config.json --results results/video_atrio_cues/video20.json
#python app/group07.py --video data/video_atrio_cues/video21.mp4 --configuration config.json --results results/video_atrio_cues/video21.json
#python app/group07.py --video data/video_atrio_cues/video22.mp4 --configuration config.json --results results/video_atrio_cues/video22.json
#python app/group07.py --video data/video_atrio_cues/video23.mp4 --configuration config.json --results results/video_atrio_cues/video23.json
#python app/group07.py --video data/video_atrio_cues/video24.mp4 --configuration config.json --results results/video_atrio_cues/video24.json


import argparse
from tracking import ObjectTracker
from ultralytics.utils import LOGGER
LOGGER.setLevel("WARNING")  # Puoi impostare anche su "ERROR" o "CRITICAL" per stampare solo gli avvisi critici
import time

def get_parameters():
    parser = argparse.ArgumentParser(description="Tracking delle persone")

    # Aggiungi gli argomenti
    parser.add_argument("--video", type=str, help="Percorso del video da processare")
    parser.add_argument("--configuration", type=str, help="Percorso del file di configurazione")
    parser.add_argument("--results", type=str, help="Percorso del file dei risultati")

    # Parsa gli argomenti dalla riga di comando
    args = parser.parse_args()

    # Recupera i valori degli argomenti
    video = args.video
    configuration = args.configuration
    results = args.results
    return video,configuration,results


### START READING ###
start_time = time.time()
VERBOSE = False
video,configuration,results = get_parameters()
tracker = ObjectTracker(video,configuration,results,verbose=VERBOSE)
# tracker.save_single_frame()
tracker.perform_tracking()
stop_time = time.time()
print('Processing time: ', stop_time-start_time)