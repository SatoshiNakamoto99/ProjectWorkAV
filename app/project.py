#python app/project.py --video data/video_prisco_tagliato.mp4 --configuration config.json --results results/video_prisco_tagliato/results.json
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