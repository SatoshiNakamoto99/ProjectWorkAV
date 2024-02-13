import argparse
from tracking import ObjectTracker
from ultralytics.utils import LOGGER
LOGGER.setLevel("WARNING")
import time

def get_parameters():
    parser = argparse.ArgumentParser(description="Tracking delle persone")

    parser.add_argument("--video", type=str, help="Percorso del video da processare")
    parser.add_argument("--configuration", type=str, help="Percorso del file di configurazione")
    parser.add_argument("--results", type=str, help="Percorso del file dei risultati")

    args = parser.parse_args()

    video = args.video
    configuration = args.configuration
    results = args.results
    return video,configuration,results


if __name__ == '__main__':
    start_time = time.time()
    VERBOSE = False
    video,configuration,results = get_parameters()
    tracker = ObjectTracker(video,configuration,results,verbose=VERBOSE)
    tracker.perform_tracking()
    stop_time = time.time()
    print('Processing time: ', stop_time-start_time)