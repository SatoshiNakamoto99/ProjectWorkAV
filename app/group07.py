#python app/group07.py --video data/video_atrio_cues/video01.mp4 --configuration config.txt --results results/video_atrio_cues/video01.txt
#python app/group07.py --video data/video_atrio_cues/video02.mp4 --configuration config.txt --results results/video_atrio_cues/video02.txt
#python app/group07.py --video data/video_atrio_cues/video03.mp4 --configuration config.txt --results results/video_atrio_cues/video03.txt
#python app/group07.py --video data/video_atrio_cues/video04.mp4 --configuration config.txt --results results/video_atrio_cues/video04.txt
#python app/group07.py --video data/video_atrio_cues/video05.mp4 --configuration config.txt --results results/video_atrio_cues/video05.txt
#python app/group07.py --video data/video_atrio_cues/video06.mp4 --configuration config.txt --results results/video_atrio_cues/video06.txt
#python app/group07.py --video data/video_atrio_cues/video07.mp4 --configuration config.txt --results results/video_atrio_cues/video07.txt
#python app/group07.py --video data/video_atrio_cues/video08.mp4 --configuration config.txt --results results/video_atrio_cues/video08.txt
#python app/group07.py --video data/video_atrio_cues/video09.mp4 --configuration config.txt --results results/video_atrio_cues/video09.txt
#python app/group07.py --video data/video_atrio_cues/video10.mp4 --configuration config.txt --results results/video_atrio_cues/video10.txt
#python app/group07.py --video data/video_atrio_cues/video11.mp4 --configuration config.txt --results results/video_atrio_cues/video11.txt
#python app/group07.py --video data/video_atrio_cues/video12.mp4 --configuration config.txt --results results/video_atrio_cues/video12.txt
#python app/group07.py --video data/video_atrio_cues/video13.mp4 --configuration config.txt --results results/video_atrio_cues/video13.txt
#python app/group07.py --video data/video_atrio_cues/video14.mp4 --configuration config.txt --results results/video_atrio_cues/video14.txt
#python app/group07.py --video data/video_atrio_cues/video15.mp4 --configuration config.txt --results results/video_atrio_cues/video15.txt
#python app/group07.py --video data/video_atrio_cues/video16.mp4 --configuration config.txt --results results/video_atrio_cues/video16.txt
#python app/group07.py --video data/video_atrio_cues/video17.mp4 --configuration config.txt --results results/video_atrio_cues/video17.txt
#python app/group07.py --video data/video_atrio_cues/video18.mp4 --configuration config.txt --results results/video_atrio_cues/video18.txt
#python app/group07.py --video data/video_atrio_cues/video19.mp4 --configuration config.txt --results results/video_atrio_cues/video19.txt
#python app/group07.py --video data/video_atrio_cues/video20.mp4 --configuration config.txt --results results/video_atrio_cues/video20.txt
#python app/group07.py --video data/video_atrio_cues/video21.mp4 --configuration config.txt --results results/video_atrio_cues/video21.txt
#python app/group07.py --video data/video_atrio_cues/video22.mp4 --configuration config.txt --results results/video_atrio_cues/video22.txt
#python app/group07.py --video data/video_atrio_cues/video23.mp4 --configuration config.txt --results results/video_atrio_cues/video23.txt
#python app/group07.py --video data/video_atrio_cues/video24.mp4 --configuration config.txt --results results/video_atrio_cues/video24.txt

#python app/group07.py --video data/colors/black.mp4 --configuration config.txt --results results/colors/black.txt
#python app/group07.py --video data/colors/blue.mp4 --configuration config.txt --results results/colors/blue.txt
#python app/group07.py --video data/colors/brown_purple.mp4 --configuration config.txt --results results/colors/brown_purple.txt
#python app/group07.py --video data/colors/gray.mp4 --configuration config.txt --results results/colors/gray.txt
#python app/group07.py --video data/colors/green.mp4 --configuration config.txt --results results/colors/green.txt
#python app/group07.py --video data/colors/pink.mp4 --configuration config.txt --results results/colors/pink.txt
#python app/group07.py --video data/colors/red.mp4 --configuration config.txt --results results/colors/red.txt
#python app/group07.py --video data/colors/white.mp4 --configuration config.txt --results results/colors/white.txt
#python app/group07.py --video data/colors/yellow.mp4 --configuration config.txt --results results/colors/yellow.txt


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