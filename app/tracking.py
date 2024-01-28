import cv2 as cv
import numpy as np
from collections import defaultdict
import json
from math import ceil
from pathlib import Path
from datetime import timedelta
import shutil
from par_inference import PARModuleInference

import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, ".."))
sys.path.append(project_path)

from my_yolo import MyYOLO
from loaders import LoadVideoStream
import settings as settings


RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)


class ObjectTracker:
    """
    A class for object tracking in a video using YOLOv8 and tracking algorithms.

    Attributes:
        tracking_model (MyYOLO): The YOLOv8 model for object detection and tracking.
        track_history (defaultdict): A dictionary to store the tracking history of objects.
        video_path (str): The path to the input video file.
        colors (dict): A dictionary mapping class IDs to BGR color values for visualization.

    Methods:
        get_rescaled_rois: Extract and rescale Regions of Interest (ROIs) from a video.
        perform_tracking: Perform object tracking on the video, compute various metrics, and display results.
        compute_masks: Compute masks for detected objects in the video frames.
        save_images: Save images of maximum height bounding box, upper, and lower frames for each tracked object.
        estimate_predominant_color: Estimate the predominant color in a pixel region using color analysis.
        get_roi_of_belonging: Determine the ROI to which a point belongs based on its coordinates.
        get_roi_passages_and_persistence: Update ROI passages and persistence metrics for tracked objects.
        get_persitence_for_no_more_tracked_people: Update persistence metrics for objects that are no longer tracked.
        update_persistence: Update persistence metrics for tracked objects at the end of tracking.
        milliseconds_to_hh_mm_ss: Convert milliseconds to a formatted time string (hh:mm:ss).
        display_annoted_frame: Display the annotated frame with ROIs and tracking information.
        save_tracking_results: Save the tracking results, including color information and persistence metrics, to a JSON file.
    """

    # Initializes the tracking process with the specified YOLO model and video path
    def __init__(self, video_path, configuration_path, results_path, model_path=settings.DETECTION_MODEL, verbose=True):
        self.tracking_model = MyYOLO(model_path)
        self.track_history = defaultdict(lambda: [])
        self.video_path = video_path
        self.configuration_path = configuration_path
        self.results_path = results_path
        self.colors = {0:RED,1:BLUE,2:GREEN}
        self.verbose = verbose
        video_name = Path(self.video_path).name.split('.')[0]
        folder_path = 'results' + '/' + video_name
        self.par_module_path = Path("attributes_recognition_module/model/MultiTaskNN_ConvNeXt_v1_CBAM.pth")
        self.par_module = PARModuleInference(self.par_module_path)
        self.actual_detected_person = None
        shutil.rmtree(folder_path, ignore_errors=True)
        

    # Gets the rescaled Regions of Interest (ROIs) from the video based on a configuration JSON file
    def get_rescaled_rois(self):
        cap = cv.VideoCapture(self.video_path)
        if not cap.isOpened():
            if self.verbose:
                print("Errore nell'apertura del video.")
        else:
            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

            # Carica il file JSON
            with open(self.configuration_path) as json_file:
                data = json.load(json_file)

            # Recupera un singolo frame per testare
            # single_frame = cv.imread("single_frame.jpg")

            # Estrai le ROIs dal dizionario JSON
            rois = data.values()
            processed_rois = []

            # Disegna i bounding box sull'immagine
            for roi in rois:
                # Riscalare le coordinate e le dimensioni relative a valori assoluti
                x = int(roi["x"] * width)
                y = int(roi["y"] * height)
                w = int(roi["w"] * width)
                h = int(roi["h"] * height)
                processed_rois.append((x,y,w,h))

            return processed_rois


    # Performs the tracking process using the YOLO model
    def perform_tracking(self):
        rois = self.get_rescaled_rois()
        x_roi1,y_roi1,w_roi1,h_roi1 = rois[0]
        x_roi2,y_roi2,w_roi2,h_roi2 = rois[1]

        people = {}    
        last_time = 0

        # Set the path to your video file
        source_name = Path(self.video_path).name.split('.')[0]

        frame_id = 0
        # bbox_history = defaultdict(lambda: {'max_height': 0, 'max_height_frame': None, 'track': [], 'upper_frame': None, 'lower_frame': None, 'upper_color': None, 'lower_color': None})

        # Create a video stream loader
        stream_loader = LoadVideoStream(source=self.video_path, fps_out=5)

        # Store the track history
        #track_history = defaultdict(lambda: [])

        try:
            for _, images, _, _ in stream_loader:
                for frame, timestamp in images:

                    tracking_results = self.tracking_model.track(frame, persist=True, conf=0.3, classes=[0], device='cpu', tracker="config/botsort.yaml")

                    frame_id = frame_id+1

                    # Visualize the results on the frame
                    tracking_annotated_frame = tracking_results[0].plot()

                    # Get the boxes and track IDs
                    boxes = tracking_results[0].boxes.xywh.cpu()
                    track_ids = tracking_results[0].boxes.id.int().cpu().tolist()

                    # Plot the tracks
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        # track = track_history[track_id]
                        # track.append((float(x), float(y)))  # x, y center point
                        # if len(track) > 30:  # retain 90 tracks for 90 frames
                        #     track.pop(0)

                        # Draw the tracking lines
                        # points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        # cv.polylines(tracking_annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

                        roi = self.get_roi_of_belonging(x,y,x_roi1,y_roi1,w_roi1,h_roi1,x_roi2,y_roi2,w_roi2,h_roi2)

                        # Disegna il bounding box con il colore appropriato
                        color = self.colors[roi]
                        cv.rectangle(tracking_annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, 4)
                        
                        self.actual_detected_person = frame[ int(y-h/2):int(y+h/2),
                                        int(x-w/2):int(x+w/2) ] 
                        
                        # cv.imshow("prova", self.actual_detected_person)
                        # cv.waitKey(0)
                        
                        self.get_roi_passages_and_persistence(people, track_id, roi, timestamp)
                        
                        attributes_string = "gender:" + people[track_id]["gender"] +  "\n bag:" + people[track_id]["bag"] + "\n hat:" + people[track_id]["hat"] + "\n upper_color:" + people[track_id]["upper_color"] + "\n lower_color:" + people[track_id]["lower_color"]
                                            
                        y0 = y
                        dy = 10
                        for i, line in enumerate(attributes_string.split('\n')):
                            y = y0 + i*dy
                            # cv.putText(img, line, (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                            cv.putText(tracking_annotated_frame, line, (int(x+w/2), int(y+h/2)), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=color, thickness=1)
                        
                        
                        
                        # print(f"results:\n{par_results}")
                        
                        # exit()


                    self.get_persitence_for_no_more_tracked_people(people,track_ids,timestamp)
                    self.display_annoted_frame(tracking_annotated_frame,x_roi1,y_roi1,w_roi1,h_roi1,x_roi2,y_roi2,w_roi2,h_roi2)
                    last_time=timestamp

                    cv.waitKey(1)

        except KeyboardInterrupt:
            print("Keyboard interrupt. Stopping the stream.")
        finally:
            self.update_persistence(people,last_time)
            self.save_tracking_results(people)
            stream_loader.close()
            #cv.destroyAllWindows()

    
    
    # Returns the Region of Interest (ROI) to which a point (x, y) belongs
    def get_roi_of_belonging(self,x,y,x_roi1,y_roi1,w_roi1,h_roi1,x_roi2,y_roi2,w_roi2,h_roi2):
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
    

    # Handles ROI passages and persistence of tracked people
    def get_roi_passages_and_persistence(self, people, track_id, roi, timestamp):
        if track_id not in people:
            
            par_results = self.par_module.prediction(self.actual_detected_person)
            
            people[track_id] = {
                "gender":par_results["gender"],
                "bag":par_results["bag"],
                "hat":par_results["hat"],
                "upper_color":par_results["upper_color"],
                "lower_color":par_results["lower_color"],
                "roi1_passages": 0,
                "roi1_persistence_time": 0,
                "roi2_passages": 0,
                "roi2_persistence_time": 0,
                "prev_roi": roi,
                "lost_tracking": False,
            }
            if roi == 1:
                people[track_id]["roi1_passages"] = 1
                people[track_id]["start_persistence"] = timestamp
                time = self.milliseconds_to_hh_mm_ss(timestamp)
                if self.verbose:
                    print(f"[{time}]: track id {track_id} entered roi1")
            elif roi == 2:
                people[track_id]["roi2_passages"] = 1
                people[track_id]["start_persistence"] = timestamp
                time = self.milliseconds_to_hh_mm_ss(timestamp)
                if self.verbose:
                    print(f"[{time}]: track id {track_id} entered roi2")
            else:
                people[track_id]["start_persistence"] = -1

        else:
            time = self.milliseconds_to_hh_mm_ss(timestamp)

            if roi == 1 and (people[track_id]["prev_roi"] != 1 or people[track_id]["lost_tracking"]):
                if self.verbose:
                    print(f"[{time}]: track id {track_id} entered roi1")
                people[track_id]["roi1_passages"] = people[track_id]["roi1_passages"] + 1
                people[track_id]["start_persistence"] = timestamp
            elif roi == 2 and (people[track_id]["prev_roi"] != 2 or people[track_id]["lost_tracking"]):
                if self.verbose:
                    print(f"[{time}]: track id {track_id} entered roi2")
                people[track_id]["roi2_passages"] = people[track_id]["roi2_passages"] + 1
                people[track_id]["start_persistence"] = timestamp
            elif((roi != 1 and people[track_id]["prev_roi"] == 1) or (roi != 2 and people[track_id]["prev_roi"] == 2)):
                stop_persistence = timestamp
                time_of_persistence=stop_persistence-people[track_id]["start_persistence"]
                if people[track_id]["prev_roi"] == 1 and people[track_id]["roi1_persistence_time"] != -1:
                    if self.verbose:
                        print(f"[{time}]: track id {track_id} exited roi1")
                    people[track_id]["roi1_persistence_time"] = people[track_id]["roi1_persistence_time"]+time_of_persistence/1000.0
                    people[track_id]["start_persistence"] = -1
                elif people[track_id]["prev_roi"] == 2 and people[track_id]["roi2_persistence_time"] != -1:
                    if self.verbose:
                        print(f"[{time}]: track id {track_id} exited roi2")
                    people[track_id]["roi2_persistence_time"] = people[track_id]["roi2_persistence_time"]+time_of_persistence/1000.0
                    people[track_id]["start_persistence"] = -1
            
        people[track_id]["prev_roi"] = roi
        people[track_id]["lost_tracking"]=False


    # Handles persistence of tracked people when they are no longer visible
    def get_persitence_for_no_more_tracked_people(self, people, track_ids, timestamp):
        for track_id in people.keys():
            if track_id not in track_ids:
                people[track_id]["lost_tracking"]=True
                if people[track_id]["start_persistence"] != -1:
                    stop_persistence = timestamp
                    time = self.milliseconds_to_hh_mm_ss(timestamp)
                    if self.verbose:
                        print(f"[{time}]: track id {track_id} no more tracked")
                    time_of_persistence=stop_persistence-people[track_id]["start_persistence"]
                    if people[track_id]["prev_roi"] == 1:
                        people[track_id]["roi1_persistence_time"] = people[track_id]["roi1_persistence_time"]+time_of_persistence/1000.0
                    elif people[track_id]["prev_roi"] == 2:
                        people[track_id]["roi2_persistence_time"] = people[track_id]["roi2_persistence_time"]+time_of_persistence/1000.0
                    people[track_id]["start_persistence"] = -1


    # Updates the persistence of tracked people at the end of the tracking process
    def update_persistence(self, people, last_time):
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


    # Converts milliseconds to hh:mm:ss format
    def milliseconds_to_hh_mm_ss(self, milliseconds):
        # Calcola la differenza di tempo in secondi
        seconds = milliseconds / 1000.0
        
        # Crea un oggetto timedelta
        delta = timedelta(seconds=seconds)
        
        # Utilizza l'oggetto timedelta per ottenere la rappresentazione hh:mm:ss
        formatted_time = str(delta)
        
        # Estrai solo la parte hh:mm:ss dalla rappresentazione
        hh_mm_ss = formatted_time.split(".")[0]
        
        return hh_mm_ss


    # Displays the annotated frame with ROIs
    def display_annoted_frame(self,annotated_frame,x_roi1,y_roi1,w_roi1,h_roi1,x_roi2,y_roi2,w_roi2,h_roi2):
        cv.rectangle(annotated_frame, (x_roi1, y_roi1), (x_roi1 + w_roi1, y_roi1 + h_roi1), self.colors[1], 2)
        cv.rectangle(annotated_frame, (x_roi2, y_roi2), (x_roi2 + w_roi2, y_roi2 + h_roi2), self.colors[2], 2)
        cv.imshow("YOLOv8 Tracking", annotated_frame)


    # Saves tracking results to a JSON file
    def save_tracking_results(self, people):
        filtered_people = []
        for person_id, person in people.items():
            filtered_people.append({"id": person_id,
                                    "gender": person["gender"],
                                    "bag": person["bag"],
                                    "hat": person["hat"], 
                                    "upper_color": person["upper_color"],
                                    "lower_color": person["lower_color"],
                                    "roi1_passages": person["roi1_passages"],
                                    "roi1_persistence_time": ceil(person["roi1_persistence_time"]),
                                    "roi2_passages": person["roi2_passages"],
                                    "roi2_persistence_time": ceil(person["roi2_persistence_time"])
                                    })

        data = {"people": filtered_people}


        dir_path = os.path.dirname(self.results_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)    
        
        with open(self.results_path, 'w') as file:
            json.dump(data, file, indent=2)