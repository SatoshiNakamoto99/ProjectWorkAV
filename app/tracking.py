import cv2 as cv
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

from app.my_yolo import MyYOLO
from loaders import LoadVideoStream
import settings as settings


RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
WHITE = (255,255,255)
BLACK = (0,0,0)


class ObjectTracker:


    # Initializes the tracking process with the specified YOLO model and video path
    def __init__(self, video_path, configuration_path, results_path, model_path=settings.DETECTION_MODEL, par_module_path=settings.PAR_MODEL , verbose=True):
        self.tracking_model = MyYOLO(model_path)
        self.track_history = defaultdict(lambda: [])
        self.video_path = video_path
        self.configuration_path = configuration_path
        self.results_path = results_path
        self.colors = {0:RED,1:BLUE,2:GREEN,3:WHITE}
        self.verbose = verbose
        video_name = Path(self.video_path).name.split('.')[0]
        folder_path = 'results' + '/' + video_name
        self.par_module_path = par_module_path
        self.par_module = PARModuleInference(self.par_module_path)
        self.actual_detected_person = None
        shutil.rmtree(folder_path, ignore_errors=True)
        self.par_results = {}
        self.final_par_results = {}
        self.FRAME_THRESHOLD = 21
        self.FRAME_DETECTION = 3
        

    # Gets the rescaled Regions of Interest (ROIs) from the video based on a configuration JSON file
    def get_rescaled_rois(self):
        cap = cv.VideoCapture(self.video_path)
        if not cap.isOpened():
            if self.verbose:
                print("\nErrore nell'apertura del video.")
                exit()
        else:
            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

            # Carica il file JSON
            with open(self.configuration_path) as json_file:
                data = json.load(json_file)


            # Estrai le ROIs dal dizionario JSON
            rois = data.values()
            processed_rois = []

            # Disegna i bounding box sull'immagine
            for roi in rois:
                # Riscalare le coordinate e le dimensioni relative a valori assoluti
                relative_x = roi["x"]
                relative_y = roi["y"]
                relative_w = roi["w"]
                relative_h = roi["h"]

                if(relative_x+relative_w > 1):
                    relative_w = 1-relative_x
                if(relative_y+relative_h > 1):
                    relative_h = 1-relative_y

                x = int(relative_x * width)
                y = int(relative_y * height)
                w = int(relative_w * width)
                h = int(relative_h * height)
                processed_rois.append((x,y,w,h))

            return processed_rois


    # Performs the tracking process using the YOLO model
    def perform_tracking(self):
        rois = self.get_rescaled_rois()
        x_roi1,y_roi1,w_roi1,h_roi1 = rois[0]
        x_roi2,y_roi2,w_roi2,h_roi2 = rois[1]

        people = {}    
        last_time = 0

        frame_id = 0
        # Create a video stream loader
        stream_loader = LoadVideoStream(source=self.video_path, fps_out=5)


        try:
            for _, images, _, _ in stream_loader:
                for frame, timestamp in images:

                    tracking_results = self.tracking_model.track(frame, persist=True, conf=0.3, classes=[0], device='cpu', tracker="config/botsort.yaml")

                    frame_id = frame_id+1

                    tracking_annotated_frame = frame.copy()

                    # Get the boxes and track IDs
                    try:
                        boxes = tracking_results[0].boxes.xywh.cpu()
                        track_ids = tracking_results[0].boxes.id.int().cpu().tolist()

                        # Plot the tracks
                        for box, track_id in zip(boxes, track_ids):
                            x, y, w, h = box

                            roi = self.get_roi_of_belonging(x,y,x_roi1,y_roi1,w_roi1,h_roi1,x_roi2,y_roi2,w_roi2,h_roi2)

                            # Disegna il bounding box con il colore appropriato
                            color = self.colors[roi]
                            cv.rectangle(tracking_annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, 2)
                            
                            self.actual_detected_person = frame[ int(y-h/2):int(y+h/2),
                                            int(x-w/2):int(x+w/2) ] 
                            

                            self.update_informations(people, track_id, roi, timestamp)
                            self.put_informations(tracking_annotated_frame, people, track_id, color, x,y,w,h)


                        self.get_persitence_for_no_more_tracked_people(people,track_ids,timestamp)
                        

                        last_time=timestamp


                        people_in_roi = 0
                        total_person = len(track_ids)
                        passages_in_roi1 = 0
                        passages_in_roi2 = 0
                        for value in people.values():
                            if value["start_persistence1"] != -1 or value["start_persistence2"] != -1:
                                people_in_roi = people_in_roi+1
                            passages_in_roi1 = passages_in_roi1 + value["roi1_passages"]
                            passages_in_roi2 = passages_in_roi2 + value["roi2_passages"]


                        cv.rectangle(tracking_annotated_frame, (0,0), (200,70), WHITE, -1)
                        cv.putText(tracking_annotated_frame, 'People in ROI: ' + str(people_in_roi), (10,15), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=BLACK, thickness=1)
                        cv.putText(tracking_annotated_frame, 'Total persons: ' + str(total_person), (10,30), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=BLACK, thickness=1)
                        cv.putText(tracking_annotated_frame, 'Passages in ROI 1: ' + str(passages_in_roi1), (10,45), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=BLACK, thickness=1)
                        cv.putText(tracking_annotated_frame, 'Passages in ROI 2: ' + str(passages_in_roi2), (10,60), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=BLACK, thickness=1)
                        
                        
                        
                        self.display_annoted_frame(tracking_annotated_frame,x_roi1,y_roi1,w_roi1,h_roi1,x_roi2,y_roi2,w_roi2,h_roi2)

                        cv.waitKey(1)
                    except:
                        pass  

        except KeyboardInterrupt:
            print("Keyboard interrupt. Stopping the stream.")
        finally:
            self.update_persistence(people,last_time)
            self.save_tracking_results(people)
            stream_loader.close()

    
    # Returns the Region of Interest (ROI) to which a point (x, y) belongs
    def get_roi_of_belonging(self,x,y,x_roi1,y_roi1,w_roi1,h_roi1,x_roi2,y_roi2,w_roi2,h_roi2):
        # Verifica se il punto (x, y) appartiene a ROI1
        if x_roi1 <= x <= x_roi1 + w_roi1 and y_roi1 <= y <= y_roi1 + h_roi1:
            if x_roi2 <= x <= x_roi2 + w_roi2 and y_roi2 <= y <= y_roi2 + h_roi2:
                roi = 3
            else:    
                roi = 1
        # Verifica se il punto (x, y) appartiene a ROI2
        elif x_roi2 <= x <= x_roi2 + w_roi2 and y_roi2 <= y <= y_roi2 + h_roi2:
            roi = 2
        # Se il punto non appartiene a nessuna ROI, assegna il colore no_roi
        else:
            roi = 0
        return roi
    

    # Handles ROI passages and persistence of tracked people
    def update_informations(self, people, track_id, roi, timestamp):
        if track_id not in people:
            
            self.par_results[track_id] = []
            
            people[track_id] = {
                "gender":" ",
                "bag":" ",
                "hat":" ",
                "upper_color":" ",
                "lower_color":" ",
                "roi1_passages": 0,
                "roi1_persistence_time": 0,
                "roi2_passages": 0,
                "roi2_persistence_time": 0,
                "prev_roi": roi,
                "lost_tracking": False,
                "num_frames": 1
            }
            if roi == 1 or roi == 3:
                people[track_id]["roi1_passages"] = 1
                people[track_id]["start_persistence1"] = timestamp
                people[track_id]["start_persistence2"] = -1
                time = self.milliseconds_to_hh_mm_ss(timestamp)
                if roi == 3:
                    people[track_id]["roi2_passages"] = 1
                    people[track_id]["start_persistence1"] = timestamp
                    people[track_id]["start_persistence2"] = timestamp
                
                if self.verbose:
                    if roi == 1:
                        print(f"[{time}]: track id {track_id} entered roi1")
                    else: 
                        print(f"[{time}]: track id {track_id} entered roi1")
                        print(f"[{time}]: track id {track_id} entered roi2")
            elif roi == 2:
                people[track_id]["roi2_passages"] = 1
                people[track_id]["start_persistence2"] = timestamp
                people[track_id]["start_persistence1"] = -1
                time = self.milliseconds_to_hh_mm_ss(timestamp)
                if self.verbose:
                    print(f"[{time}]: track id {track_id} entered roi2")
            else:
                people[track_id]["start_persistence1"] = -1
                people[track_id]["start_persistence2"] = -1
                
            ### make first prediciton for showing something on screen the first time
            temp_par_results = self.par_module.prediction(self.actual_detected_person)
            people[track_id]["gender"] = temp_par_results["gender"]
            people[track_id]["hat"] = temp_par_results["hat"]
            people[track_id]["bag"] = temp_par_results["bag"]
            people[track_id]["upper_color"] = temp_par_results["upper_color"]
            people[track_id]["lower_color"] = temp_par_results["lower_color"]
            

        else:
                        
            time = self.milliseconds_to_hh_mm_ss(timestamp)

            ### roi tracking
            prev_roi = people[track_id]["prev_roi"]
            prev_roi_is_roi1 = False
            prev_roi_is_roi2 = False
            if prev_roi == 3:
                prev_roi_is_roi1 = True
                prev_roi_is_roi2 = True
            elif prev_roi == 1:
                prev_roi_is_roi1 = True
            elif prev_roi == 2:
                prev_roi_is_roi2 = True


            if (roi == 1 or roi == 3) and (not prev_roi_is_roi1 or people[track_id]["lost_tracking"]):
                if self.verbose:
                    print(f"[{time}]: track id {track_id} entered roi1")
                people[track_id]["roi1_passages"] = people[track_id]["roi1_passages"] + 1
                people[track_id]["start_persistence1"] = timestamp
            if (roi == 2 or roi == 3) and (not prev_roi_is_roi2 or people[track_id]["lost_tracking"]):
                if self.verbose:
                    print(f"[{time}]: track id {track_id} entered roi2")
                people[track_id]["roi2_passages"] = people[track_id]["roi2_passages"] + 1
                people[track_id]["start_persistence2"] = timestamp
            if((roi != 1 and roi != 3 and prev_roi_is_roi1) or (roi != 2 and roi != 3 and prev_roi_is_roi2)):
                stop_persistence = timestamp
                if prev_roi_is_roi1 and people[track_id]["roi1_persistence_time"] != -1:
                    if self.verbose:
                        print(f"[{time}]: track id {track_id} exited roi1")
                    time_of_persistence=stop_persistence-people[track_id]["start_persistence1"]
                    people[track_id]["roi1_persistence_time"] = people[track_id]["roi1_persistence_time"]+time_of_persistence/1000.0
                    people[track_id]["start_persistence1"] = -1
                elif prev_roi_is_roi2 and people[track_id]["roi2_persistence_time"] != -1:
                    if self.verbose:
                        print(f"[{time}]: track id {track_id} exited roi2")
                    time_of_persistence=stop_persistence-people[track_id]["start_persistence2"]
                    people[track_id]["roi2_persistence_time"] = people[track_id]["roi2_persistence_time"]+time_of_persistence/1000.0
                    people[track_id]["start_persistence2"] = -1
            
            ### model prediction at each frame interval
            if people[track_id]["num_frames"] <= self.FRAME_THRESHOLD:
                if people[track_id]["num_frames"] % self.FRAME_DETECTION == 0:
                    self.par_results[track_id].append(self.par_module.prediction(self.actual_detected_person))  # output del modello è un dict

                if people[track_id]["num_frames"] == self.FRAME_THRESHOLD:
                    # performs majority voting for the self.FRAME_THRESHOLD/self.FRAME_DETECTION frames
                    self.majority_voting(track_id)

                people[track_id]["num_frames"] += 1
            
        people[track_id]["prev_roi"] = roi
        people[track_id]["lost_tracking"]=False

    def most_common(self, lst):
        return max(set(lst), key=lst.count)

    # performs majority voting for multiple prediction from parmodule for a single person with his own track_id
    def majority_voting(self,track_id):
        # self.par_results[track_id] è una lista di results
        
        gender_list = []
        hat_list = []
        bag_list = []
        upper_color_list = []
        lower_color_list = []

        for results in self.par_results[track_id]:
            gender_list.append(results["gender"])
            hat_list.append(results["hat"])
            bag_list.append(results["bag"])
            upper_color_list.append(results["upper_color"])
            lower_color_list.append(results["lower_color"])
            
        self.final_par_results[track_id] = {}
        
        self.final_par_results[track_id]["gender"] = self.most_common(gender_list)
        self.final_par_results[track_id]["hat"] = self.most_common(hat_list)
        self.final_par_results[track_id]["bag"] = self.most_common(bag_list)
        self.final_par_results[track_id]["upper_color"] = self.most_common(upper_color_list)
        self.final_par_results[track_id]["lower_color"] = self.most_common(lower_color_list)
        

    def put_informations(self, tracking_annotated_frame, people, track_id, color, x,y,w,h):
        # fino a quando non si ha la predizione ufficiale
        if people[track_id]["num_frames"] >= 2 and people[track_id]["num_frames"] <= self.FRAME_THRESHOLD:
            # cv.rectangle(tracking_annotated_frame, (x-10,y), (x+10,y+50), WHITE, -1)
            attributes_string = " id: " +  str(track_id) + "\n Gender: " 
            attributes_string = attributes_string + 'M' if people[track_id]["gender"] == 'male' else attributes_string + 'F'
            attributes_string = attributes_string +  "\n bag: " + people[track_id]["bag"] + "\n hat: " + people[track_id]["hat"] + "\n U-L: " + people[track_id]["upper_color"] + "-" + people[track_id]["lower_color"]    
            y0 = y
            dy = 10
            for i, line in enumerate(attributes_string.split('\n')):
                y = y0 + i*dy
                cv.putText(tracking_annotated_frame, line, (int(x+w/2), int(y+h/2)), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=BLACK, thickness=1)
            
        # quando si ha predizione ufficiale
        if people[track_id]["num_frames"] == self.FRAME_THRESHOLD + 1:
            
            people[track_id]["gender"] = self.final_par_results[track_id]["gender"]
            people[track_id]["bag"] = self.final_par_results[track_id]["bag"]
            people[track_id]["hat"] = self.final_par_results[track_id]["hat"]
            people[track_id]["upper_color"] = self.final_par_results[track_id]["upper_color"]
            people[track_id]["lower_color"] = self.final_par_results[track_id]["lower_color"]
            
            # cv.rectangle(tracking_annotated_frame, (x-10,y), (x+10,y+50), WHITE, -1)
            # cv.putText(tracking_annotated_frame, str(track_id) , (x+10,y+10), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=BLACK, thickness=1)
            attributes_string = " id: " +  str(track_id) + "\n Gender: " 
            attributes_string = attributes_string + 'M' if people[track_id]["gender"] == 'male' else attributes_string + 'F'
            attributes_string = attributes_string +  "\n bag: " + people[track_id]["bag"] + "\n hat: " + people[track_id]["hat"] + "\n U-L: " + people[track_id]["upper_color"] + "-" + people[track_id]["lower_color"]    
            y0 = y
            dy = 10
            for i, line in enumerate(attributes_string.split('\n')):
                y = y0 + i*dy
                cv.putText(tracking_annotated_frame, line, (int(x+w/2), int(y+h/2)), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=BLACK, thickness=1)
        


    # Handles persistence of tracked people when they are no longer visible
    def get_persitence_for_no_more_tracked_people(self, people, track_ids, timestamp):
        for track_id in people.keys():
            if track_id not in track_ids:
                people[track_id]["lost_tracking"]=True
                if people[track_id]["start_persistence1"] != -1 or people[track_id]["start_persistence2"] != -1:
                    stop_persistence = timestamp
                    time = self.milliseconds_to_hh_mm_ss(timestamp)
                    if self.verbose:
                        print(f"[{time}]: track id {track_id} no more tracked")
                    
                    prev_roi = people[track_id]["prev_roi"]
                    if prev_roi == 1 or prev_roi == 3:
                        time_of_persistence=stop_persistence-people[track_id]["start_persistence1"]
                        if(time_of_persistence < 1000):
                            time_of_persistence = 1000
                        people[track_id]["roi1_persistence_time"] = people[track_id]["roi1_persistence_time"]+time_of_persistence/1000.0
                    if prev_roi == 2 or prev_roi == 3:
                        time_of_persistence=stop_persistence-people[track_id]["start_persistence2"]
                        if(time_of_persistence < 1000):
                            time_of_persistence = 1000
                        people[track_id]["roi2_persistence_time"] = people[track_id]["roi2_persistence_time"]+time_of_persistence/1000.0
                    people[track_id]["start_persistence1"] = -1
                    people[track_id]["start_persistence2"] = -1


    # Updates the persistence of tracked people at the end of the tracking process
    def update_persistence(self, people, last_time):
        for track_id in people.keys():
            if people[track_id]["start_persistence1"] != -1:
                stop_persistence = last_time
                time_of_persistence=stop_persistence-people[track_id]["start_persistence1"]
                if(time_of_persistence < 1000):
                    time_of_persistence = 1000
                people[track_id]["roi1_persistence_time"] = people[track_id]["roi1_persistence_time"]+time_of_persistence/1000.0
                people[track_id]["start_persistence1"] = -1
            if people[track_id]["start_persistence2"] != -1:
                stop_persistence = last_time
                time_of_persistence=stop_persistence-people[track_id]["start_persistence2"]
                if(time_of_persistence < 1000):
                    time_of_persistence = 1000
                people[track_id]["roi2_persistence_time"] = people[track_id]["roi2_persistence_time"]+time_of_persistence/1000.0
                people[track_id]["start_persistence2"] = -1    



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
        cv.rectangle(annotated_frame, (x_roi1, y_roi1), (x_roi1 + w_roi1, y_roi1 + h_roi1), BLACK, 2)
        cv.putText(annotated_frame, "1", (x_roi1+10, y_roi1+30), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=BLACK, thickness=3)
        cv.rectangle(annotated_frame, (x_roi2, y_roi2), (x_roi2 + w_roi2, y_roi2 + h_roi2), BLACK, 2)
        cv.putText(annotated_frame, "2", (x_roi2+10, y_roi2+30), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=BLACK, thickness=3)
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