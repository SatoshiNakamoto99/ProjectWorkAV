import cv2 as cv
import os
import sys
from pathlib import Path

import numpy as np

from tqdm import tqdm

from ultralytics.utils import LOGGER
LOGGER.setLevel("WARNING")


current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, ".."))
sys.path.append(project_path)
from my_yolo import MyYOLO



def find_nearest_color(rgb_tuple):
    #color_names = ['black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']
    color_names = [1,2,3,4,5,6,7,8,9,10,11]
    color_values = [(0, 0, 0), (0, 0, 255), (165, 42, 42), (128, 128, 128), (0, 128, 0), (255, 165, 0),
                    (255, 192, 203), (128, 0, 128), (255, 0, 0), (255, 255, 255), (255, 255, 0)]

    min_distance = float('inf')
    nearest_color = None

    for name, value in zip(color_names, color_values):
        distance = sum((a - b) ** 2 for a, b in zip(rgb_tuple, value))
        if distance < min_distance:
            min_distance = distance
            nearest_color = name

    return nearest_color

# Estimates the predominant color in a pixel region using color analysis
def estimate_predominant_color(pixel_region,mask):
    # Resizing parameters

    image_array = np.array(pixel_region)
    mask_array = np.array(mask)
    
    # Espandi le dimensioni della maschera
    mask_array = np.expand_dims(mask_array, axis=-1)

    # Creazione di un array risultante con pixel dell'immagine originale dove la maschera è 255
    result_array = np.where(mask_array == 255, image_array, 0)

    # Ottieni i colori dall'array risultante
    pixels = [tuple(pixel) for pixel in result_array.reshape(-1, 3) if not all(value == 0 for value in pixel)]

    # Ordina i pixel per conteggio (primo elemento della tupla)
    sorted_pixels = sorted(pixels, key=lambda t: pixels.count(t))

    # Verifica se ci sono pixel nella lista ordinata prima di accedere all'indice -1
    if sorted_pixels:
        # Ottieni il colore dominante
        dominant_color = sorted_pixels[-1]

        # Converti il colore dominante nel colore più vicino
        nearest_color = find_nearest_color(dominant_color)

        return nearest_color
    else:
        # Nessun pixel valido trovato, restituisci un valore predefinito o gestisci il caso come appropriato
        return "black"


if __name__ == "__main__":

    # Percorsi delle immagini e della cartella dei risultati
    input_path = Path("datasets/PAR/validation_set")
    output_path = Path("test_colors_on_PAR_dataset/resultsPAR")
    results_path = 'test_colors_on_PAR_dataset/PARresults.txt'

    # Crea la cartella dei risultati se non esiste
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Lista di tutti i file nella cartella di input
    image_files = [f for f in os.listdir(input_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    model_path = 'models/yolov8n-seg.pt'
    tracking_model = MyYOLO(model_path)

    for image_file in tqdm(image_files,desc='Processing PAR Dataset'):
        # Costruisci il percorso completo del file
        image_path = Path(os.path.join(input_path, image_file))
        if not os.path.exists(str(output_path) + '/' + str(image_path.name.split('.')[0])):
            os.makedirs(str(output_path) + '/' + str(image_path.name.split('.')[0]))

        image = cv.imread(image_path)

        tracking_results = tracking_model.predict(image,tracker="config/botsort.yaml")
        for r in tracking_results:
                img = np.copy(r.orig_img)

                # iterate each object contour
                for _, c in enumerate(r):

                    if c.boxes is not None:
                        #label = c.names[c.boxes.cls.tolist().pop()]

                        b_mask = np.zeros(img.shape[:2], np.uint8)

                        # Create contour mask
                        contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                        _ = cv.drawContours(b_mask, [contour], -1, (255, 255, 255), cv.FILLED)

                        # Retrieve bounding box coordinates
                        x, y, w, h = c.boxes.xywh.cpu().tolist()[0]

                        # Crop the region of interest (ROI) based on the bounding box
                        roi = img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
                        
                        # Extract pixels from the original image based on the mask
                        mask=b_mask[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
                        masked_pixels = cv.bitwise_and(roi, roi, mask=mask)

                        # cv.imshow('1',masked_pixels)

                        upper_pixels = masked_pixels[int(0.3*h/2):int(0.9*h/2), :]
                        # cv.imshow('2',upper_pixels)
                        lower_pixels = masked_pixels[int(1.3*h/2):int(1.7*h/2), :]
                        # cv.imshow('3',lower_pixels)
                        # cv.waitKey(0)
                        # cv.destroyAllWindows()

                        upper_color = estimate_predominant_color(upper_pixels,mask[int(0.3*h/2):int(0.9*h/2), :])
                        lower_color = estimate_predominant_color(lower_pixels,mask[int(1.3*h/2):int(1.7*h/2), :])

                        output_file_path = str(output_path) + '/' + str(image_path.name.split('.')[0]) + '/image.jpg'
                        cv.imwrite(output_file_path, masked_pixels)

                        output_file_path = str(output_path) + '/' + str(image_path.name.split('.')[0]) + '/upper.jpg'
                        cv.imwrite(output_file_path, upper_pixels)

                        output_file_path = str(output_path) + '/' + str(image_path.name.split('.')[0]) + '/lower.jpg'
                        cv.imwrite(output_file_path, lower_pixels)
                            
                        

                        # Scrivi le informazioni nel file PARresults.txt
                        
                        with open(results_path, "a") as result_file:
                            result_file.write(f"{image_path.name},{upper_color},{lower_color}\n")
