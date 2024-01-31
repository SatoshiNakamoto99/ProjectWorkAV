import os



def modifica_file_txt(folder_path):
    # Itera attraverso tutti i file nella cartella
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)

            # Apri il file in modalità di lettura
            with open(filepath, 'r') as file:
                # Leggi tutte le righe dal file
                lines = file.readlines()

            # Apri il file in modalità di scrittura sovrascrivendo i valori
            with open(filepath, 'w') as file:
                for line in lines:
                    # Suddividi la riga in valori separati da virgole
                    values = line.strip().split(',')

                    # Modifica i valori come richiesto
                    values[0] = str(int(float(values[0])))
                    values[1] = str(int(float(values[1])) - 1)
                    #values[-5:] = ['-1'] * 4

                    # Scrivi la riga aggiornata nel file
                    file.write(','.join(values) + '\n')

# Specifica il percorso della cartella contenente i file
cartella = '/home/saramassaro/ProjectWork/TrackEval/data/trackers/mot_challenge/MOT17-train/bytetrack/data'
modifica_file_txt(cartella)

cartella = '/home/saramassaro/ProjectWork/TrackEval/data/trackers/mot_challenge/MOT17-train/botsort/data'
modifica_file_txt(cartella)
