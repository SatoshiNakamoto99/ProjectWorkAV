import os

def rinomina_video(cartella):
    # Verifica se la cartella esiste
    if not os.path.exists(cartella):
        print(f"La cartella {cartella} non esiste.")
        return

    # Elenco dei file nella cartella
    files = os.listdir(cartella)

    # Filtra solo i file video che non seguono la nomenclatura specifica
    video_files = [file for file in files if file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')) and not file.lower().startswith('video')]

    # Se non ci sono file video nella cartella, esci
    if not video_files:
        print(f"Non ci sono file video nella cartella {cartella} che richiedono la rinominazione.")
        return

    # Trova l'ultimo numero utilizzato
    numeri_utilizzati = [int(file[5:7]) for file in files if file[5:7].isdigit() and file.lower().startswith('video')]
    ultimo_numero = max(numeri_utilizzati, default=0)

    # Rinomina i file video
    for i, video_file in enumerate(video_files, start=ultimo_numero + 1):
        vecchio_nome = os.path.join(cartella, video_file)
        nuovo_nome = os.path.join(cartella, f"video{i:02d}.mp4")  # Numero a due cifre
        
        os.rename(vecchio_nome, nuovo_nome)
        print(f"Rinominato: {vecchio_nome} -> {nuovo_nome}")

if __name__ == "__main__":
    # Inserisci il percorso completo della cartella contenente i video
    cartella_video = "data/video_atrio_cues"
    
    # Chiama la funzione per rinominare i video
    rinomina_video(cartella_video)
