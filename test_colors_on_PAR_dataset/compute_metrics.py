import numpy as np
import matplotlib.pyplot as plt

def colors_to_string(value):
    colors = {'1':'black', '2':'blue', '3':'brown', '4':'gray', '5':'green', '6':'orange', '7':'pink', '8':'purple', '9':'red', '10':'white', '11':'yellow'}
    return colors[value]

def read_file(file_path):
    upper = {}
    lower = {}
    with open(file_path, 'r') as file:
        for line in file:
            elements = line.strip().split(',')
            upper[elements[0].strip()] = (elements[1].strip())
            lower[elements[0].strip()] = (elements[2].strip())
    return upper,lower

def calculate_metrics(true_data, predicted_data):

    true_positive = 0
    for image_name, truth in true_data.items():
        try:
            if predicted_data[image_name] == truth:
                true_positive = true_positive + 1
        except:
            pass
        # else:
        #     print(f'In image {image_name} was predicted {colors_to_string(prediction)} but true colow is {colors_to_string(true_data[image_name])}')

    accuracy = true_positive / len(predicted_data) if len(predicted_data) > 0 else 0

    return accuracy

def get_values(true_colors, predicted_colors):
    valori1 = [0]*11
    valori2 = [0]*11
    for image_name, label in true_colors.items():
        valori1[int(label)-1] += 1
        try:
            if predicted_colors[image_name] == label:
                valori2[int(label)-1] += 1
        except:
            pass

    return valori1, valori2
            
def main():
    # Replace 'file1.txt' and 'file2.txt' with the actual file paths
    file1_path = 'datasets/PAR/annotations/validation_set.txt'
    file2_path = 'test_colors_on_PAR_dataset/PARresults2.txt'

    # Read data from the files
    true_upper_colors, true_lower_colors = read_file(file1_path)
    predicted_upper_colors, predicted_lower_colors = read_file(file2_path)

    # Calculate metrics for upper colors
    accuracy_upper = calculate_metrics(true_upper_colors, predicted_upper_colors)

    # Calculate metrics for lower colors
    accuracy_lower = calculate_metrics(true_lower_colors, predicted_lower_colors)

    # Print the results
    print(f"Metrics for Upper Colors:")
    print(f"Accuracy: {accuracy_upper:.2f} on a total of {len(predicted_upper_colors)} predictions")

    print("\n")

    print(f"Metrics for Lower Colors:")
    print(f"Accuracy: {accuracy_lower:.2f} on a total of {len(predicted_lower_colors)} predictions")

    # Lista di valori
    valori1_upper, valori2_upper = get_values(true_upper_colors, predicted_upper_colors)
    valori1_lower, valori2_lower = get_values(true_lower_colors, predicted_lower_colors)

    etichette = ['black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']

    # Posizioni sull'asse x per ciascun gruppo di barre
    posizioni = np.arange(len(etichette))

    # Larghezza delle barre
    larghezza_barre = 0.35

    # Creazione di una figura con due subplot divisi verticalmente
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Plot per la parte superiore
    ax1.bar(posizioni - larghezza_barre/2, valori1_upper, larghezza_barre, label='Labels')
    ax1.bar(posizioni + larghezza_barre/2, valori2_upper, larghezza_barre, label='Corrected predictions')
    ax1.set_title('Upper Colors results')
    ax1.set_xticks(posizioni)
    ax1.set_xticklabels(etichette, rotation='vertical')
    ax1.legend()

    # Plot per la parte inferiore
    ax2.bar(posizioni - larghezza_barre/2, valori1_lower, larghezza_barre, label='Labels')
    ax2.bar(posizioni + larghezza_barre/2, valori2_lower, larghezza_barre, label='Corrected predictions')
    ax2.set_title('Lower Colors results')
    ax2.set_xticks(posizioni)
    ax2.set_xticklabels(etichette, rotation='vertical')
    ax2.legend()

    # Aggiungi un titolo per l'intera figura
    fig.suptitle('Results on PAR dataset')

    # Imposta uno spazio tra i due subplot
    plt.tight_layout()

    # Visualizzazione della figura
    plt.show()



if __name__ == "__main__":
    main()
