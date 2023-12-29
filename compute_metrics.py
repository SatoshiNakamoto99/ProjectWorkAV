import numpy as np

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
    for image_name, prediction in predicted_data.items():
        if true_data[image_name] == prediction:
            true_positive = true_positive + 1
        else:
            print(f'In image {image_name} was predicted {colors_to_string(prediction)} but true colow is {colors_to_string(true_data[image_name])}')

    accuracy = true_positive / len(predicted_data) if len(predicted_data) > 0 else 0

    return accuracy

def main():
    # Replace 'file1.txt' and 'file2.txt' with the actual file paths
    file1_path = 'datasets/PAR/annotations/validation_set.txt'
    file2_path = 'PARresults.txt'

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

if __name__ == "__main__":
    main()