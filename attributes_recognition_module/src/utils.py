
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_metrics(num_epochs,task_names, task_metrics, overall_metrics):
    print(f"Epoch {num_epochs} Metrics:")
    # Visualize metrics for each task
    for task in task_names:
        print(f"{task} Metrics:")
        print(f"Loss: {task_metrics[task]['Loss']:.4f}, Accuracy: {task_metrics[task]['Accuracy']:.4f}, Precision: {task_metrics[task]['Precision']:.4f}, Recall: {task_metrics[task]['Recall']:.4f}, Accuracy_Personal: {task_metrics[task]['Accuracy_Personal']:.4f}")
        print()

    # Visualize overall metrics
    print("Overall Metrics:")
    print(f"Loss: {overall_metrics['Loss']:.4f}, Accuracy: {overall_metrics['Accuracy']:.4f}, Precision: {overall_metrics['Precision']:.4f}, Recall: {overall_metrics['Recall']:.4f}, Accuracy_Personal: {overall_metrics['Accuracy_Personal']:.4f}")

def save_metrics_to_csv(num_epochs, task_names, task_metrics, overall_metrics, train_loss ,csv_filename='metrics.csv'):
    
    # Create a DataFrame to store metrics
    
    df = pd.DataFrame(columns=['Epoch','Task', 'Loss', 'Accuracy', 'Precision', 'Recall'])
    
      # Add task metrics to DataFrame
    i = 0
    for task in task_names:
        df.loc[i] = [num_epochs, task, task_metrics[task]['Loss'], task_metrics[task]['Accuracy'], task_metrics[task]['Precision'], task_metrics[task]['Recall']]
        i += 1
        
    
    
    # Add overall metrics to DataFrame
    df.loc[i] = [num_epochs, 'Total Val', overall_metrics['Loss'], overall_metrics['Accuracy'], overall_metrics['Precision'], overall_metrics['Recall']]
    i = i+1
    df.loc[i] = [num_epochs, 'Total Train', train_loss, '-', '-', '-']
    i = i+1
    # Add a blank row
    df.loc[i] = ['', '', '', '', '', '']

    # Save DataFrame to CSV if the csv file exists it write the new metrics in the same file but not overwrite the old metrics
    if os.path.exists(csv_filename):
        df.to_csv(csv_filename, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_filename, index=False)
    print(f"Metrics saved to {csv_filename}")

def calculate_metrics(labels, predictions):
    true_positives = 0
    #compute true positives consider that if the label is equal to the prediction is a true positive
    # label e prediction are related to a batch
    for label, prediction in zip(labels, predictions):
        if label == prediction:
            true_positives += 1
    total_samples = len(labels)
    accuracy_personal = true_positives / total_samples

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    return accuracy, precision, recall, accuracy_personal


# function that give a path if is not exists create it
def create_path(path):
    if not os.path.exists(path):
        print("Creating directory: " + path)
        os.makedirs(path)
    else:
        print("Directory already exists: " + path)
    return path