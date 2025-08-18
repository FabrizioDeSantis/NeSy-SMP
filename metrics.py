import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
import torch
import seaborn as sns

def compute_accuracy(loader, model, device):
    y_pred = []
    y_true = []
    for data, labels in loader:
        data = data.to(device)
        predictions = model(data).detach().cpu().numpy()
        predictions = np.where(predictions > 0.5, 1., 0.).flatten()
        for i in range(len(labels)):
            y_pred.append(predictions[i])
            y_true.append(labels[i].cpu())
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def compute_highlactate(loader, model, device, scalers):
    lactate = lambda x: x[:, 180:210]
    lactate_2 = lambda x: (x[:, 180:210] > scalers["Lactate"].transform([[2.0]])[0][0]).any(dim=1)
    for data, labels in loader: 
        data = data.to(device)
        print("--------------")
        print(lactate_2(data))
        predictions = model(lactate(data)).detach().cpu().numpy()
        print(predictions)
        print("--------------")

def compute_metrics(loader, model, device, mode, scalers, features_dict, sequence_length):
    
    y_pred = []
    y_true = []
    y_scores = []
    case_ids = []
    for data, labels, c_id in loader:
        data = data.to(device)
        predictions = model(data).detach().cpu().numpy()
        predictions_th = np.where(predictions > 0.5, 1., 0.).flatten()
        for i in range(len(labels)):
            y_scores.append(predictions[i])
            # if labels[i].cpu() == 0 and predictions_th[i] == 1:
            #     print("-----------------")
            #     print(f"False positive for case {c_id[i]}: Predicted {predictions[i]}, True {labels[i].cpu()}")
            #     print(f"Lactate: {lactate_res[i]}")
            #     print(f"Albumin: {albumin_res[i]}")
            #     print(f"Albumin Measured: {albumin_measured_res[i]}")
            #     print(f"Respiratory Rate: {respiratory_rate_res[i]}")
            #     print(f"ABPS: {abps_res[i]}")
            #     print("-----------------")
            y_pred.append(predictions_th[i])
            y_true.append(labels[i].cpu())
    
    accuracy = accuracy_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    auroc_value = roc_auc_score(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if mode == "lstm":
        plt.savefig('confusion_matrix.png')
    elif mode == "ltn":
        plt.savefig('confusion_matrix_ltn.png')
    elif mode == "ltn_w_k":
        plt.savefig('confusion_matrix_ltn_w_k.png')
    elif mode == "ltn_w_k_now":
        plt.savefig('confusion_matrix_ltn_w_k_now.png')
    plt.close()
    
    return accuracy, f1score, precision, recall, auroc_value

def compute_predicate_values(loader, model, device, features_dict, sequence_length):
    lactate = lambda x: x[:,(features_dict["Lactate"]*sequence_length-sequence_length):(features_dict["Lactate"]*sequence_length)]
    age = lambda x: x[:,(features_dict["anchor_age"]*sequence_length-sequence_length):(features_dict["anchor_age"]*sequence_length)][:, 0]
    comorbidities = lambda x: x[:, -23:]
    for data, labels, c_id in loader:
        data = data.to(device)
        predictions = model(lactate(data), comorbidities(data), age(data)).detach().cpu().numpy()
        for i in range(len(labels)):
            print(f"Case ID: {c_id[i]}, Predicted Value: {predictions[i]}, True Label: {labels[i].cpu().item()}")