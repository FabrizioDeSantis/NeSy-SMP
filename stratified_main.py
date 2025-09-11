import pandas as pd
from sqlalchemy import values
import ltn
import torch
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
from model.models import LSTMModel, MLP, SimpleMLP, SimpleMLPAge
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import statistics
from collections import Counter
from data.preprocessing import preprocess_eventlog
from data.dataset import SepsisDataset, ModelConfig
from metrics import compute_metrics
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
from xgboost import XGBClassifier

import argparse

import warnings
warnings.filterwarnings("ignore")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_args():
    parser = argparse.ArgumentParser()

    # general network parameters
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size of the LSTM model")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the LSTM model")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for the LSTM model")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--num_epochs_nesy", type=int, default=20, help="Number of epochs for training LTN model")
    # training configuration
    parser.add_argument("--train_vanilla", type=bool, default=True, help="Train vanilla LSTM model")
    parser.add_argument("--train_nesy", type=bool, default=True, help="Train LTN model")
    parser.add_argument("--dataset_size", type=float, default=30, help="Size of the dataset (10%, 20%, 50%, 70%, 90%, 100%)")
    parser.add_argument("--sampling", type=bool, default=False, help="Train LTN model")

    return parser.parse_args()

args = get_args()

seed = 42
random.seed(seed)

classes = ["No Death", "Death"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("-- INFO: reading dataset")
data = pd.read_csv("data/subset/events_48h_before_death_gcs.csv", dtype={"hadm_id": str, "subject_id": str})
print(data.columns)
columns_to_drop = ["SOFA Score", "APACHEII-Renal failure", "Urine output_ApacheIV", "APACHE II"]
data = data[~data["concept:name"].isin(columns_to_drop)]

(X_all, y_all, feature_names), vocab_sizes, scalers, sequence_length = preprocess_eventlog(data, seed, args.sampling)

config = ModelConfig(
    hidden_size=args.hidden_size,
    num_layers=args.num_layers,
    dropout_rate=args.dropout_rate,
    sequence_length=sequence_length,
    num_epochs = args.num_epochs
)

features_dict = {s: i for i, s in enumerate(feature_names, start=1)}
print(feature_names)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
skf.get_n_splits(X_all, y_all)

lstm_accuracies = []
lstm_f1_scores = []
lstm_precisions = []
lstm_recalls = []
lstm_roc_aucs = []

ltn_accuracies = []
ltn_f1_scores = []
ltn_precisions = []
ltn_recalls = []
ltn_roc_aucs = []

ltn_log_accuracies = []
ltn_log_f1_scores = []
ltn_log_precisions = []
ltn_log_recalls = []
ltn_log_roc_aucs = []

ltn_now_accuracies = []
ltn_now_f1_scores = []
ltn_now_precisions = []
ltn_now_recalls = []
ltn_now_roc_aucs = []

xgb_accuracies = []
xgb_f1_scores = []
xgb_precisions = []
xgb_recalls = []
xgb_roc_aucs = []

rf_accuracies = []
rf_f1_scores = []
rf_precisions = []
rf_recalls = []
rf_roc_aucs = []

for i, (train_index, test_index) in enumerate(skf.split(X_all, y_all)):

    print(f"Fold {i+1}")
    print(train_index)
    X_train = [X_all[i] for i in train_index]
    X_test = [X_all[i] for i in test_index]
    y_train = [y_all[i] for i in train_index]
    y_test = [y_all[i] for i in test_index]

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, stratify=y_train, random_state=seed)

    X_train_xgb = []
    y_train_xgb = []

    X_val_xgb = []
    y_val_xgb = []

    X_test_xgb = []
    y_test_xgb = []

    def mean_excluding_zeros(values):
        non_zero_values = [v for v in values if v != 0.0]
        if not non_zero_values:
            return 0.0
        return sum(non_zero_values) / len(non_zero_values)
    
    def std_excluding_zeros(values):
        non_zero_values = [v for v in values if v != 0.0]
        if not non_zero_values:
            return 0.0
        mean = sum(non_zero_values) / len(non_zero_values)
        variance = sum((v - mean) ** 2 for v in non_zero_values) / len(non_zero_values)
        return math.sqrt(variance)

    for x, y in zip(X_train, y_train):
        features = [lst for i, lst in enumerate(x[0]) if i not in (0, 1, 2, 3)]
        features = [mean_excluding_zeros(lst) for lst in features]
        features_std = [std_excluding_zeros(lst) for lst in features]
        features = features + features_std
        X_train_xgb.append(np.array(features))
        y_train_xgb.append(y)

    for x, y in zip(X_val, y_val):
        features = [lst for i, lst in enumerate(x[0]) if i not in (0, 1, 2, 3)]
        features = [mean_excluding_zeros(lst) for lst in features]
        features_std = [std_excluding_zeros(lst) for lst in features]
        features = features + features_std
        X_val_xgb.append(np.array(features))
        y_val_xgb.append(y)

    for x, y in zip(X_test, y_test):
        features = [lst for i, lst in enumerate(x[0]) if i not in (0, 1, 2, 3)]
        features = [mean_excluding_zeros(lst) for lst in features]
        features_std = [std_excluding_zeros(lst) for lst in features]
        features = features + features_std
        X_test_xgb.append(np.array(features))
        y_test_xgb.append(y)
    
    print("--- Training XGBoost model")
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=seed)
    xgb_model.fit(X_train_xgb, y_train_xgb)
    # predict on test set
    y_pred_xgb = xgb_model.predict(X_test_xgb)
    # predict probabilities
    y_proba_xgb = xgb_model.predict_proba(X_test_xgb)[:, 1]
    # calculate metrics
    accuracy_xgb = accuracy_score(y_test_xgb, y_pred_xgb)
    f1_xgb = f1_score(y_test_xgb, y_pred_xgb, average='macro')
    precision_xgb = precision_score(y_test_xgb, y_pred_xgb, average='macro')
    recall_xgb = recall_score(y_test_xgb, y_pred_xgb, average='macro')
    roc_auc_xgb = roc_auc_score(y_test_xgb, y_proba_xgb)
    print("Metrics XGBoost")
    print("Accuracy:", accuracy_xgb)
    xgb_accuracies.append(accuracy_xgb)
    print("F1 Score:", f1_xgb)
    xgb_f1_scores.append(f1_xgb)
    print("Precision:", precision_xgb)
    xgb_precisions.append(precision_xgb)
    print("Recall:", recall_xgb)
    xgb_recalls.append(recall_xgb)
    print("ROC AUC:", roc_auc_xgb)
    xgb_roc_aucs.append(roc_auc_xgb)

    print("--- Training Random Forest model")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train_xgb, y_train_xgb)
    y_pred = rf.predict(X_test_xgb)
    probs = rf.predict_proba(X_test_xgb)[:, 1]
    accuracy_rf = accuracy_score(y_test_xgb, y_pred)
    f1_rf = f1_score(y_test_xgb, y_pred, average='macro')
    precision_rf = precision_score(y_test_xgb, y_pred, average='macro')
    recall_rf = recall_score(y_test_xgb, y_pred, average='macro')
    roc_auc_rf = roc_auc_score(y_test_xgb, probs)
    print("Metrics Random Forest")
    print("Accuracy:", accuracy_rf)
    rf_accuracies.append(accuracy_rf)
    print("F1 Score:", f1_rf)
    rf_f1_scores.append(f1_rf)
    print("Precision:", precision_rf)
    rf_precisions.append(precision_rf)
    print("Recall:", recall_rf)
    rf_recalls.append(recall_rf)
    print("ROC AUC:", roc_auc_rf)
    rf_roc_aucs.append(roc_auc_rf)

    train_dataset = SepsisDataset(X_train, y_train, feature_names)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = SepsisDataset(X_val, y_val, feature_names)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataset = SepsisDataset(X_test, y_test, feature_names)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("--- Label distribution")
    print("--- Training set")
    counts = Counter(y_train)
    print(counts)
    print("--- Test set")
    counts = Counter(y_test)
    print(counts)

    # Knowledge Theory
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())

    lstm = LSTMModel(vocab_sizes, config, 1, feature_names).to(device)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=config.learning_rate)
    criterion = torch.nn.BCELoss()

    lstm.train()
    val_losses_epochs = []
    best_f1_val = 0
    count_stop = 0
    for epoch in range(config.num_epochs):
        train_losses = []
        val_losses = []
        for enum, (x, y, c_id) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            output = lstm(x)
            loss = criterion(output.squeeze(1).cpu(), y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        lstm.eval()
        with torch.no_grad():
            val_losses = []
            y_pred = []
            y_true = []
            for x, y, c_id in val_loader:
                x = x.to(device)
                output = lstm(x)
                loss = criterion(output.squeeze(1).cpu(), y)
                predictions = np.where(output.detach().cpu().numpy() > 0.5, 1., 0.).flatten()
                val_losses.append(loss.item())
                for i in range(len(y)):
                    y_pred.append(predictions[i])
                    y_true.append(y[i].cpu())
        f1_val = f1_score(y_true, y_pred, average='macro')
        print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {statistics.mean(train_losses)}")
        print("Validation Loss:", statistics.mean(val_losses))
        if f1_val > best_f1_val:
            best_f1_val = f1_val
            torch.save(lstm.state_dict(), "lstm_best.pth")
            count_stop = 0
        else:
            count_stop += 1
            if epoch >= 10 and count_stop > 15:
                print("Early stopping at epoch", epoch+1)
                break
        lstm.train()

    lstm.eval()
    y_pred = []
    y_true = []
    y_scores = []
    for enum, (x, y, c_id) in enumerate(test_loader):
        with torch.no_grad():
            x = x.to(device)
            outputs = lstm(x).detach().cpu().numpy()
            predictions_th = np.where(outputs > 0.5, 1., 0.).flatten()
            for i in range(len(y)):
                y_scores.append(outputs[i])
                y_pred.append(predictions_th[i])
                y_true.append(y[i].cpu())

    print("Metrics LSTM")
    accuracy = accuracy_score(y_true, y_pred)
    lstm_accuracies.append(accuracy)
    print("Accuracy:", accuracy)
    f1 = f1_score(y_true, y_pred, average='macro')
    lstm_f1_scores.append(f1)
    print("F1 Score:", f1)
    precision = precision_score(y_true, y_pred, average='macro')
    lstm_precisions.append(precision)
    print("Precision:", precision)
    recall = recall_score(y_true, y_pred, average='macro')
    lstm_recalls.append(recall)
    print("Recall:", recall)
    roc_auc = roc_auc_score(y_true, y_scores)
    lstm_roc_aucs.append(roc_auc)
    print("ROC AUC:", roc_auc)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()


    lstm = LSTMModel(vocab_sizes, config, 1, feature_names)
    P = ltn.Predicate(lstm).to(device)

    SatAgg = ltn.fuzzy_ops.SatAgg()

    params = list(P.parameters())

    optimizer = torch.optim.Adam(params, lr=config.learning_rate)

    def compute_satisfaction(loader):
        mean_sat = 0
        with torch.no_grad():
            for enum, (x, y, c_id) in enumerate(loader):
                x_D = ltn.Variable("x_D", x[y==1])
                x_not_D = ltn.Variable("x_not_D", x[y==0])
                formulas = []
                if x_D.value.numel()>0:
                    formulas.extend([
                        Forall(x_D, P(x_D)).value,
                    ])
                if x_not_D.value.numel()>0:
                    formulas.extend([
                        Forall(x_not_D, Not(P(x_not_D))).value,
                    ])
                mean_sat += SatAgg(*formulas).detach().cpu()
                del x_D, x_not_D
        return mean_sat / len(loader)

    def compute_accuracy(loader):
        y_pred = []
        y_true = []
        with torch.no_grad():
            for enum, (x, y, c_id) in enumerate(loader):
                x = x.to(device)
                output = lstm(x).detach().cpu().numpy()
                predictions = np.where(output > 0.5, 1., 0.).flatten()
                for i in range(len(y)):
                    y_pred.append(predictions[i])
                    y_true.append(y[i].cpu())
        # accuracy = accuracy_score(y_true, y_pred)
        f1score = f1_score(y_true, y_pred, average='macro')
        return f1score

    best_f1_val = 0.0
    count_early_stop = 0
    for epoch in range(args.num_epochs_nesy):
        train_loss = 0.0
        for enum, (x, y, c_id) in enumerate(train_loader):
            optimizer.zero_grad()
            x_D = ltn.Variable("x_D", x[y==1])
            x_not_D = ltn.Variable("x_not_D", x[y==0])
            x_All = ltn.Variable("x_All", x)
            formulas = []
            if x_D.value.numel()>0:
                formulas.extend([
                    Forall(x_D, P(x_D)).value,
                ])
            if x_not_D.value.numel()>0:
                formulas.extend([
                    Forall(x_not_D, Not(P(x_not_D)), p=6).value,
                ])
            sat_agg = SatAgg(*formulas)
            loss = 1 - sat_agg
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            del x_D, x_not_D, sat_agg
        train_loss = train_loss / len(train_loader)
        f1_val = compute_accuracy(val_loader)
        if f1_val > best_f1_val:
            best_f1_val = f1_val
            torch.save(lstm.state_dict(), "ltn_w_o_k.pth")
            count_early_stop = 0
        else:
            count_early_stop += 1
            if epoch >=10 and count_early_stop > 15:
                print("Early stopping at epoch", epoch+1)
                break

        print(" epoch %d | loss %.4f | val f1 %.4f | test f1 %.4f" %(epoch+1, train_loss, f1_val, compute_accuracy(test_loader)))
        

    lstm.load_state_dict(torch.load("ltn_w_o_k.pth"))
    lstm.eval()

    print("Metrics LTN w/o knowledge")
    accuracy_ltn, f1score_ltn, precision_ltn, recall_ltn, roc_auc_ltn = compute_metrics(test_loader, lstm, device, "ltn", scalers, features_dict, sequence_length)
    print("Accuracy:", accuracy_ltn)
    ltn_accuracies.append(accuracy_ltn)
    print("F1 Score:", f1score_ltn)
    ltn_f1_scores.append(f1score_ltn)
    print("Precision:", precision_ltn)
    ltn_precisions.append(precision_ltn)
    print("Recall:", recall_ltn)
    ltn_recalls.append(recall_ltn)
    print("ROC AUC:", roc_auc_ltn)
    ltn_roc_aucs.append(roc_auc_ltn)

    lstm = LSTMModel(vocab_sizes, config, 1, feature_names)
    lstm.train()
    P = ltn.Predicate(lstm).to(device)

    SatAgg = ltn.fuzzy_ops.SatAgg()

    lactate = ltn.Function(func = lambda x: x[:,(features_dict["Lactate"]*sequence_length-sequence_length):(features_dict["Lactate"]*sequence_length)])
    albumin = ltn.Function(func = lambda x: x[:,(features_dict["Albumin"]*sequence_length-sequence_length):(features_dict["Albumin"]*sequence_length)])
    creatinine = ltn.Function(func = lambda x: x[:,(features_dict["Creatinine (serum)"]*sequence_length-sequence_length):(features_dict["Creatinine (serum)"]*sequence_length)])
    bilirubin = ltn.Function(func = lambda x: x[:,(features_dict["Total Bilirubin"]*sequence_length-sequence_length):(features_dict["Total Bilirubin"]*sequence_length)])
    glucose = ltn.Function(func = lambda x: x[:,(features_dict["Glucose"]*sequence_length-sequence_length):(features_dict["Glucose"]*sequence_length)])
    age = ltn.Function(func = lambda x: x[:,(features_dict["anchor_age"]*sequence_length-sequence_length):(features_dict["anchor_age"]*sequence_length)][:, 0])
    wbc = ltn.Function(func = lambda x: x[:,(features_dict["White Blood Cells"]*sequence_length-sequence_length):(features_dict["White Blood Cells"]*sequence_length)])
    crp = ltn.Function(func = lambda x: x[:,(features_dict["C-Reactive Protein"]*sequence_length-sequence_length):(features_dict["C-Reactive Protein"]*sequence_length)])
    comorbidities = ltn.Function(func = lambda x: x[:, -23:])
    gcs = ltn.Function(func = lambda x: x[:,(features_dict["gcs"]*sequence_length-sequence_length):(features_dict["gcs"]*sequence_length)])
    platelet = ltn.Function(func = lambda x: x[:,(features_dict["Platelet Count"]*sequence_length-sequence_length):(features_dict["Platelet Count"]*sequence_length)])
    # mews score risk
    abps = ltn.Function(func = lambda x: x[:,(features_dict["Arterial Blood Pressure systolic"]*sequence_length-sequence_length):(features_dict["Arterial Blood Pressure systolic"]*sequence_length)])
    mabp = ltn.Function(func = lambda x: x[:,(features_dict["Arterial Blood Pressure mean"]*sequence_length-sequence_length):(features_dict["Arterial Blood Pressure mean"]*sequence_length)])
    respiratory_rate = ltn.Function(func = lambda x: x[:,(features_dict["Respiratory Rate"]*sequence_length-sequence_length):(features_dict["Respiratory Rate"]*sequence_length)])
    hear_rate = ltn.Function(func = lambda x: x[:,(features_dict["Heart Rate"]*sequence_length-sequence_length):(features_dict["Heart Rate"]*sequence_length)])

    # Functions for checking comorbidities
    HasCancer = ltn.Function(func = lambda x: x[:, -23:][:, 4])
    IsImmunoCompromised = ltn.Function(func = lambda x: x[:, -23:][:, 1])
    HasAKI = ltn.Function(func = lambda x: x[:, -23:][:, 0])
    HasCAD = ltn.Function(func = lambda x: x[:, -23:][:, 3])
    HasCOPD = ltn.Function(func = lambda x: x[:, -23:][:, 7])
    HasDementia = ltn.Function(func = lambda x: x[:, -23:][:, 8])
    HasDiabetesMellitus = ltn.Function(func = lambda x: x[:, -23:][:, 10])
    HasHeartFailure = ltn.Function(func = lambda x: x[:, -23:][:, 11])
    HasLeukemia = ltn.Function(func = lambda x: x[:, -23:][:, 16])
    HasLymphoma = ltn.Function(func = lambda x: x[:, -23:][:, 17])
    HasMetastaticCancer = ltn.Function(func = lambda x: x[:, -23:][:, 18])
    # Functions for checking lab values

    NoComorbidities = ltn.Function(func = lambda x: (x[:, -23:] == 0).all(axis=1))
    #Predicates modeling risk factors
    model_lactate = MLP(sequence_length, 64)
    LactateRisk = ltn.Predicate(model_lactate).to(device)
    model_albumin = MLP(sequence_length, 64)
    AlbuminRisk = ltn.Predicate(model_albumin).to(device)
    model_creatinine = MLP(sequence_length, 64)
    CreatinineRisk = ltn.Predicate(model_creatinine).to(device)
    model_birulin = MLP(sequence_length, 64)
    HighBilirubin = ltn.Predicate(model_birulin).to(device)
    model_crp = MLP(sequence_length, 64)
    CRPRisk = ltn.Predicate(model_crp).to(device)
    model_platelet = MLP(sequence_length, 64)
    PlateletLow = ltn.Predicate(model_platelet).to(device)
    model_glucose = MLP(sequence_length, 64)
    GlucoseRisk = ltn.Predicate(model_glucose).to(device)
    model_respiratory_rate = MLP(sequence_length, 64)
    RespiratoryRateRisk = ltn.Predicate(model_respiratory_rate).to(device)
    model_abps = MLP(sequence_length, 64)
    ArterialBloodPressureSystolicRisk = ltn.Predicate(model_abps).to(device)
    model_gcs = MLP(sequence_length, 64)
    GCSRisk = ltn.Predicate(model_gcs).to(device)
    model_lactate_not_clearing = SimpleMLP(sequence_length, 64)
    LactateNotClearing = ltn.Predicate(model_lactate_not_clearing).to(device)
    model_map = MLP(sequence_length, 64)
    MeanArterialPressureRisk = ltn.Predicate(model_map).to(device)
    model_cronic_conditions = SimpleMLP(23, 64)
    CronicConditionsRisk = ltn.Predicate(model_cronic_conditions).to(device)
    model_wbc = MLP(sequence_length, 64)
    WBCRisk = ltn.Predicate(model_wbc).to(device)
    model_age = SimpleMLPAge(24, 64)
    AgeRisk = ltn.Predicate(model_age).to(device)

    params = list(P.parameters()) + list(LactateRisk.parameters()) + list(HighBilirubin.parameters()) + list(RespiratoryRateRisk.parameters()) + list(ArterialBloodPressureSystolicRisk.parameters()) + list(GCSRisk.parameters()) + list(PlateletLow.parameters()) + list(LactateNotClearing.parameters()) + list(CreatinineRisk.parameters()) + list(CRPRisk.parameters()) + list(CronicConditionsRisk.parameters()) + list(GlucoseRisk.parameters()) + list(WBCRisk.parameters()) + list(AgeRisk.parameters())

    optimizer = torch.optim.Adam(params, lr=config.learning_rate)

    model_lactate.train()
    model_abps.train()
    model_albumin.train()
    model_creatinine.train()
    model_birulin.train()
    model_glucose.train()
    model_platelet.train()
    model_respiratory_rate.train()
    model_lactate_not_clearing.train()
    model_cronic_conditions.train()
    model_crp.train()
    model_wbc.train()

    # Functions modeling risk factors
    # Functions modeling risk factors
    lactate_above_threshold = lambda x: (x[:,(features_dict["Lactate"]*sequence_length-sequence_length):(features_dict["Lactate"]*sequence_length)] > scalers["Lactate"].transform(np.array([[4.0]]))[0][0]).any(axis=1)
    age_above_threshold = lambda x: (x[:,(features_dict["anchor_age"]*sequence_length-sequence_length):(features_dict["anchor_age"]*sequence_length)][:, 0] > scalers["anchor_age"].transform(np.array([[65.0]]))[0][0])
    glucose_above_threshold = lambda x: (x[:,(features_dict["Glucose"]*sequence_length-sequence_length):(features_dict["Glucose"]*sequence_length)] > scalers["Glucose"].transform(np.array([[100.0]]))[0][0]).any(axis=1)
    bloodpressuresystolic_below_threshold = lambda x: (x[:,(features_dict["Arterial Blood Pressure systolic"]*sequence_length-sequence_length):(features_dict["Arterial Blood Pressure systolic"]*sequence_length)] <= scalers["Arterial Blood Pressure systolic"].transform(np.array([[100.0]]))[0][0]).any(axis=1)
    creatinine_above_threshold = lambda x: (x[:,(features_dict["Creatinine (serum)"]*sequence_length-sequence_length):(features_dict["Creatinine (serum)"]*sequence_length)] >= scalers["Creatinine (serum)"].transform(np.array([[1.5]]))[0][0]).all(axis=1)
    respiratory_rate_risk = lambda x: (x[:,(features_dict["Respiratory Rate"]*sequence_length-sequence_length):(features_dict["Respiratory Rate"]*sequence_length)] >= scalers["Respiratory Rate"].transform(np.array([[29.0]]))[0][0]).all(axis=1) | (x[:,(features_dict["Respiratory Rate"]*sequence_length-sequence_length):(features_dict["Respiratory Rate"]*sequence_length)] < scalers["Respiratory Rate"].transform(np.array([[9.0]]))[0][0]).all(axis=1)
    blood_pressure_systolic_risk = lambda x: (x[:,(features_dict["Arterial Blood Pressure systolic"]*sequence_length-sequence_length):(features_dict["Arterial Blood Pressure systolic"]*sequence_length)] <= scalers["Arterial Blood Pressure systolic"].transform(np.array([[100.0]]))[0][0]).any(axis=1) & (x[:,(features_dict["Arterial Blood Pressure systolic"]*sequence_length-sequence_length):(features_dict["Arterial Blood Pressure systolic"]*sequence_length)] > scalers["Arterial Blood Pressure systolic"].transform(np.array([[0.0]]))[0][0]).all(axis=1)
    blood_pressure_systolic_risk = lambda x: (x[:,(features_dict["Arterial Blood Pressure systolic"]*sequence_length-sequence_length):(features_dict["Arterial Blood Pressure systolic"]*sequence_length)] <= scalers["Arterial Blood Pressure systolic"].transform(np.array([[100.0]]))[0][0]).any(axis=1)
    bilirubin_above_threshold = lambda x: (x[:,(features_dict["Total Bilirubin"]*sequence_length-sequence_length):(features_dict["Total Bilirubin"]*sequence_length)] >= scalers["Total Bilirubin"].transform(np.array([[2.0]]))[0][0]).any(axis=1)
    low_gcs = lambda x: (x[:,(features_dict["gcs"]*sequence_length-sequence_length):(features_dict["gcs"]*sequence_length)] < scalers["gcs"].transform(np.array([[8.0]]))[0][0]).any(axis=1)
    platelet_below_threshold = lambda x: (x[:,(features_dict["Platelet Count"]*sequence_length-sequence_length):(features_dict["Platelet Count"]*sequence_length)] < scalers["Platelet Count"].transform(np.array([[50.0]]))[0][0]).all(axis=1)
    crp_above_threshold = lambda x: (x[:,(features_dict["C-Reactive Protein"]*sequence_length-sequence_length):(features_dict["C-Reactive Protein"]*sequence_length)] >= scalers["C-Reactive Protein"].transform(np.array([[100.0]]))[0][0]).any(axis=1)
    map_below_threshold = lambda x: (x[:,(features_dict["Arterial Blood Pressure mean"]*sequence_length-sequence_length):(features_dict["Arterial Blood Pressure mean"]*sequence_length)] < scalers["Arterial Blood Pressure mean"].transform(np.array([[65.0]]))[0][0]).any(axis=1)
    lactate_not_clearining = lambda x: np.array([
        (np.all(np.diff(seq[seq != 0]) >= 0) and np.any(seq > scalers["Lactate"].transform(np.array([[2.0]]))[0][0])) if np.sum(seq != 0) > 1 else False
        for seq in x[:, (features_dict["Lactate"]*sequence_length - sequence_length):(features_dict["Lactate"]*sequence_length)].numpy()
    ])
    wbc_above_threshold = lambda x: (x[:,(features_dict["White Blood Cells"]*sequence_length-sequence_length):(features_dict["White Blood Cells"]*sequence_length)] > scalers["White Blood Cells"].transform(np.array([[30.0]]))[0][0]).any(axis=1)
    has_cronic_condition = lambda x : (x[:, -23:][:, 0] == 1) | (x[:, -23:][:, 1] == 1) | (x[:, -23:][:, 3] == 1) | (x[:, -23:][:, 4] == 1) | (x[:, -23:][:, 6] == 1) | (x[:, -23:][:, 7] == 1)  | (x[:, -23:][:, 6] == 8)  | (x[:, -23:][:, 12] == 1)  | (x[:, -23:][:, 14] == 1) | (x[:, -23:][:, 18] == 1)

    w_data = 0.8
    w_knowledge = 0.2
    best_f1_val = 0.0
    count_early_stop = 0
    for epoch in range(args.num_epochs_nesy):
        sa_knowledge = 0
        sa_data = 0
        count = 0
        train_loss = 0.0
        for enum, (x, y, c_id) in enumerate(train_loader):
            optimizer.zero_grad()
            x_D = ltn.Variable("x_D", x[y==1])
            x_not_D = ltn.Variable("x_not_D", x[y==0])
            x_All = ltn.Variable("x_All", x)
            lactate_above_th = lactate_above_threshold(x)
            x_above_lactate = ltn.Variable("x_above_lactate", x[lactate_above_th==1])
            x_above_glucose = ltn.Variable("x_above_glucose", x[glucose_above_threshold==1])
            x_above_bilirubin = ltn.Variable("x_above_bilirubin", x[bilirubin_above_threshold(x)==1])
            x_below_bilirubin = ltn.Variable("x_below_bilirubin", x[bilirubin_above_threshold(x)==0])
            x_above_creatinine = ltn.Variable("x_above_creatinine", x[creatinine_above_threshold(x)==1])
            x_not_above_creatinine = ltn.Variable("x_not_above_creatinine", x[creatinine_above_threshold(x)==0])
            x_risk_respiratory_rate = ltn.Variable("x_risk_respiratory_rate", x[respiratory_rate_risk(x)==1])
            x_risk_pressure_systolic = ltn.Variable("x_risk_pressure_systolic", x[blood_pressure_systolic_risk(x)==1])
            x_above_age = ltn.Variable("x_above_age", x[age_above_threshold(x)==1])
            x_below_platelet = ltn.Variable("x_below_platelet", x[platelet_below_threshold(x)==1])
            x_below_gcs = ltn.Variable("x_risk_gcs_low", x[low_gcs(x)==1])
            x_lactate_not_clear = ltn.Variable("x_lactate_not_clear", x[lactate_not_clearining(x)==1])
            x_lactate_clearing = ltn.Variable("x_lactate_clearing", x[lactate_not_clearining(x)==0])
            x_below_map = ltn.Variable("x_below_map", x[map_below_threshold(x)==1])
            x_above_crp = ltn.Variable("x_above_crp", x[crp_above_threshold(x)==1])
            x_cronic_condition = ltn.Variable("x_chronic_condition", x[has_cronic_condition(x)==1])
            x_above_wbc = ltn.Variable("x_above_wbc", x[wbc_above_threshold(x)==1])
            formulas = []
            formulas_knowledge = []
            if x_D.value.numel()>0:
                formulas.extend([
                    Forall(x_D, P(x_D)).value,
                ])
            if x_not_D.value.numel()>0:
                formulas.extend([
                    Forall(x_not_D, Not(P(x_not_D)), p=6).value,
                ])
            if x_below_gcs.value.numel()>0:
                formulas_knowledge.extend([
                    Forall(x_below_gcs, GCSRisk(gcs(x_below_gcs), comorbidities(x_below_gcs), age(x_below_gcs))).value,
                ])
            if x_above_lactate.value.numel()>0:
                formulas_knowledge.extend([
                    Forall(x_above_lactate, LactateRisk(lactate(x_above_lactate), comorbidities(x_above_lactate), age(x_above_lactate))).value,
                ])
            if x_below_platelet.value.numel()>0:
                formulas_knowledge.extend([
                    Forall(x_below_platelet, PlateletLow(platelet(x_below_platelet), comorbidities(x_below_platelet), age(x_below_platelet))).value,
                ])
            if x_lactate_not_clear.value.numel()>0:
                formulas_knowledge.extend([
                    Forall(x_lactate_not_clear, LactateNotClearing(lactate(x_lactate_not_clear))).value,
                ])
            if x_lactate_clearing.value.numel()>0:
                formulas_knowledge.extend([
                    Forall(x_lactate_clearing, Not(LactateNotClearing(lactate(x_lactate_clearing)))).value,
                ])
            if x_above_bilirubin.value.numel()>0:
                formulas_knowledge.extend([
                    Forall(x_above_bilirubin, HighBilirubin(bilirubin(x_above_bilirubin), comorbidities(x_above_bilirubin), age(x_above_bilirubin))).value,
                ])
            if x_risk_respiratory_rate.value.numel()>0 and x_risk_pressure_systolic.value.numel()>0:
                formulas_knowledge.extend([
                    Forall(x_risk_respiratory_rate, RespiratoryRateRisk(respiratory_rate(x_risk_respiratory_rate), comorbidities(x_risk_respiratory_rate), age(x_risk_respiratory_rate))).value,
                    Forall(x_risk_pressure_systolic, ArterialBloodPressureSystolicRisk(abps(x_risk_pressure_systolic), comorbidities(x_risk_pressure_systolic), age(x_risk_pressure_systolic))).value,
                ])
            if x_above_creatinine.value.numel()>0:
                formulas_knowledge.extend([
                    Forall(x_above_creatinine, CreatinineRisk(creatinine(x_above_creatinine), comorbidities(x_above_creatinine), age(x_above_creatinine))).value,
                ])
            if x_above_crp.value.numel()>0:
                formulas_knowledge.extend([
                    Forall(x_above_crp, CRPRisk(crp(x_above_crp), comorbidities(x_above_crp), age(x_above_crp))).value,
                ])
            if x_cronic_condition.value.numel()>0:
                formulas_knowledge.extend([
                    Forall(x_cronic_condition, CronicConditionsRisk(comorbidities(x_cronic_condition))).value,
                ])
            if x_above_glucose.value.numel()>0:
                formulas_knowledge.extend([
                    Forall(x_above_glucose, GlucoseRisk(glucose(x_above_glucose), comorbidities(x_above_glucose), age(x_above_glucose))).value,
                ])
            if x_above_wbc.value.numel()>0:
                formulas_knowledge.extend([
                    Forall(x_above_wbc, WBCRisk(wbc(x_above_wbc), comorbidities(x_above_wbc), age(x_above_wbc))).value,
                ])
            if x_above_age.value.numel()>0:
                formulas_knowledge.extend([
                    Forall(x_above_age, AgeRisk(age(x_above_age), comorbidities(x_above_age))).value,
                ])
            formulas_knowledge.extend([
                Forall(x_All, Implies(LactateRisk(lactate(x_All), comorbidities(x_All), age(x_All)), P(x_All))).value,
                Forall(x_All, Implies(HighBilirubin(bilirubin(x_All), comorbidities(x_All), age(x_All)), P(x_All))).value,
                Forall(x_All, Implies(PlateletLow(platelet(x_All), comorbidities(x_All), age(x_All)), P(x_All))).value,
                Forall(x_All, Implies(LactateNotClearing(lactate(x_All)), P(x_All))).value,
                Forall(x_All, Implies(CRPRisk(crp(x_All), comorbidities(x_All), age(x_All)), P(x_All))).value,
                Forall(x_All, Implies(CronicConditionsRisk(comorbidities(x_All)), P(x_All))).value,
                Forall(x_All, Implies(WBCRisk(wbc(x_All), comorbidities(x_All), age(x_All)), P(x_All))).value,
                Forall(x_All, Implies(AgeRisk(age(x_All), comorbidities(x_All)), P(x_All))).value,
            ])
            sat_agg = SatAgg(*formulas)
            if len(formulas_knowledge) > 0:
                sat_agg_knowledge = SatAgg(*formulas_knowledge)
                count += 1
                loss = 1 - (w_data * sat_agg + w_knowledge * sat_agg_knowledge)
            else:
                loss = 1 - sat_agg
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            del x_D, x_not_D, sat_agg
        train_loss = train_loss / len(train_loader)
        f1_val = compute_accuracy(val_loader)
        if f1_val > best_f1_val:
            best_f1_val = f1_val
            torch.save(lstm.state_dict(), "ltn_w_k.pth")
            count_early_stop = 0
        else:
            count_early_stop += 1
            if epoch >=10 and count_early_stop > 20:
                print("Early stopping at epoch", epoch+1)
                break

        print(" epoch %d | loss %.4f | val f1 %.4f | test f1 %.4f" %(epoch+1, train_loss, f1_val, compute_accuracy(test_loader)))
        # compute_satisfaction_level(train_loader)

    lstm.load_state_dict(torch.load("ltn_w_k.pth"))
    lstm.eval()

    print("Metrics LTN w/ knowledge")
    accuracy_ltn_w, f1score_ltn_w, precision_ltn_w, recall_ltn_w, roc_auc_ltn_w = compute_metrics(test_loader, lstm, device, "ltn_w_k", scalers, features_dict, sequence_length)
    print("Accuracy:", accuracy_ltn_w)
    ltn_log_accuracies.append(accuracy_ltn_w)
    print("F1 Score:", f1score_ltn_w)
    ltn_log_f1_scores.append(f1score_ltn_w)
    print("Precision:", precision_ltn_w)
    ltn_log_precisions.append(precision_ltn_w)
    print("Recall:", recall_ltn_w)
    ltn_log_recalls.append(recall_ltn_w)
    print("ROC AUC:", roc_auc_ltn_w)
    ltn_log_roc_aucs.append(roc_auc_ltn_w)

    #####################################

# write mean and std of the metrics to a file
with open("stratified_results.txt", "w") as f:
    f.write("LSTM Results:\n")
    f.write(f"Accuracy: {np.mean(lstm_accuracies):.4f} ± {np.std(lstm_accuracies):.4f}\n")
    f.write(f"F1 Score: {np.mean(lstm_f1_scores):.4f} ± {np.std(lstm_f1_scores):.4f}\n")
    f.write(f"Precision: {np.mean(lstm_precisions):.4f} ± {np.std(lstm_precisions):.4f}\n")
    f.write(f"Recall: {np.mean(lstm_recalls):.4f} ± {np.std(lstm_recalls):.4f}\n")
    f.write(f"ROC AUC: {np.mean(lstm_roc_aucs):.4f} ± {np.std(lstm_roc_aucs):.4f}\n\n")

    f.write("LTN Results without Knowledge:\n")
    f.write(f"Accuracy: {np.mean(ltn_accuracies):.4f} ± {np.std(ltn_accuracies):.4f}\n")
    f.write(f"F1 Score: {np.mean(ltn_f1_scores):.4f} ± {np.std(ltn_f1_scores):.4f}\n")
    f.write(f"Precision: {np.mean(ltn_precisions):.4f} ± {np.std(ltn_precisions):.4f}\n")
    f.write(f"Recall: {np.mean(ltn_recalls):.4f} ± {np.std(ltn_recalls):.4f}\n")
    f.write(f"ROC AUC: {np.mean(ltn_roc_aucs):.4f} ± {np.std(ltn_roc_aucs):.4f}\n\n")

    f.write("LTN Results with Weighted Knowledge:\n")
    f.write(f"Accuracy: {np.mean(ltn_log_accuracies):.4f} ± {np.std(ltn_log_accuracies):.4f}\n")
    f.write(f"F1 Score: {np.mean(ltn_log_f1_scores):.4f} ± {np.std(ltn_log_f1_scores):.4f}\n")
    f.write(f"Precision: {np.mean(ltn_log_precisions):.4f} ± {np.std(ltn_log_precisions):.4f}\n")
    f.write(f"Recall: {np.mean(ltn_log_recalls):.4f} ± {np.std(ltn_log_recalls):.4f}\n")
    f.write(f"ROC AUC: {np.mean(ltn_log_roc_aucs):.4f} ± {np.std(ltn_log_roc_aucs):.4f}\n")

    f.write("XGBoost Results:\n")
    f.write(f"Accuracy: {np.mean(xgb_accuracies):.4f} ± {np.std(xgb_accuracies):.4f}\n")
    f.write(f"F1 Score: {np.mean(xgb_f1_scores):.4f} ± {np.std(xgb_f1_scores):.4f}\n")
    f.write(f"Precision: {np.mean(xgb_precisions):.4f} ± {np.std(xgb_precisions):.4f}\n")
    f.write(f"Recall: {np.mean(xgb_recalls):.4f} ± {np.std(xgb_recalls):.4f}\n")
    f.write(f"ROC AUC: {np.mean(xgb_roc_aucs):.4f} ± {np.std(xgb_roc_aucs):.4f}\n")

    f.write("Random Forest Results:\n")
    f.write(f"Accuracy: {np.mean(rf_accuracies):.4f} ± {np.std(rf_accuracies):.4f}\n")
    f.write(f"F1 Score: {np.mean(rf_f1_scores):.4f} ± {np.std(rf_f1_scores):.4f}\n")
    f.write(f"Precision: {np.mean(rf_precisions):.4f} ± {np.std(rf_precisions):.4f}\n")
    f.write(f"Recall: {np.mean(rf_recalls):.4f} ± {np.std(rf_recalls):.4f}\n")
    f.write(f"ROC AUC: {np.mean(rf_roc_aucs):.4f} ± {np.std(rf_roc_aucs):.4f}\n")
