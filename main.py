import pandas as pd
import ltn
import torch
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
from model.models import LSTMModel, MLP, SimpleMLP, SimpleMLPAge
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import statistics
from collections import defaultdict, Counter
from data.preprocessing import preprocess_eventlog
from data.dataset import SepsisDataset, ModelConfig
from metrics import compute_metrics, compute_highlactate
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
from xgboost import XGBClassifier

from plot_facts import plot_concept_level_verification, plot_knowledge_axioms, plot_concept_level_verification_scatter

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
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs for training")
    parser.add_argument("--num_epochs_nesy", type=int, default=20, help="Number of epochs for training LTN model")
    # training configuration
    parser.add_argument("--train_vanilla", type=bool, default=True, help="Train vanilla LSTM model")
    parser.add_argument("--train_nesy", type=bool, default=True, help="Train LTN model")
    parser.add_argument("--dataset_size", type=float, default=30, help="Size of the dataset (10%, 20%, 50%, 70%, 90%, 100%)")
    parser.add_argument("--sampling", type=bool, default=False, help="Train LTN model")

    return parser.parse_args()

args = get_args()

seed = 32
random.seed(32)

classes = ["No Death", "Death"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("-- INFO: reading dataset")
data = pd.read_csv("data/subset/events_6h_before_death_gcs.csv", dtype={"hadm_id": str, "subject_id": str})
print(data.columns)
columns_to_drop = ["SOFA Score", "APACHEII-Renal failure", "Urine output_ApacheIV", "APACHE II"]
data = data[~data["concept:name"].isin(columns_to_drop)]

(X, y, feature_names), vocab_sizes, scalers, sequence_length = preprocess_eventlog(data, seed, args.sampling)

config = ModelConfig(
    hidden_size=args.hidden_size,
    num_layers=args.num_layers,
    dropout_rate=args.dropout_rate,
    sequence_length=sequence_length,
    num_epochs = args.num_epochs
)

features_dict = {s: i for i, s in enumerate(feature_names, start=1)}
print(feature_names)

# Extract features for XGBoost and Random Forest models

X_train_xgb = []
y_train_xgb = []

X_val_xgb = []
y_val_xgb = []

X_test_xgb = []
y_test_xgb = []

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

# print("Dataset size:", len(data))
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=seed)

def mean_excluding_zeros(values):
    non_zero_values = [v for v in values if v != 0.0]
    if not non_zero_values:
        return 0.0  # or `None` depending on your needs
    return sum(non_zero_values) / len(non_zero_values)

for x, y in zip(X_train, y_train):
    features = [lst for i, lst in enumerate(x[0]) if i not in (0, 1, 2, 3)]
    features = [mean_excluding_zeros(lst) for lst in features]
    X_train_xgb.append(np.array(features))
    y_train_xgb.append(y)

for x, y in zip(X_val, y_val):
    features = [lst for i, lst in enumerate(x[0]) if i not in (0, 1, 2, 3)]
    features = [mean_excluding_zeros(lst) for lst in features]
    X_val_xgb.append(np.array(features))
    y_val_xgb.append(y)

for x, y in zip(X_test, y_test):
    features = [lst for i, lst in enumerate(x[0]) if i not in (0, 1, 2, 3)]
    features = [mean_excluding_zeros(lst) for lst in features]
    X_test_xgb.append(np.array(features))
    y_test_xgb.append(y)
    
# train xgboost model
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
print("F1 Score:", f1_xgb)
print("Precision:", precision_xgb)
print("Recall:", recall_xgb)
print("ROC AUC:", roc_auc_xgb)

print("--- Training Random Forest model")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
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
print("F1 Score:", f1_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("ROC AUC:", roc_auc_rf)

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
print("Accuracy:", accuracy)
f1 = f1_score(y_true, y_pred, average='macro')
print("F1 Score:", f1)
precision = precision_score(y_true, y_pred, average='macro')
print("Precision:", precision)
recall = recall_score(y_true, y_pred, average='macro')
print("Recall:", recall)
roc_auc = roc_auc_score(y_true, y_scores)
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
        for _, (x, y, _) in enumerate(loader):
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
        for _, (x, y, _) in enumerate(loader):
            x = x.to(device)
            output = lstm(x).detach().cpu().numpy()
            predictions = np.where(output > 0.5, 1., 0.).flatten()
            for i in range(len(y)):
                y_pred.append(predictions[i])
                y_true.append(y[i].cpu())
    accuracy = accuracy_score(y_true, y_pred)
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
                Forall(x_not_D, Not(P(x_not_D))).value,
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
    # print(" epoch %d | loss %.4f"
    #             %(epoch, train_loss))
    print(" epoch %d | loss %.4f | val f1 %.4f | test f1 %.4f" %(epoch+1, train_loss, f1_val, compute_accuracy(test_loader)))
    # compute_satisfaction_level(train_loader)

lstm.load_state_dict(torch.load("ltn_w_o_k.pth"))
lstm.eval()

print("Metrics LTN w/o knowledge")
accuracy_ltn, f1score_ltn, precision_ltn, recall_ltn, roc_auc_ltn = compute_metrics(test_loader, lstm, device, "ltn", scalers, features_dict, sequence_length)
print("Accuracy:", accuracy_ltn)
print("F1 Score:", f1score_ltn)
print("Precision:", precision_ltn)
print("Recall:", recall_ltn)
print("ROC AUC:", roc_auc_ltn)

lstm = LSTMModel(vocab_sizes, config, 1, feature_names)
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
#mews score risk
abps = ltn.Function(func = lambda x: x[:,(features_dict["Arterial Blood Pressure systolic"]*sequence_length-sequence_length):(features_dict["Arterial Blood Pressure systolic"]*sequence_length)])
mabp = ltn.Function(func = lambda x: x[:,(features_dict["Arterial Blood Pressure mean"]*sequence_length-sequence_length):(features_dict["Arterial Blood Pressure mean"]*sequence_length)])
respiratory_rate = ltn.Function(func = lambda x: x[:,(features_dict["Respiratory Rate"]*sequence_length-sequence_length):(features_dict["Respiratory Rate"]*sequence_length)])
hear_rate = ltn.Function(func = lambda x: x[:,(features_dict["Heart Rate"]*sequence_length-sequence_length):(features_dict["Heart Rate"]*sequence_length)])
wbc = ltn.Function(func = lambda x: x[:,(features_dict["White Blood Cells"]*sequence_length-sequence_length):(features_dict["White Blood Cells"]*sequence_length)])

#Functions for checking comorbidities
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
#Functions for checking lab values

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

lactate_values_before_training = []
platelet_values_before_training = []
bilirubin_values_before_training = []
qSOFA_values_before_training = []
creatinine_values_before_training = []
crp_values_before_training = []
map_values_before_training = []
respiratory_rate_values_before_training = []
blood_pressure_systolic_risk_values_before_training = []
glucose_values_before_training = []
wbc_values_before_training = []

lactate_clinical_concept_values = []
platelet_clinical_concept_values = []
bilirubin_clinical_concept_values = []
qSOFA_clinical_concept_values = []

for x, y, c_id in test_loader:
    x_All = ltn.Variable("x_All", x)
    #lactate clinical concept values
    lactate_values_before_training.append(LactateRisk(lactate(x_All), comorbidities(x_All), age(x_All)).value.cpu().detach().numpy())
    lactate_clinical_concept_values.append(lactate_above_threshold(x).cpu().detach().numpy())
    #platelet clinical concept values
    platelet_values_before_training.append(PlateletLow(platelet(x_All), comorbidities(x_All), age(x_All)).value.cpu().detach().numpy())
    platelet_clinical_concept_values.append(platelet_below_threshold(x).cpu().detach().numpy())
    #bilirubin clinical concept values
    bilirubin_values_before_training.append(HighBilirubin(bilirubin(x_All), comorbidities(x_All), age(x_All)).value.cpu().detach().numpy())
    bilirubin_clinical_concept_values.append(bilirubin_above_threshold(x).cpu().detach().numpy())
    #qSOFA clinical concept values
    qSOFA_values_before_training.append(And(RespiratoryRateRisk(respiratory_rate(x_All), comorbidities(x_All), age(x_All)), ArterialBloodPressureSystolicRisk(abps(x_All), comorbidities(x_All), age(x_All))).value.cpu().detach().numpy())
    qSOFA_clinical_concept_values.append(np.logical_and(blood_pressure_systolic_risk(x).cpu().detach().numpy(), respiratory_rate_risk(x).cpu().detach().numpy()))
    #creatinine clinical concept values
    creatinine_values_before_training.append(CreatinineRisk(creatinine(x_All), comorbidities(x_All), age(x_All)).value.cpu().detach().numpy())
    crp_values_before_training.append(CRPRisk(crp(x_All), comorbidities(x_All), age(x_All)).value.cpu().detach().numpy())
    #map clinical concept values
    map_values_before_training.append(MeanArterialPressureRisk(mabp(x_All), comorbidities(x_All), age(x_All)).value.cpu().detach().numpy())
    respiratory_rate_values_before_training.append(RespiratoryRateRisk(respiratory_rate(x_All), comorbidities(x_All), age(x_All)).value.cpu().detach().numpy())
    blood_pressure_systolic_risk_values_before_training.append(ArterialBloodPressureSystolicRisk(abps(x_All), comorbidities(x_All), age(x_All)).value.cpu().detach().numpy())
    glucose_values_before_training.append(GlucoseRisk(glucose(x_All), comorbidities(x_All), age(x_All)).value.cpu().detach().numpy())
    wbc_values_before_training.append(WBCRisk(wbc(x_All), comorbidities(x_All), age(x_All)).value.cpu().detach().numpy())
#flatten the list of lists
lactate_values_before_training = np.concatenate(lactate_values_before_training, axis=0)
lactate_clinical_concept_values = np.concatenate(lactate_clinical_concept_values, axis=0)
lactate_clinical_concept_values = lactate_clinical_concept_values.astype(int)
platelet_values_before_training = np.concatenate(platelet_values_before_training, axis=0)
platelet_clinical_concept_values = np.concatenate(platelet_clinical_concept_values, axis=0)
platelet_clinical_concept_values = platelet_clinical_concept_values.astype(int)
bilirubin_values_before_training = np.concatenate(bilirubin_values_before_training, axis=0)
bilirubin_clinical_concept_values = np.concatenate(bilirubin_clinical_concept_values, axis=0)
bilirubin_clinical_concept_values = bilirubin_clinical_concept_values.astype(int)
qSOFA_values_before_training = np.concatenate(qSOFA_values_before_training, axis=0)
qSOFA_clinical_concept_values = np.concatenate(qSOFA_clinical_concept_values, axis=0)
qSOFA_clinical_concept_values = qSOFA_clinical_concept_values.astype(int)
creatinine_values_before_training = np.concatenate(creatinine_values_before_training, axis=0)
crp_values_before_training = np.concatenate(crp_values_before_training, axis=0)
map_values_before_training = np.concatenate(map_values_before_training, axis=0)
respiratory_rate_values_before_training = np.concatenate(respiratory_rate_values_before_training, axis=0)
blood_pressure_systolic_risk_values_before_training = np.concatenate(blood_pressure_systolic_risk_values_before_training, axis=0)
glucose_values_before_training = np.concatenate(glucose_values_before_training, axis=0)
wbc_values_before_training = np.concatenate(wbc_values_before_training, axis=0)

w_data = 0.8
w_knowledge = 0.2
best_f1_val = 0.0
count_early_stop = 0
for epoch in range(args.num_epochs_nesy):
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
        if epoch >=10 and count_early_stop > 15:
            print("Early stopping at epoch", epoch+1)
            break

    print(" epoch %d | loss %.4f | val f1 %.4f | test f1 %.4f" %(epoch+1, train_loss, f1_val, compute_accuracy(test_loader)))
    #compute_satisfaction_level(train_loader)

lstm.load_state_dict(torch.load("ltn_w_k.pth"))
lstm.eval()

print("Metrics LTN w/ knowledge")
accuracy_ltn_w, f1score_ltn_w, precision_ltn_w, recall_ltn_w, roc_auc_ltn_w = compute_metrics(test_loader, lstm, device, "ltn_w_k", scalers, features_dict, sequence_length)
print("Accuracy:", accuracy_ltn_w)
print("F1 Score:", f1score_ltn_w)
print("Precision:", precision_ltn_w)
print("Recall:", recall_ltn_w)
print("ROC AUC:", roc_auc_ltn_w)

from metrics import compute_predicate_values
# Compute predicate values for the test set
print("Computing predicate values for the test set")
compute_predicate_values(test_loader, model_lactate, device, features_dict, sequence_length)

#write all metrics to a file
with open("metrics.txt", "w") as f:
    f.write("Metrics LSTM\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"ROC AUC: {roc_auc}\n\n")

    f.write("Metrics LTN w/o knowledge\n")
    f.write(f"Accuracy: {accuracy_ltn}\n")
    f.write(f"F1 Score: {f1score_ltn}\n")
    f.write(f"Precision: {precision_ltn}\n")
    f.write(f"Recall: {recall_ltn}\n")
    f.write(f"ROC AUC: {roc_auc_ltn}\n\n")

    f.write("Metrics LTN w/ knowledge\n")
    f.write(f"Accuracy: {accuracy_ltn_w}\n")
    f.write(f"F1 Score: {f1score_ltn_w}\n")
    f.write(f"Precision: {precision_ltn_w}\n")
    f.write(f"Recall: {recall_ltn_w}\n")
    f.write(f"ROC AUC: {roc_auc_ltn_w}\n")


qSOFA_values_after_training = []
lactate_values_after_training = []
platelet_values_after_training = []
bilirubin_values_after_training = []
creatinine_values_after_training = []
crp_values_after_training = []
map_values_after_training = []
respiratory_rate_values_after_training = []
blood_pressure_systolic_risk_values_after_training = []
glucose_values_after_training = []
wbc_values_after_training = []
lactate_not_clearing = []
chronic_conditions = []
age_values_after_training = []

actual_lactate_values = []
actual_bilirubin_values = []
actual_platelet_values = []
actual_qSOFA_values = []
actual_creatinine_values = []
actual_crp_values = []
actual_map_values = []
actual_resp_rate_values = []
actual_blood_pressure_systolic_values = []
actual_glucose_values = []
actual_wbc_values = []
actual_age_values = []

labels = []
predictions = []
mortality_probabilities = []
predicted_labels = []
cases_ids = []

lactate_implies_death = []
bilirubin_implies_death = []
crp_implies_death = []
wbc_implies_death = []
platelet_implies_death = []
map_implies_death = []
age_implies_death = []
lactate_not_clearing_implies_death = []
cronic_conditions_implies_death = []

for x, y, c_id in test_loader:
    x_All = ltn.Variable("x_All", x)
    qSOFA_values_after_training.append(And(RespiratoryRateRisk(respiratory_rate(x_All), comorbidities(x_All), age(x_All)), ArterialBloodPressureSystolicRisk(abps(x_All), comorbidities(x_All), age(x_All))).value.cpu().detach().numpy())
    lactate_values_after_training.append(LactateRisk(lactate(x_All), comorbidities(x_All), age(x_All)).value.cpu().detach().numpy())
    actual_lactate_values.append(scalers["Lactate"].inverse_transform(lactate(x_All).value.cpu().detach().numpy()))
    platelet_values_after_training.append(PlateletLow(platelet(x_All), comorbidities(x_All), age(x_All)).value.cpu().detach().numpy())
    actual_platelet_values.append(scalers["Platelet Count"].inverse_transform(platelet(x_All).value.cpu().detach().numpy()))
    bilirubin_values_after_training.append(HighBilirubin(bilirubin(x_All), comorbidities(x_All), age(x_All)).value.cpu().detach().numpy())
    actual_bilirubin_values.append(scalers["Total Bilirubin"].inverse_transform(bilirubin(x_All).value.cpu().detach().numpy()))
    creatinine_values_after_training.append(CreatinineRisk(creatinine(x_All), comorbidities(x_All), age(x_All)).value.cpu().detach().numpy())
    actual_creatinine_values.append(scalers["Creatinine (serum)"].inverse_transform(creatinine(x_All).value.cpu().detach().numpy()))
    crp_values_after_training.append(CRPRisk(crp(x_All), comorbidities(x_All), age(x_All)).value.cpu().detach().numpy())
    actual_crp_values.append(scalers["C-Reactive Protein"].inverse_transform(crp(x_All).value.cpu().detach().numpy()))
    map_values_after_training.append(MeanArterialPressureRisk(mabp(x_All), comorbidities(x_All), age(x_All)).value.cpu().detach().numpy())
    actual_map_values.append(scalers["Arterial Blood Pressure mean"].inverse_transform(mabp(x_All).value.cpu().detach().numpy()))
    respiratory_rate_values_after_training.append(RespiratoryRateRisk(respiratory_rate(x_All), comorbidities(x_All), age(x_All)).value.cpu().detach().numpy())
    actual_resp_rate_values.append(scalers["Respiratory Rate"].inverse_transform(respiratory_rate(x_All).value.cpu().detach().numpy()))
    blood_pressure_systolic_risk_values_after_training.append(ArterialBloodPressureSystolicRisk(abps(x_All), comorbidities(x_All), age(x_All)).value.cpu().detach().numpy())
    actual_blood_pressure_systolic_values.append(scalers["Arterial Blood Pressure systolic"].inverse_transform(abps(x_All).value.cpu().detach().numpy()))
    glucose_values_after_training.append(GlucoseRisk(glucose(x_All), comorbidities(x_All), age(x_All)).value.cpu().detach().numpy())
    actual_glucose_values.append(scalers["Glucose"].inverse_transform(glucose(x_All).value.cpu().detach().numpy()))
    wbc_values_after_training.append(WBCRisk(wbc(x_All), comorbidities(x_All), age(x_All)).value.cpu().detach().numpy())
    actual_wbc_values.append(scalers["White Blood Cells"].inverse_transform(wbc(x_All).value.cpu().detach().numpy()))
    age_arr = age(x_All).value.cpu().detach().numpy()
    actual_age_values.append(scalers["anchor_age"].inverse_transform(age_arr.reshape(-1, 1)))
    age_values_after_training.append(AgeRisk(age(x_All), comorbidities(x_All)).value.cpu().detach().numpy())
    lactate_not_clearing.append(LactateNotClearing(lactate(x_All)).value.cpu().detach().numpy())
    chronic_conditions.append(CronicConditionsRisk(comorbidities(x_All)).value.cpu().detach().numpy())
    mortality_probabilities.append(P(x_All).value.cpu().detach().numpy())
    labels.append(y.cpu().detach().numpy())
    cases_ids.append(c_id)
    # implies values
    lactate_implies_death.append(Implies(LactateRisk(lactate(x_All), comorbidities(x_All), age(x_All)), P(x_All)).value.cpu().detach().numpy())
    bilirubin_implies_death.append(Implies(HighBilirubin(bilirubin(x_All), comorbidities(x_All), age(x_All)), P(x_All)).value.cpu().detach().numpy())
    crp_implies_death.append(Implies(CRPRisk(crp(x_All), comorbidities(x_All), age(x_All)), P(x_All)).value.cpu().detach().numpy())
    wbc_implies_death.append(Implies(WBCRisk(wbc(x_All), comorbidities(x_All), age(x_All)), P(x_All)).value.cpu().detach().numpy())
    platelet_implies_death.append(Implies(PlateletLow(platelet(x_All), comorbidities(x_All), age(x_All)), P(x_All)).value.cpu().detach().numpy())
    map_implies_death.append(Implies(MeanArterialPressureRisk(mabp(x_All), comorbidities(x_All), age(x_All)), P(x_All)).value.cpu().detach().numpy())
    age_implies_death.append(Implies(AgeRisk(age(x_All), comorbidities(x_All)), P(x_All)).value.cpu().detach().numpy())
    lactate_not_clearing_implies_death.append(Implies(LactateNotClearing(lactate(x_All)), P(x_All)).value.cpu().detach().numpy())
    cronic_conditions_implies_death.append(Implies(CronicConditionsRisk(comorbidities(x_All)), P(x_All)).value.cpu().detach().numpy())

qSOFA_values_after_training = np.concatenate(qSOFA_values_after_training, axis=0)
lactate_values_after_training = np.concatenate(lactate_values_after_training, axis=0)
platelet_values_after_training = np.concatenate(platelet_values_after_training, axis=0)
bilirubin_values_after_training = np.concatenate(bilirubin_values_after_training, axis=0)
creatinine_values_after_training = np.concatenate(creatinine_values_after_training, axis=0)
crp_values_after_training = np.concatenate(crp_values_after_training, axis=0)
map_values_after_training = np.concatenate(map_values_after_training, axis=0)
blood_pressure_systolic_risk_values_after_training = np.concatenate(blood_pressure_systolic_risk_values_after_training, axis=0)
respiratory_rate_values_after_training = np.concatenate(respiratory_rate_values_after_training, axis=0)
glucose_values_after_training = np.concatenate(glucose_values_after_training, axis=0)
wbc_values_after_training = np.concatenate(wbc_values_after_training, axis=0)
lactate_not_clearing = np.concatenate(lactate_not_clearing, axis=0)
age_values_after_training = np.concatenate(age_values_after_training, axis=0)
chronic_conditions = np.concatenate(chronic_conditions, axis=0)
actual_lactate_values = np.concatenate(actual_lactate_values, axis=0)
actual_lactate_values = [max(inner) for inner in actual_lactate_values]
actual_lactate_values = np.array(actual_lactate_values)
actual_bilirubin_values = np.concatenate(actual_bilirubin_values, axis=0)
actual_bilirubin_values = [max(inner) for inner in actual_bilirubin_values]
actual_bilirubin_values = np.array(actual_bilirubin_values)
actual_platelet_values = np.concatenate(actual_platelet_values, axis=0)
actual_platelet_values = [min([v for v in inner if v != 0]) if any(v != 0 for v in inner) else 0 for inner in actual_platelet_values]
actual_platelet_values = np.array(actual_platelet_values)
actual_creatinine_values = np.concatenate(actual_creatinine_values, axis=0)
actual_creatinine_values = [max(inner) for inner in actual_creatinine_values]
actual_creatinine_values = np.array(actual_creatinine_values)
actual_crp_values = np.concatenate(actual_crp_values, axis=0)
actual_crp_values = [max(inner) for inner in actual_crp_values]
actual_crp_values = np.array(actual_crp_values)
actual_resp_rate_values = np.concatenate(actual_resp_rate_values, axis=0)
actual_resp_rate_values = [min(inner) for inner in actual_resp_rate_values]
actual_resp_rate_values = np.array(actual_resp_rate_values)
actual_blood_pressure_systolic_values = np.concatenate(actual_blood_pressure_systolic_values, axis=0)
actual_blood_pressure_systolic_values = [min([v for v in inner if v != 0]) if any(v != 0 for v in inner) else 0 for inner in actual_blood_pressure_systolic_values]
actual_blood_pressure_systolic_values = np.array(actual_blood_pressure_systolic_values)
actual_glucose_values = np.concatenate(actual_glucose_values, axis=0)
actual_glucose_values = [max(inner) for inner in actual_glucose_values]
actual_glucose_values = np.array(actual_glucose_values)
actual_wbc_values = np.concatenate(actual_wbc_values, axis=0)
actual_wbc_values = [max(inner) for inner in actual_wbc_values]
actual_wbc_values = np.array(actual_wbc_values)
actual_age_values = np.concatenate(actual_age_values, axis=0)
actual_map_values = np.concatenate(actual_map_values, axis=0)
actual_map_values = [min([v for v in inner if v != 0]) if any(v != 0 for v in inner) else 0 for inner in actual_map_values]
actual_map_values = np.array(actual_map_values)
# implies values
lactate_implies_death = np.concatenate(lactate_implies_death, axis=0)
bilirubin_implies_death = np.concatenate(bilirubin_implies_death, axis=0)
crp_implies_death = np.concatenate(crp_implies_death, axis=0)
wbc_implies_death = np.concatenate(wbc_implies_death, axis=0)
platelet_implies_death = np.concatenate(platelet_implies_death, axis=0)
map_implies_death = np.concatenate(map_implies_death, axis=0)
age_implies_death = np.concatenate(age_implies_death, axis=0)
lactate_not_clearing_implies_death = np.concatenate(lactate_not_clearing_implies_death, axis=0)
cronic_conditions_implies_death = np.concatenate(cronic_conditions_implies_death, axis=0)

labels = np.concatenate(labels, axis=0)
cases_ids = np.concatenate(cases_ids, axis=0)
mortality_probabilities = np.concatenate(mortality_probabilities, axis=0)
predicted_labels = mortality_probabilities > 0.5
predicted_labels = predicted_labels.astype(int)

# take one true positive and one false positive
true_positive_indices = np.random.choice(np.where((labels == 1) & (predicted_labels == 1))[0], size=1, replace=False).tolist()
false_positive_indices = np.random.choice(np.where((labels == 0) & (predicted_labels == 1))[0], size=1, replace=False).tolist()

print(true_positive_indices)
print(false_positive_indices)

# take all predicates values, actual values, and predicted mortality probability of these indices
true_positive_values = {
    "lactate": actual_lactate_values[true_positive_indices][0],
    "bilirubin": actual_bilirubin_values[true_positive_indices][0],
    "platelet": actual_platelet_values[true_positive_indices][0],
    "crp": actual_crp_values[true_positive_indices][0],
    "blood_pressure_systolic": actual_blood_pressure_systolic_values[true_positive_indices][0],
    "glucose": actual_glucose_values[true_positive_indices][0],
    "wbc": actual_wbc_values[true_positive_indices][0],
    "map": actual_map_values[true_positive_indices][0],
    "HighLactate": lactate_values_after_training[true_positive_indices][0],
    "HighBilirubin": bilirubin_values_after_training[true_positive_indices][0],
    "LowPlatelets": platelet_values_after_training[true_positive_indices][0],
    "HighCRP": crp_values_after_training[true_positive_indices][0],
    "HighBloodPressureSystolic": blood_pressure_systolic_risk_values_after_training[true_positive_indices][0],
    "HighGlucose": glucose_values_after_training[true_positive_indices][0],
    "HighWBC": wbc_values_after_training[true_positive_indices][0],
    "LowMAP": map_values_after_training[true_positive_indices][0],
    "AgeRisk": age_values_after_training[true_positive_indices][0],
    "LactateNotClearing": lactate_values_after_training[true_positive_indices][0],
    "ChronicConditions": chronic_conditions[true_positive_indices][0],
    "LactateImpliesDeath": lactate_implies_death[true_positive_indices][0],
    "BilirubinImpliesDeath": bilirubin_implies_death[true_positive_indices][0],
    "CRPImpliesDeath": crp_implies_death[true_positive_indices][0],
    "WBCImpliesDeath": wbc_implies_death[true_positive_indices][0],
    "PlateletImpliesDeath": platelet_implies_death[true_positive_indices][0],
    "MAPImpliesDeath": map_implies_death[true_positive_indices][0],
    "AgeImpliesDeath": age_implies_death[true_positive_indices][0],
    "LactateNotClearingImpliesDeath": lactate_not_clearing_implies_death[true_positive_indices][0],
    "CronicConditionsImpliesDeath": cronic_conditions_implies_death[true_positive_indices][0],
    "actual": labels[true_positive_indices][0],
    "probability": mortality_probabilities[true_positive_indices][0],
    "case_id": cases_ids[true_positive_indices][0]
}

false_positive_values = {
    "lactate": actual_lactate_values[false_positive_indices][0],
    "bilirubin": actual_bilirubin_values[false_positive_indices][0],
    "platelet": actual_platelet_values[false_positive_indices][0],
    "crp": actual_crp_values[false_positive_indices][0],
    "blood_pressure_systolic": actual_blood_pressure_systolic_values[false_positive_indices][0],
    "glucose": actual_glucose_values[false_positive_indices][0],
    "wbc": actual_wbc_values[false_positive_indices][0],
    "map": actual_map_values[false_positive_indices][0],
    "HighLactate": lactate_values_after_training[false_positive_indices][0],
    "HighBilirubin": bilirubin_values_after_training[false_positive_indices][0],
    "LowPlatelets": platelet_values_after_training[false_positive_indices][0],
    "HighCRP": crp_values_after_training[false_positive_indices][0],
    "HighBloodPressureSystolic": blood_pressure_systolic_risk_values_after_training[false_positive_indices][0],
    "HighGlucose": glucose_values_after_training[false_positive_indices][0],
    "HighWBC": wbc_values_after_training[false_positive_indices][0],
    "LowMAP": map_values_after_training[false_positive_indices][0],
    "AgeRisk": age_values_after_training[false_positive_indices][0],
    "LactateNotClearing": lactate_values_after_training[false_positive_indices][0],
    "ChronicConditions": chronic_conditions[false_positive_indices][0],
    "LactateImpliesDeath": lactate_implies_death[false_positive_indices][0],
    "BilirubinImpliesDeath": bilirubin_implies_death[false_positive_indices][0],
    "CRPImpliesDeath": crp_implies_death[false_positive_indices][0],
    "WBCImpliesDeath": wbc_implies_death[false_positive_indices][0],
    "PlateletImpliesDeath": platelet_implies_death[false_positive_indices][0],
    "MAPImpliesDeath": map_implies_death[false_positive_indices][0],
    "AgeImpliesDeath": age_implies_death[false_positive_indices][0],
    "LactateNotClearingImpliesDeath": lactate_not_clearing_implies_death[false_positive_indices][0],
    "CronicConditionsImpliesDeath": cronic_conditions_implies_death[false_positive_indices][0],
    "actual": labels[false_positive_indices][0],
    "probability": mortality_probabilities[false_positive_indices][0],
    "case_id": cases_ids[false_positive_indices][0]
}

print(true_positive_values)
print(false_positive_values)


indices_gt = np.random.choice(np.where(labels == 1)[0], size=50, replace=False).tolist() + \
                np.random.choice(np.where(labels == 0)[0], size=50, replace=False).tolist()

# plot_concept_level_verification(
#     lactate_values_before_training[indices_lactate],
#     lactate_values_after_training[indices_lactate],
#     lactate_clinical_concept_values[indices_lactate],
#     bilirubin_values_before_training[indices_bilirubin],
#     bilirubin_values_after_training[indices_bilirubin],
#     bilirubin_clinical_concept_values[indices_bilirubin],
#     platelet_values_before_training[indices_platelet],
#     platelet_values_after_training[indices_platelet],
#     platelet_clinical_concept_values[indices_platelet],
#     creatinine_values_before_training[indices_gt],
#     creatinine_values_after_training[indices_gt],
#     actual_creatinine_values[indices_gt],
# )

plot_concept_level_verification_scatter(
    lactate_values_before_training[indices_gt],
    lactate_values_after_training[indices_gt],
    actual_lactate_values[indices_gt],
    bilirubin_values_before_training[indices_gt],
    bilirubin_values_after_training[indices_gt],
    actual_bilirubin_values[indices_gt],
    platelet_values_before_training[indices_gt],
    platelet_values_after_training[indices_gt],
    actual_platelet_values[indices_gt],
    wbc_values_before_training[indices_gt],
    wbc_values_after_training[indices_gt],
    actual_wbc_values[indices_gt],
    labels[indices_gt]
)

qSOFA_implies_death_values_d = []
lactate_implies_death_values_d = []
platelet_implies_death_values_d = []
bilirubin_implies_death_values_d = []
lactate_not_clearing_values_d = []
creatinine_implies_death_values_d = []
crp_implies_death_values_d = []
cronic_conditions_implies_death_values_d = []
low_map_implies_death_values_d = []
wbc_implies_death_values_d = []

qSOFA_implies_death_values_s = []
lactate_implies_death_values_s = []
platelet_implies_death_values_s = []
bilirubin_implies_death_values_s = []
lactate_not_clearing_values_s = []
creatinine_implies_death_values_s = []
crp_implies_death_values_s = []
cronic_conditions_implies_death_values_s = []
low_map_implies_death_values_s = []
wbc_implies_death_values_s = []

for x, y, c_id in test_loader:
    x_D = ltn.Variable("x_D", x[y==1])
    x_not_D = ltn.Variable("x_not_D", x[y==0])
    qSOFA_implies_death_values_d.append(And(RespiratoryRateRisk(respiratory_rate(x_D), comorbidities(x_D), age(x_D)), ArterialBloodPressureSystolicRisk(abps(x_D), comorbidities(x_D), age(x_D))).value.cpu().detach().numpy())
    qSOFA_implies_death_values_s.append(And(RespiratoryRateRisk(respiratory_rate(x_not_D), comorbidities(x_not_D), age(x_not_D)), ArterialBloodPressureSystolicRisk(abps(x_not_D), comorbidities(x_not_D), age(x_not_D))).value.cpu().detach().numpy())
    lactate_implies_death_values_d.append(LactateRisk(lactate(x_D), comorbidities(x_D), age(x_D)).value.cpu().detach().numpy())
    lactate_implies_death_values_s.append(LactateRisk(lactate(x_not_D), comorbidities(x_not_D), age(x_not_D)).value.cpu().detach().numpy())
    platelet_implies_death_values_d.append(PlateletLow(platelet(x_D), comorbidities(x_D), age(x_D)).value.cpu().detach().numpy())
    platelet_implies_death_values_s.append(PlateletLow(platelet(x_not_D), comorbidities(x_not_D), age(x_not_D)).value.cpu().detach().numpy())
    bilirubin_implies_death_values_d.append(HighBilirubin(bilirubin(x_D), comorbidities(x_D), age(x_D)).value.cpu().detach().numpy())
    bilirubin_implies_death_values_s.append(HighBilirubin(bilirubin(x_not_D), comorbidities(x_not_D), age(x_not_D)).value.cpu().detach().numpy())
    lactate_not_clearing_values_d.append(LactateNotClearing(lactate(x_D)).value.cpu().detach().numpy())
    lactate_not_clearing_values_s.append(LactateNotClearing(lactate(x_not_D)).value.cpu().detach().numpy())
    creatinine_implies_death_values_d.append(CreatinineRisk(creatinine(x_D), comorbidities(x_D), age(x_D)).value.cpu().detach().numpy())
    creatinine_implies_death_values_s.append(CreatinineRisk(creatinine(x_not_D), comorbidities(x_not_D), age(x_not_D)).value.cpu().detach().numpy())
    crp_implies_death_values_d.append(CRPRisk(crp(x_D), comorbidities(x_D), age(x_D)).value.cpu().detach().numpy())
    crp_implies_death_values_s.append(CRPRisk(crp(x_not_D), comorbidities(x_not_D), age(x_not_D)).value.cpu().detach().numpy())
    cronic_conditions_implies_death_values_d.append(CronicConditionsRisk(comorbidities(x_D)).value.cpu().detach().numpy())
    cronic_conditions_implies_death_values_s.append(CronicConditionsRisk(comorbidities(x_not_D)).value.cpu().detach().numpy())
    low_map_implies_death_values_d.append(MeanArterialPressureRisk(mabp(x_D), comorbidities(x_D), age(x_D)).value.cpu().detach().numpy())
    low_map_implies_death_values_s.append(MeanArterialPressureRisk(mabp(x_not_D), comorbidities(x_not_D), age(x_not_D)).value.cpu().detach().numpy())
    wbc_implies_death_values_d.append(WBCRisk(wbc(x_D), comorbidities(x_D), age(x_D)).value.cpu().detach().numpy())
    wbc_implies_death_values_s.append(WBCRisk(wbc(x_not_D), comorbidities(x_not_D), age(x_not_D)).value.cpu().detach().numpy())
# average the values

qSOFA_implies_death_values_d_ = np.concatenate(qSOFA_implies_death_values_d, axis=0)
qSOFA_implies_death_values_s = np.concatenate(qSOFA_implies_death_values_s, axis=0)
lactate_implies_death_values_d = np.concatenate(lactate_implies_death_values_d, axis=0)
lactate_implies_death_values_s = np.concatenate(lactate_implies_death_values_s, axis=0)
platelet_implies_death_values_d = np.concatenate(platelet_implies_death_values_d, axis=0)
platelet_implies_death_values_s = np.concatenate(platelet_implies_death_values_s, axis=0)
bilirubin_implies_death_values_d = np.concatenate(bilirubin_implies_death_values_d, axis=0)
bilirubin_implies_death_values_s = np.concatenate(bilirubin_implies_death_values_s, axis=0)
lactate_not_clearing_values_d = np.concatenate(lactate_not_clearing_values_d, axis=0)
lactate_not_clearing_values_s = np.concatenate(lactate_not_clearing_values_s, axis=0)
creatinine_implies_death_values_d = np.concatenate(creatinine_implies_death_values_d, axis=0)
creatinine_implies_death_values_s = np.concatenate(creatinine_implies_death_values_s, axis=0)
crp_implies_death_values_d = np.concatenate(crp_implies_death_values_d, axis=0)
crp_implies_death_values_s = np.concatenate(crp_implies_death_values_s, axis=0)
cronic_conditions_implies_death_values_d = np.concatenate(cronic_conditions_implies_death_values_d, axis=0)
cronic_conditions_implies_death_values_s = np.concatenate(cronic_conditions_implies_death_values_s, axis=0)
low_map_implies_death_values_d = np.concatenate(low_map_implies_death_values_d, axis=0)
low_map_implies_death_values_s = np.concatenate(low_map_implies_death_values_s, axis=0)
wbc_implies_death_values_d = np.concatenate(wbc_implies_death_values_d, axis=0)
wbc_implies_death_values_s = np.concatenate(wbc_implies_death_values_s, axis=0)

all_values = lactate_implies_death_values_d.tolist() + lactate_implies_death_values_s.tolist() + lactate_not_clearing_values_d.tolist() + lactate_not_clearing_values_s.tolist() + bilirubin_implies_death_values_d.tolist() + bilirubin_implies_death_values_s.tolist() + platelet_implies_death_values_d.tolist() + platelet_implies_death_values_s.tolist() + low_map_implies_death_values_d.tolist() + low_map_implies_death_values_s.tolist() + wbc_implies_death_values_d.tolist() + wbc_implies_death_values_s.tolist() + crp_implies_death_values_d.tolist() + crp_implies_death_values_s.tolist() + cronic_conditions_implies_death_values_d.tolist() + cronic_conditions_implies_death_values_s.tolist()
from plot_facts import plot_global_axioms_violin, plot_global_axioms_density
plot_global_axioms_density(all_values, len(lactate_implies_death_values_d.tolist()), len(lactate_implies_death_values_s.tolist()))