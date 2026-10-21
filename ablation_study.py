import pandas as pd
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
    parser.add_argument("--num_epochs_nesy", type=int, default=15, help="Number of epochs for training LTN model")
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

# print number of patients and mortality rate using the hospital admission id and hospital expire flag grouping the events by tha hadm_id
# data_grouped = data.groupby("hadm_id").agg({"hospital_expire_flag": "max"}).reset_index()
# num_patients = data_grouped["hadm_id"].nunique()
# mortality_rate = data_grouped["hospital_expire_flag"].mean()
# print(f"Number of patients: {num_patients}")
# print(f"Mortality rate: {mortality_rate:.2f}")

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

ltn_without_weak_anchoring_without_weighted_axiom_loss_accuracies = []
ltn_without_weak_anchoring_without_weighted_axiom_loss_f1_scores = []
ltn_without_weak_anchoring_without_weighted_axiom_loss_precisions = []
ltn_without_weak_anchoring_without_weighted_axiom_loss_recalls = []
ltn_without_weak_anchoring_without_weighted_axiom_loss_roc_aucs = []

ltn_without_weak_anchoring_with_weighted_axiom_loss_accuracies = []
ltn_without_weak_anchoring_with_weighted_axiom_loss_f1_scores = []
ltn_without_weak_anchoring_with_weighted_axiom_loss_precisions = []
ltn_without_weak_anchoring_with_weighted_axiom_loss_recalls = []
ltn_without_weak_anchoring_with_weighted_axiom_loss_roc_aucs = []

ltn_no_weights_accuracies = []
ltn_no_weights_f1_scores = []
ltn_no_weights_precisions = []
ltn_no_weights_recalls = []
ltn_no_weights_roc_aucs = []

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
    
    w_data = 0.8
    w_knowledge = 0.2

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

    #####################################

    # LTN with data and knowledge axioms and without weak anchoring and without weighted axiom loss

    # lstm = LSTMModel(vocab_sizes, config, 1, feature_names)
    # P = ltn.Predicate(lstm).to(device)
    # lstm.train()

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
    model_birulin = MLP(sequence_length, 64)
    HighBilirubin = ltn.Predicate(model_birulin).to(device)
    model_crp = MLP(sequence_length, 64)
    CRPRisk = ltn.Predicate(model_crp).to(device)
    model_platelet = MLP(sequence_length, 64)
    PlateletLow = ltn.Predicate(model_platelet).to(device)
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

    # params = list(P.parameters()) + list(LactateRisk.parameters()) + list(HighBilirubin.parameters()) + list(PlateletLow.parameters()) + list(LactateNotClearing.parameters()) + list(MeanArterialPressureRisk.parameters()) + list(CRPRisk.parameters()) + list(CronicConditionsRisk.parameters()) + list(WBCRisk.parameters()) + list(AgeRisk.parameters())

    # ltn_with_k_param_count = sum(p.numel() for p in params if p.requires_grad)
    # print(f"LTN with knowledge parameter count: {ltn_with_k_param_count:,}")

    # optimizer = torch.optim.Adam(params, lr=config.learning_rate)

    # model_lactate.train()
    # model_birulin.train()
    # model_platelet.train()
    # model_lactate_not_clearing.train()
    # model_cronic_conditions.train()
    # model_crp.train()
    # model_wbc.train()

    # # Functions modeling risk factors
    # lactate_above_threshold = lambda x: (x[:,(features_dict["Lactate"]*sequence_length-sequence_length):(features_dict["Lactate"]*sequence_length)] > scalers["Lactate"].transform(np.array([[4.0]]))[0][0]).any(axis=1)
    # age_above_threshold = lambda x: (x[:,(features_dict["anchor_age"]*sequence_length-sequence_length):(features_dict["anchor_age"]*sequence_length)][:, 0] > scalers["anchor_age"].transform(np.array([[65.0]]))[0][0])
    # glucose_above_threshold = lambda x: (x[:,(features_dict["Glucose"]*sequence_length-sequence_length):(features_dict["Glucose"]*sequence_length)] > scalers["Glucose"].transform(np.array([[100.0]]))[0][0]).any(axis=1)
    # bloodpressuresystolic_below_threshold = lambda x: (x[:,(features_dict["Arterial Blood Pressure systolic"]*sequence_length-sequence_length):(features_dict["Arterial Blood Pressure systolic"]*sequence_length)] <= scalers["Arterial Blood Pressure systolic"].transform(np.array([[100.0]]))[0][0]).any(axis=1)
    # creatinine_above_threshold = lambda x: (x[:,(features_dict["Creatinine (serum)"]*sequence_length-sequence_length):(features_dict["Creatinine (serum)"]*sequence_length)] >= scalers["Creatinine (serum)"].transform(np.array([[1.5]]))[0][0]).all(axis=1)
    # respiratory_rate_risk = lambda x: (x[:,(features_dict["Respiratory Rate"]*sequence_length-sequence_length):(features_dict["Respiratory Rate"]*sequence_length)] >= scalers["Respiratory Rate"].transform(np.array([[29.0]]))[0][0]).all(axis=1) | (x[:,(features_dict["Respiratory Rate"]*sequence_length-sequence_length):(features_dict["Respiratory Rate"]*sequence_length)] < scalers["Respiratory Rate"].transform(np.array([[9.0]]))[0][0]).all(axis=1)
    # blood_pressure_systolic_risk = lambda x: (x[:,(features_dict["Arterial Blood Pressure systolic"]*sequence_length-sequence_length):(features_dict["Arterial Blood Pressure systolic"]*sequence_length)] <= scalers["Arterial Blood Pressure systolic"].transform(np.array([[100.0]]))[0][0]).any(axis=1) & (x[:,(features_dict["Arterial Blood Pressure systolic"]*sequence_length-sequence_length):(features_dict["Arterial Blood Pressure systolic"]*sequence_length)] > scalers["Arterial Blood Pressure systolic"].transform(np.array([[0.0]]))[0][0]).all(axis=1)
    # blood_pressure_systolic_risk = lambda x: (x[:,(features_dict["Arterial Blood Pressure systolic"]*sequence_length-sequence_length):(features_dict["Arterial Blood Pressure systolic"]*sequence_length)] <= scalers["Arterial Blood Pressure systolic"].transform(np.array([[100.0]]))[0][0]).any(axis=1)
    # bilirubin_above_threshold = lambda x: (x[:,(features_dict["Total Bilirubin"]*sequence_length-sequence_length):(features_dict["Total Bilirubin"]*sequence_length)] >= scalers["Total Bilirubin"].transform(np.array([[2.0]]))[0][0]).any(axis=1)
    # low_gcs = lambda x: (x[:,(features_dict["gcs"]*sequence_length-sequence_length):(features_dict["gcs"]*sequence_length)] < scalers["gcs"].transform(np.array([[8.0]]))[0][0]).any(axis=1)
    # platelet_below_threshold = lambda x: (x[:,(features_dict["Platelet Count"]*sequence_length-sequence_length):(features_dict["Platelet Count"]*sequence_length)] < scalers["Platelet Count"].transform(np.array([[50.0]]))[0][0]).all(axis=1)
    # crp_above_threshold = lambda x: (x[:,(features_dict["C-Reactive Protein"]*sequence_length-sequence_length):(features_dict["C-Reactive Protein"]*sequence_length)] >= scalers["C-Reactive Protein"].transform(np.array([[100.0]]))[0][0]).any(axis=1)
    # map_below_threshold = lambda x: (x[:,(features_dict["Arterial Blood Pressure mean"]*sequence_length-sequence_length):(features_dict["Arterial Blood Pressure mean"]*sequence_length)] < scalers["Arterial Blood Pressure mean"].transform(np.array([[65.0]]))[0][0]).any(axis=1)
    # lactate_not_clearining = lambda x: np.array([
    #     (np.all(np.diff(seq[seq != 0]) >= 0) and np.any(seq > scalers["Lactate"].transform(np.array([[2.0]]))[0][0])) if np.sum(seq != 0) > 1 else False
    #     for seq in x[:, (features_dict["Lactate"]*sequence_length - sequence_length):(features_dict["Lactate"]*sequence_length)].cpu().numpy()
    # ])
    # wbc_above_threshold = lambda x: (x[:,(features_dict["White Blood Cells"]*sequence_length-sequence_length):(features_dict["White Blood Cells"]*sequence_length)] > scalers["White Blood Cells"].transform(np.array([[30.0]]))[0][0]).any(axis=1)
    # has_cronic_condition = lambda x : (x[:, -23:][:, 0] == 1) | (x[:, -23:][:, 1] == 1) | (x[:, -23:][:, 3] == 1) | (x[:, -23:][:, 4] == 1) | (x[:, -23:][:, 6] == 1) | (x[:, -23:][:, 7] == 1)  | (x[:, -23:][:, 6] == 8)  | (x[:, -23:][:, 12] == 1)  | (x[:, -23:][:, 14] == 1) | (x[:, -23:][:, 18] == 1)
    
    # best_f1_val = 0.0
    # count_early_stop = 0
    # for epoch in range(args.num_epochs_nesy):
    #     train_loss = 0.0
    #     for enum, (x, y, c_id) in enumerate(train_loader):
    #         optimizer.zero_grad()
    #         x_D = ltn.Variable("x_D", x[y==1])
    #         x_not_D = ltn.Variable("x_not_D", x[y==0])
    #         x_All = ltn.Variable("x_All", x)
    #         lactate_above_th = lactate_above_threshold(x)
    #         x_above_lactate = ltn.Variable("x_above_lactate", x[lactate_above_th==1])
    #         x_above_bilirubin = ltn.Variable("x_above_bilirubin", x[bilirubin_above_threshold(x)==1])
    #         x_below_bilirubin = ltn.Variable("x_below_bilirubin", x[bilirubin_above_threshold(x)==0])
    #         x_above_age = ltn.Variable("x_above_age", x[age_above_threshold(x)==1])
    #         x_below_platelet = ltn.Variable("x_below_platelet", x[platelet_below_threshold(x)==1])
    #         x_lactate_not_clear = ltn.Variable("x_lactate_not_clear", x[lactate_not_clearining(x)==1])
    #         x_lactate_clearing = ltn.Variable("x_lactate_clearing", x[lactate_not_clearining(x)==0])
    #         x_below_map = ltn.Variable("x_below_map", x[map_below_threshold(x)==1])
    #         x_above_crp = ltn.Variable("x_above_crp", x[crp_above_threshold(x)==1])
    #         x_cronic_condition = ltn.Variable("x_chronic_condition", x[has_cronic_condition(x)==1])
    #         x_above_wbc = ltn.Variable("x_above_wbc", x[wbc_above_threshold(x)==1])
    #         formulas = []
    #         formulas_knowledge = []
    #         if x_D.value.numel()>0:
    #             formulas.extend([
    #                 Forall(x_D, P(x_D)).value,
    #             ])
    #         if x_not_D.value.numel()>0:
    #             formulas.extend([
    #                 Forall(x_not_D, Not(P(x_not_D)), p=6).value,
    #             ])

    #         formulas.extend([
    #             Forall(x_All, Implies(LactateRisk(lactate(x_All), comorbidities(x_All), age(x_All)), P(x_All))).value,
    #             Forall(x_All, Implies(HighBilirubin(bilirubin(x_All), comorbidities(x_All), age(x_All)), P(x_All))).value,
    #             Forall(x_All, Implies(PlateletLow(platelet(x_All), comorbidities(x_All), age(x_All)), P(x_All))).value,
    #             Forall(x_All, Implies(LactateNotClearing(lactate(x_All)), P(x_All))).value,
    #             Forall(x_All, Implies(CRPRisk(crp(x_All), comorbidities(x_All), age(x_All)), P(x_All))).value,
    #             Forall(x_All, Implies(CronicConditionsRisk(comorbidities(x_All)), P(x_All))).value,
    #             Forall(x_All, Implies(WBCRisk(wbc(x_All), comorbidities(x_All), age(x_All)), P(x_All))).value,
    #             Forall(x_All, Implies(AgeRisk(age(x_All), comorbidities(x_All)), P(x_All))).value,
    #         ])
    #         sat_agg = SatAgg(*formulas)
    #         if len(formulas_knowledge) > 0:
    #             sat_agg_knowledge = SatAgg(*formulas_knowledge)
    #             loss = 1 - (w_data * sat_agg + w_knowledge * sat_agg_knowledge)
    #         else:
    #             loss = 1 - sat_agg
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.item()
    #         del x_D, x_not_D, sat_agg
    #     train_loss = train_loss / len(train_loader)
    #     f1_val = compute_accuracy(val_loader)
    #     if f1_val > best_f1_val:
    #         best_f1_val = f1_val
    #         torch.save(lstm.state_dict(), "ltn_w_k_now.pth")
    #         count_early_stop = 0
    #     else:
    #         count_early_stop += 1
    #         if epoch >=10 and count_early_stop > 20:
    #             print("Early stopping at epoch", epoch+1)
    #             break

    #     print(" epoch %d | loss %.4f | val f1 %.4f | test f1 %.4f" %(epoch+1, train_loss, f1_val, compute_accuracy(test_loader)))

    # # lstm.load_state_dict(torch.load("ltn_w_k_now.pth"))
    # lstm.eval()

    # print("Metrics LTN w/ knowledge no weak anchoring no weighted axiom loss")
    # accuracy_ltn_w_nowa, f1score_ltn_w_nowa, precision_ltn_w_nowa, recall_ltn_w_nowa, roc_auc_ltn_w_nowa, record = compute_metrics(test_loader, lstm, device, "ltn_w_k_now", scalers, features_dict, sequence_length)
    # print("Accuracy:", accuracy_ltn_w_nowa)
    # ltn_without_weak_anchoring_without_weighted_axiom_loss_accuracies.append(accuracy_ltn_w_nowa)
    # print("F1 Score:", f1score_ltn_w_nowa)
    # ltn_without_weak_anchoring_without_weighted_axiom_loss_f1_scores.append(f1score_ltn_w_nowa)
    # print("Precision:", precision_ltn_w_nowa)
    # ltn_without_weak_anchoring_without_weighted_axiom_loss_precisions.append(precision_ltn_w_nowa)
    # print("Recall:", recall_ltn_w_nowa)
    # ltn_without_weak_anchoring_without_weighted_axiom_loss_recalls.append(recall_ltn_w_nowa)
    # print("ROC AUC:", roc_auc_ltn_w_nowa)
    # ltn_without_weak_anchoring_without_weighted_axiom_loss_roc_aucs.append(roc_auc_ltn_w_nowa)

    # with open("full_metrics_separated.txt", "a") as f:
    #     f.write(f"Fold {i+1} LTN w/ knowledge no weak anchoring no weighted axiom loss Metrics:\n")
    #     f.write(f"Accuracy: {accuracy_ltn_w_nowa}\n")
    #     f.write(f"F1 Score: {f1score_ltn_w_nowa}\n")
    #     f.write(f"Precision: {precision_ltn_w_nowa}\n")
    #     f.write(f"Recall: {recall_ltn_w_nowa}\n")
    #     f.write(f"ROC AUC: {roc_auc_ltn_w_nowa}\n\n")

    #####################################

    # LTN with data and knowledge axioms and without weak anchoring and with weighted axiom loss
    
    # lstm = LSTMModel(vocab_sizes, config, 1, feature_names)
    # P = ltn.Predicate(lstm).to(device)
    # lstm.train()

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
    model_birulin = MLP(sequence_length, 64)
    HighBilirubin = ltn.Predicate(model_birulin).to(device)
    model_crp = MLP(sequence_length, 64)
    CRPRisk = ltn.Predicate(model_crp).to(device)
    model_platelet = MLP(sequence_length, 64)
    PlateletLow = ltn.Predicate(model_platelet).to(device)
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

    # params = list(P.parameters()) + list(LactateRisk.parameters()) + list(HighBilirubin.parameters()) + list(PlateletLow.parameters()) + list(LactateNotClearing.parameters()) + list(MeanArterialPressureRisk.parameters()) + list(CRPRisk.parameters()) + list(CronicConditionsRisk.parameters()) + list(WBCRisk.parameters()) + list(AgeRisk.parameters())

    # ltn_with_k_param_count = sum(p.numel() for p in params if p.requires_grad)
    # print(f"LTN with knowledge parameter count: {ltn_with_k_param_count:,}")

    # optimizer = torch.optim.Adam(params, lr=config.learning_rate)

    # model_lactate.train()
    # model_birulin.train()
    # model_platelet.train()
    # model_lactate_not_clearing.train()
    # model_cronic_conditions.train()
    # model_crp.train()
    # model_wbc.train()

    # # Functions modeling risk factors
    # lactate_above_threshold = lambda x: (x[:,(features_dict["Lactate"]*sequence_length-sequence_length):(features_dict["Lactate"]*sequence_length)] > scalers["Lactate"].transform(np.array([[4.0]]))[0][0]).any(axis=1)
    # age_above_threshold = lambda x: (x[:,(features_dict["anchor_age"]*sequence_length-sequence_length):(features_dict["anchor_age"]*sequence_length)][:, 0] > scalers["anchor_age"].transform(np.array([[65.0]]))[0][0])
    # glucose_above_threshold = lambda x: (x[:,(features_dict["Glucose"]*sequence_length-sequence_length):(features_dict["Glucose"]*sequence_length)] > scalers["Glucose"].transform(np.array([[100.0]]))[0][0]).any(axis=1)
    # bloodpressuresystolic_below_threshold = lambda x: (x[:,(features_dict["Arterial Blood Pressure systolic"]*sequence_length-sequence_length):(features_dict["Arterial Blood Pressure systolic"]*sequence_length)] <= scalers["Arterial Blood Pressure systolic"].transform(np.array([[100.0]]))[0][0]).any(axis=1)
    # creatinine_above_threshold = lambda x: (x[:,(features_dict["Creatinine (serum)"]*sequence_length-sequence_length):(features_dict["Creatinine (serum)"]*sequence_length)] >= scalers["Creatinine (serum)"].transform(np.array([[1.5]]))[0][0]).all(axis=1)
    # respiratory_rate_risk = lambda x: (x[:,(features_dict["Respiratory Rate"]*sequence_length-sequence_length):(features_dict["Respiratory Rate"]*sequence_length)] >= scalers["Respiratory Rate"].transform(np.array([[29.0]]))[0][0]).all(axis=1) | (x[:,(features_dict["Respiratory Rate"]*sequence_length-sequence_length):(features_dict["Respiratory Rate"]*sequence_length)] < scalers["Respiratory Rate"].transform(np.array([[9.0]]))[0][0]).all(axis=1)
    # blood_pressure_systolic_risk = lambda x: (x[:,(features_dict["Arterial Blood Pressure systolic"]*sequence_length-sequence_length):(features_dict["Arterial Blood Pressure systolic"]*sequence_length)] <= scalers["Arterial Blood Pressure systolic"].transform(np.array([[100.0]]))[0][0]).any(axis=1) & (x[:,(features_dict["Arterial Blood Pressure systolic"]*sequence_length-sequence_length):(features_dict["Arterial Blood Pressure systolic"]*sequence_length)] > scalers["Arterial Blood Pressure systolic"].transform(np.array([[0.0]]))[0][0]).all(axis=1)
    # blood_pressure_systolic_risk = lambda x: (x[:,(features_dict["Arterial Blood Pressure systolic"]*sequence_length-sequence_length):(features_dict["Arterial Blood Pressure systolic"]*sequence_length)] <= scalers["Arterial Blood Pressure systolic"].transform(np.array([[100.0]]))[0][0]).any(axis=1)
    # bilirubin_above_threshold = lambda x: (x[:,(features_dict["Total Bilirubin"]*sequence_length-sequence_length):(features_dict["Total Bilirubin"]*sequence_length)] >= scalers["Total Bilirubin"].transform(np.array([[2.0]]))[0][0]).any(axis=1)
    # low_gcs = lambda x: (x[:,(features_dict["gcs"]*sequence_length-sequence_length):(features_dict["gcs"]*sequence_length)] < scalers["gcs"].transform(np.array([[8.0]]))[0][0]).any(axis=1)
    # platelet_below_threshold = lambda x: (x[:,(features_dict["Platelet Count"]*sequence_length-sequence_length):(features_dict["Platelet Count"]*sequence_length)] < scalers["Platelet Count"].transform(np.array([[50.0]]))[0][0]).all(axis=1)
    # crp_above_threshold = lambda x: (x[:,(features_dict["C-Reactive Protein"]*sequence_length-sequence_length):(features_dict["C-Reactive Protein"]*sequence_length)] >= scalers["C-Reactive Protein"].transform(np.array([[100.0]]))[0][0]).any(axis=1)
    # map_below_threshold = lambda x: (x[:,(features_dict["Arterial Blood Pressure mean"]*sequence_length-sequence_length):(features_dict["Arterial Blood Pressure mean"]*sequence_length)] < scalers["Arterial Blood Pressure mean"].transform(np.array([[65.0]]))[0][0]).any(axis=1)
    # lactate_not_clearining = lambda x: np.array([
    #     (np.all(np.diff(seq[seq != 0]) >= 0) and np.any(seq > scalers["Lactate"].transform(np.array([[2.0]]))[0][0])) if np.sum(seq != 0) > 1 else False
    #     for seq in x[:, (features_dict["Lactate"]*sequence_length - sequence_length):(features_dict["Lactate"]*sequence_length)].cpu().numpy()
    # ])
    # wbc_above_threshold = lambda x: (x[:,(features_dict["White Blood Cells"]*sequence_length-sequence_length):(features_dict["White Blood Cells"]*sequence_length)] > scalers["White Blood Cells"].transform(np.array([[30.0]]))[0][0]).any(axis=1)
    # has_cronic_condition = lambda x : (x[:, -23:][:, 0] == 1) | (x[:, -23:][:, 1] == 1) | (x[:, -23:][:, 3] == 1) | (x[:, -23:][:, 4] == 1) | (x[:, -23:][:, 6] == 1) | (x[:, -23:][:, 7] == 1)  | (x[:, -23:][:, 6] == 8)  | (x[:, -23:][:, 12] == 1)  | (x[:, -23:][:, 14] == 1) | (x[:, -23:][:, 18] == 1)
    # best_f1_val = 0.0
    # count_early_stop = 0
    # for epoch in range(args.num_epochs_nesy):
    #     train_loss = 0.0
    #     for enum, (x, y, c_id) in enumerate(train_loader):
    #         optimizer.zero_grad()
    #         x_D = ltn.Variable("x_D", x[y==1])
    #         x_not_D = ltn.Variable("x_not_D", x[y==0])
    #         x_All = ltn.Variable("x_All", x)
    #         lactate_above_th = lactate_above_threshold(x)
    #         x_above_lactate = ltn.Variable("x_above_lactate", x[lactate_above_th==1])
    #         x_above_bilirubin = ltn.Variable("x_above_bilirubin", x[bilirubin_above_threshold(x)==1])
    #         x_below_bilirubin = ltn.Variable("x_below_bilirubin", x[bilirubin_above_threshold(x)==0])
    #         x_above_age = ltn.Variable("x_above_age", x[age_above_threshold(x)==1])
    #         x_below_platelet = ltn.Variable("x_below_platelet", x[platelet_below_threshold(x)==1])
    #         x_lactate_not_clear = ltn.Variable("x_lactate_not_clear", x[lactate_not_clearining(x)==1])
    #         x_lactate_clearing = ltn.Variable("x_lactate_clearing", x[lactate_not_clearining(x)==0])
    #         x_below_map = ltn.Variable("x_below_map", x[map_below_threshold(x)==1])
    #         x_above_crp = ltn.Variable("x_above_crp", x[crp_above_threshold(x)==1])
    #         x_cronic_condition = ltn.Variable("x_chronic_condition", x[has_cronic_condition(x)==1])
    #         x_above_wbc = ltn.Variable("x_above_wbc", x[wbc_above_threshold(x)==1])
    #         formulas = []
    #         formulas_knowledge = []
    #         if x_D.value.numel()>0:
    #             formulas.extend([
    #                 Forall(x_D, P(x_D)).value,
    #             ])
    #         if x_not_D.value.numel()>0:
    #             formulas.extend([
    #                 Forall(x_not_D, Not(P(x_not_D)), p=6).value,
    #             ])
    #         # Define knowledge rules as a list of tuples: (variable, condition_result, predicate_call)
    #         knowledge_rules = [
    #             (x_above_lactate, lactate_above_th, 
    #              lambda v: LactateRisk(lactate(v), comorbidities(v), age(v))),
    #             (x_below_platelet, platelet_below_threshold(x), 
    #              lambda v: PlateletLow(platelet(v), comorbidities(v), age(v))),
    #             (x_lactate_not_clear, lactate_not_clearining(x), 
    #              lambda v: LactateNotClearing(lactate(v))),
    #             (x_above_bilirubin, bilirubin_above_threshold(x), 
    #              lambda v: HighBilirubin(bilirubin(v), comorbidities(v), age(v))),
    #             (x_above_crp, crp_above_threshold(x), 
    #              lambda v: CRPRisk(crp(v), comorbidities(v), age(v))),
    #             (x_cronic_condition, has_cronic_condition(x), 
    #              lambda v: CronicConditionsRisk(comorbidities(v))),
    #             (x_above_wbc, wbc_above_threshold(x), 
    #              lambda v: WBCRisk(wbc(v), comorbidities(v), age(v))),
    #             (x_above_age, age_above_threshold(x), 
    #              lambda v: AgeRisk(age(v), comorbidities(v))),
    #         ]
            
    #         # Apply rules only when variables have elements
    #         for var, condition, predicate_fn in knowledge_rules:
    #             if var.value.numel() > 0:
    #                 formulas_knowledge.append(Forall(var, predicate_fn(var)).value)
            
    #         # Handle negation case separately
    #         if x_lactate_clearing.value.numel() > 0:
    #             formulas_knowledge.append(
    #                 Forall(x_lactate_clearing, Not(LactateNotClearing(lactate(x_lactate_clearing)))).value
    #             )

    #         formulas_knowledge.extend([
    #             Forall(x_All, Implies(LactateRisk(lactate(x_All), comorbidities(x_All), age(x_All)), P(x_All))).value,
    #             Forall(x_All, Implies(HighBilirubin(bilirubin(x_All), comorbidities(x_All), age(x_All)), P(x_All))).value,
    #             Forall(x_All, Implies(PlateletLow(platelet(x_All), comorbidities(x_All), age(x_All)), P(x_All))).value,
    #             Forall(x_All, Implies(LactateNotClearing(lactate(x_All)), P(x_All))).value,
    #             Forall(x_All, Implies(CRPRisk(crp(x_All), comorbidities(x_All), age(x_All)), P(x_All))).value,
    #             Forall(x_All, Implies(CronicConditionsRisk(comorbidities(x_All)), P(x_All))).value,
    #             Forall(x_All, Implies(WBCRisk(wbc(x_All), comorbidities(x_All), age(x_All)), P(x_All))).value,
    #             Forall(x_All, Implies(AgeRisk(age(x_All), comorbidities(x_All)), P(x_All))).value,
    #         ])
    #         sat_agg = SatAgg(*formulas)
    #         if len(formulas_knowledge) > 0:
    #             sat_agg_knowledge = SatAgg(*formulas_knowledge)
    #             loss = 1 - (w_data * sat_agg + w_knowledge * sat_agg_knowledge)
    #         else:
    #             loss = 1 - sat_agg
    #         loss = 1 - sat_agg
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.item()
    #         del x_D, x_not_D, sat_agg
    #     train_loss = train_loss / len(train_loader)
    #     f1_val = compute_accuracy(val_loader)
    #     if f1_val > best_f1_val:
    #         best_f1_val = f1_val
    #         torch.save(lstm.state_dict(), "ltn_w_k_now.pth")
    #         count_early_stop = 0
    #     else:
    #         count_early_stop += 1
    #         if epoch >=10 and count_early_stop > 20:
    #             print("Early stopping at epoch", epoch+1)
    #             break

    #     print(" epoch %d | loss %.4f | val f1 %.4f | test f1 %.4f" %(epoch+1, train_loss, f1_val, compute_accuracy(test_loader)))

    # # lstm.load_state_dict(torch.load("ltn_w_k_now.pth"))
    # lstm.eval()

    # print("Metrics LTN w/ knowledge no weak anchoring with weighted axiom loss")
    # accuracy_ltn_w_nowa_wal, f1score_ltn_w_nowa_wal, precision_ltn_w_nowa_wal, recall_ltn_w_nowa_wal, roc_auc_ltn_w_nowa_wal, record = compute_metrics(test_loader, lstm, device, "ltn_w_k_now", scalers, features_dict, sequence_length)
    # print("Accuracy:", accuracy_ltn_w_nowa_wal)
    # ltn_without_weak_anchoring_with_weighted_axiom_loss_accuracies.append(accuracy_ltn_w_nowa_wal)
    # print("F1 Score:", f1score_ltn_w_nowa_wal)
    # ltn_without_weak_anchoring_with_weighted_axiom_loss_f1_scores.append(f1score_ltn_w_nowa_wal)
    # print("Precision:", precision_ltn_w_nowa_wal)
    # ltn_without_weak_anchoring_with_weighted_axiom_loss_precisions.append(precision_ltn_w_nowa_wal)
    # print("Recall:", recall_ltn_w_nowa_wal)
    # ltn_without_weak_anchoring_with_weighted_axiom_loss_recalls.append(recall_ltn_w_nowa_wal)
    # print("ROC AUC:", roc_auc_ltn_w_nowa_wal)
    # ltn_without_weak_anchoring_with_weighted_axiom_loss_roc_aucs.append(roc_auc_ltn_w_nowa_wal)

    # with open("full_metrics_separated.txt", "a") as f:
    #     f.write(f"Fold {i+1} LTN w/ knowledge no weak anchoring with weighted axiom loss Metrics:\n")
    #     f.write(f"Accuracy: {accuracy_ltn_w_nowa_wal}\n")
    #     f.write(f"F1 Score: {f1score_ltn_w_nowa_wal}\n")
    #     f.write(f"Precision: {precision_ltn_w_nowa_wal}\n")
    #     f.write(f"Recall: {recall_ltn_w_nowa_wal}\n")
    #     f.write(f"ROC AUC: {roc_auc_ltn_w_nowa_wal}\n\n")

    ######################################

    # LTN with weak anchoring and without weighted axiom loss

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
    model_birulin = MLP(sequence_length, 64)
    HighBilirubin = ltn.Predicate(model_birulin).to(device)
    model_crp = MLP(sequence_length, 64)
    CRPRisk = ltn.Predicate(model_crp).to(device)
    model_platelet = MLP(sequence_length, 64)
    PlateletLow = ltn.Predicate(model_platelet).to(device)
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

    params = list(P.parameters()) + list(LactateRisk.parameters()) + list(HighBilirubin.parameters()) + list(PlateletLow.parameters()) + list(LactateNotClearing.parameters()) + list(MeanArterialPressureRisk.parameters()) + list(CRPRisk.parameters()) + list(CronicConditionsRisk.parameters()) + list(WBCRisk.parameters()) + list(AgeRisk.parameters())

    optimizer = torch.optim.Adam(params, lr=config.learning_rate)

    model_lactate.train()
    model_birulin.train()
    model_platelet.train()
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
        for seq in x[:, (features_dict["Lactate"]*sequence_length - sequence_length):(features_dict["Lactate"]*sequence_length)].cpu().numpy()
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
            x_above_bilirubin = ltn.Variable("x_above_bilirubin", x[bilirubin_above_threshold(x)==1])
            x_below_bilirubin = ltn.Variable("x_below_bilirubin", x[bilirubin_above_threshold(x)==0])
            x_above_age = ltn.Variable("x_above_age", x[age_above_threshold(x)==1])
            x_below_platelet = ltn.Variable("x_below_platelet", x[platelet_below_threshold(x)==1])
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
            # Define knowledge rules as a list of tuples: (variable, condition_result, predicate_call)
            knowledge_rules = [
                (x_above_lactate, lactate_above_th, 
                 lambda v: LactateRisk(lactate(v), comorbidities(v), age(v))),
                (x_below_platelet, platelet_below_threshold(x), 
                 lambda v: PlateletLow(platelet(v), comorbidities(v), age(v))),
                (x_lactate_not_clear, lactate_not_clearining(x), 
                 lambda v: LactateNotClearing(lactate(v))),
                (x_above_bilirubin, bilirubin_above_threshold(x), 
                 lambda v: HighBilirubin(bilirubin(v), comorbidities(v), age(v))),
                (x_above_crp, crp_above_threshold(x), 
                 lambda v: CRPRisk(crp(v), comorbidities(v), age(v))),
                (x_cronic_condition, has_cronic_condition(x), 
                 lambda v: CronicConditionsRisk(comorbidities(v))),
                (x_above_wbc, wbc_above_threshold(x), 
                 lambda v: WBCRisk(wbc(v), comorbidities(v), age(v))),
                (x_above_age, age_above_threshold(x), 
                 lambda v: AgeRisk(age(v), comorbidities(v))),
            ]
            
            # Apply rules only when variables have elements
            for var, condition, predicate_fn in knowledge_rules:
                if var.value.numel() > 0:
                    formulas.append(Forall(var, predicate_fn(var)).value)
            
            # Handle negation case separately
            if x_lactate_clearing.value.numel() > 0:
                formulas.append(
                    Forall(x_lactate_clearing, Not(LactateNotClearing(lactate(x_lactate_clearing)))).value
                )

            formulas.extend([
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

    # lstm.load_state_dict(torch.load("ltn_w_k.pth"))
    lstm.eval()

    print("Metrics LTN w/ knowledge")
    accuracy_ltn_w, f1score_ltn_w, precision_ltn_w, recall_ltn_w, roc_auc_ltn_w, prc_auc_ltn_w, record = compute_metrics(test_loader, lstm, device, "ltn_w_k", scalers, features_dict, sequence_length)
    print("Accuracy:", accuracy_ltn_w)
    ltn_no_weights_accuracies.append(accuracy_ltn_w)
    print("F1 Score:", f1score_ltn_w)
    ltn_no_weights_f1_scores.append(f1score_ltn_w)
    print("Precision:", precision_ltn_w)
    ltn_no_weights_precisions.append(precision_ltn_w)
    print("Recall:", recall_ltn_w)
    ltn_no_weights_recalls.append(recall_ltn_w)
    print("ROC AUC:", roc_auc_ltn_w)
    ltn_no_weights_roc_aucs.append(roc_auc_ltn_w)
    print("PRC AUC:", prc_auc_ltn_w)

    with open("full_metrics_separated.txt", "a") as f:
        f.write(f"Fold {i+1} LTN w/ weak anchoring and without weighted axiom loss Metrics:\n")
        f.write(f"Accuracy: {accuracy_ltn_w}\n")
        f.write(f"F1 Score: {f1score_ltn_w}\n")
        f.write(f"Precision: {precision_ltn_w}\n")
        f.write(f"Recall: {recall_ltn_w}\n")
        f.write(f"ROC AUC: {roc_auc_ltn_w}\n\n")

# write mean and std of the metrics to a file
with open("ablation_stratified_results.txt", "a") as f:

    f.write("LTN Results without weak anchoring and without weighted Knowledge:\n")
    f.write(f"Accuracy: {np.mean(ltn_without_weak_anchoring_without_weighted_axiom_loss_accuracies)*100:.2f} ± {np.std(ltn_without_weak_anchoring_without_weighted_axiom_loss_accuracies)*100:.2f}\n")
    f.write(f"F1 Score: {np.mean(ltn_without_weak_anchoring_without_weighted_axiom_loss_f1_scores)*100:.2f} ± {np.std(ltn_without_weak_anchoring_without_weighted_axiom_loss_f1_scores)*100:.2f}\n")
    f.write(f"Precision: {np.mean(ltn_without_weak_anchoring_without_weighted_axiom_loss_precisions)*100:.2f} ± {np.std(ltn_without_weak_anchoring_without_weighted_axiom_loss_precisions)*100:.2f}\n")
    f.write(f"Recall: {np.mean(ltn_without_weak_anchoring_without_weighted_axiom_loss_recalls)*100:.2f} ± {np.std(ltn_without_weak_anchoring_without_weighted_axiom_loss_recalls)*100:.2f}\n")
    f.write(f"ROC AUC: {np.mean(ltn_without_weak_anchoring_without_weighted_axiom_loss_roc_aucs)*100:.2f} ± {np.std(ltn_without_weak_anchoring_without_weighted_axiom_loss_roc_aucs)*100:.2f}\n\n")

    f.write("LTN Results without weak anchoring and with weighted Knowledge:\n")
    f.write(f"Accuracy: {np.mean(ltn_without_weak_anchoring_with_weighted_axiom_loss_accuracies)*100:.2f} ± {np.std(ltn_without_weak_anchoring_with_weighted_axiom_loss_accuracies)*100:.2f}\n")
    f.write(f"F1 Score: {np.mean(ltn_without_weak_anchoring_with_weighted_axiom_loss_f1_scores)*100:.2f} ± {np.std(ltn_without_weak_anchoring_with_weighted_axiom_loss_f1_scores)*100:.2f}\n")
    f.write(f"Precision: {np.mean(ltn_without_weak_anchoring_with_weighted_axiom_loss_precisions)*100:.2f} ± {np.std(ltn_without_weak_anchoring_with_weighted_axiom_loss_precisions)*100:.2f}\n")
    f.write(f"Recall: {np.mean(ltn_without_weak_anchoring_with_weighted_axiom_loss_recalls)*100:.2f} ± {np.std(ltn_without_weak_anchoring_with_weighted_axiom_loss_recalls)*100:.2f}\n")
    f.write(f"ROC AUC: {np.mean(ltn_without_weak_anchoring_with_weighted_axiom_loss_roc_aucs)*100:.2f} ± {np.std(ltn_without_weak_anchoring_with_weighted_axiom_loss_roc_aucs)*100:.2f}\n\n")

    f.write("LTN Results with weak anchoring and without weighted Knowledge:\n")
    f.write(f"Accuracy: {np.mean(ltn_no_weights_accuracies)*100:.2f} ± {np.std(ltn_no_weights_accuracies)*100:.2f}\n")
    f.write(f"F1 Score: {np.mean(ltn_no_weights_f1_scores)*100:.2f} ± {np.std(ltn_no_weights_f1_scores)*100:.2f}\n")
    f.write(f"Precision: {np.mean(ltn_no_weights_precisions)*100:.2f} ± {np.std(ltn_no_weights_precisions)*100:.2f}\n")
    f.write(f"Recall: {np.mean(ltn_no_weights_recalls)*100:.2f} ± {np.std(ltn_no_weights_recalls)*100:.2f}\n")  
    f.write(f"ROC AUC: {np.mean(ltn_no_weights_roc_aucs)*100:.2f} ± {np.std(ltn_no_weights_roc_aucs)*100:.2f}\n\n")