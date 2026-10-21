import argparse
import contextlib
import copy
import io
import json
import os
import random
import sys
import time

import ltn
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from xgboost import XGBClassifier

from data.dataset import ModelConfig, SepsisDataset
from data.preprocessing import preprocess_eventlog_external
from model.models import MLP, LSTMModel, SimpleMLP, SimpleMLPAge

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = ["xgboost", "rf", "lstm", "ltn", "ltn_k"]
MODEL_NAMES = {"xgboost": "XGBoost", "rf": "RF", "lstm": "LSTM", "ltn": "LTN w/o K", "ltn_k": "LTN w/ K"}
METRICS = ["accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc"]
COLUMNS_TO_DROP = ["SOFA Score", "APACHEII-Renal failure", "Urine output_ApacheIV", "APACHE II"]
W_DATA = 0.8
W_KNOWLEDGE = 0.2
GLUCOSE_THRESHOLD = 180.0  # mg/dL, glucose risk rule of LTN w/ K
# comorbidity columns (of the last 23 features) that make CronicConditionsRisk true
CHRONIC_CONDITIONS = [0, 1, 3, 4, 6, 7, 12, 14, 18]


def get_args():
    parser = argparse.ArgumentParser()

    # general network parameters
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size of the LSTM model")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the LSTM model")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for the LSTM model")
    parser.add_argument("--num_epochs", type=int, default=50, help="Maximum number of epochs for the LSTM model")
    parser.add_argument("--num_epochs_nesy", type=int, default=50, help="Maximum number of epochs for LTN w/o knowledge")
    parser.add_argument("--num_epochs_nesy_k", type=int, default=50, help="Maximum number of epochs for LTN w/ knowledge")
    parser.add_argument("--batch_size", type=int, default=32)
    # early stopping (validation AUPRC)
    parser.add_argument("--patience", type=int, default=10,
                        help="Neural models: epochs without improvement of the validation AUPRC before stopping")
    parser.add_argument("--max_trees", type=int, default=1000,
                        help="Maximum number of boosting rounds (XGBoost) and of trees (Random Forest)")
    parser.add_argument("--xgb_patience", type=int, default=50,
                        help="XGBoost: boosting rounds without improvement of the validation AUPRC before stopping")
    parser.add_argument("--rf_step", type=int, default=25, help="Random Forest: trees added at every step")
    parser.add_argument("--rf_patience", type=int, default=4,
                        help="Random Forest: steps without improvement of the validation AUPRC before stopping")
    # experiment configuration
    parser.add_argument("--horizons", type=int, nargs="+", default=[6, 12, 24, 48], help="Hours before death/discharge")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.2, 0.5, 0.7, 0.9, 1.0],
                        help="Fractions of the training period used for training")
    parser.add_argument("--n_runs", type=int, default=5, help="Repetitions (different subsamples and initialisations)")
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--skip_external", action="store_true", help="Do not evaluate on the external eICU cohort")
    # temporal split
    parser.add_argument("--train_groups", nargs="+", default=["2008 - 2010", "2011 - 2013"])
    parser.add_argument("--val_groups", nargs="+", default=["2014 - 2016"])
    parser.add_argument("--test_groups", nargs="+", default=["2017 - 2019"])
    parser.add_argument("--patients_path", default="data/patients.csv")
    # cohort
    parser.add_argument("--first_admission", action="store_true",
                        help="Keep only admissions that are the first hospital admission of their patient")
    parser.add_argument("--admissions_path", default="../MIMIC/mimic-log-extraction/hosp/admissions.csv",
                        help="Admissions table (subject_id, hadm_id, admittime) used to rank the admissions of a patient")
    # risk predicates of LTN w/ K
    parser.add_argument("--predicates", default="mlp", choices=["mlp", "sigmoid"],
                        help="Grounding of the risk predicates of LTN w/ K (HighLactate, ...): neural networks (mlp) "
                             "or sigmoids of the clinical variable around its clinical threshold (sigmoid)")
    parser.add_argument("--sigmoid_width", type=float, default=0.25,
                        help="sigmoid: initial slope such that the truth goes from 0.12 to 0.88 within "
                             "± sigmoid_width * threshold of the clinical threshold")
    parser.add_argument("--freeze_sigmoid", action="store_true",
                        help="sigmoid: keep thresholds, slopes and comorbidity weights fixed at their initial values")
    # output
    parser.add_argument("--out_dir", default=None,
                        help="Default: results_temporal_train_size[_sigmoid][_first_admission]")
    parser.add_argument("--verbose", action="store_true", help="Print loss and validation F1 of every epoch")

    args = parser.parse_args()
    args.cohort = "first_admission" if args.first_admission else "all_admissions"
    if args.out_dir is None:
        args.out_dir = "results_temporal_train_size" + ("_sigmoid" if args.predicates == "sigmoid" else "") + \
                       ("_first_admission" if args.first_admission else "")
    return args


def log(msg=""):
    print(msg, flush=True)


def section(title):
    log(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def fmt_time(seconds):
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


@contextlib.contextmanager
def quiet():
    """Silence the debugging prints of the preprocessing code."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def parse_admittime(admittime):
    # MIMIC occasionally writes midnight as "24:00:00" of the previous day
    admittime = admittime.astype(str)
    midnight = admittime.str.endswith(" 24:00:00")
    parsed = pd.to_datetime(admittime.str.replace(" 24:00:00", " 00:00:00", regex=False), errors="coerce")
    return parsed + pd.to_timedelta(midnight.astype(int), unit="D")


def first_admission_ids(admissions_path):
    """hadm_ids that are the earliest admission (by admittime) of their patient in the admissions table."""
    admissions = pd.read_csv(admissions_path, usecols=["subject_id", "hadm_id", "admittime"],
                             dtype={"subject_id": str, "hadm_id": str})
    admissions["admittime"] = parse_admittime(admissions["admittime"])
    n_unparsed = admissions["admittime"].isna().sum()
    if n_unparsed > 0:
        log(f"   WARNING: {n_unparsed} admittimes could not be parsed and are ranked last")
    first = admissions.sort_values(["subject_id", "admittime", "hadm_id"], na_position="last") \
                      .drop_duplicates("subject_id", keep="first")
    return set(first["hadm_id"]), set(admissions["hadm_id"])


def filter_first_admission(data, first_ids, known_ids):
    """Drop the admissions that are not the first hospital admission of their patient."""
    hadm_ids = pd.Series(data["hadm_id"].unique())
    unknown = ~hadm_ids.isin(known_ids)
    keep = hadm_ids.isin(first_ids)
    if unknown.any():
        log(f"   WARNING: {unknown.sum()} admissions are missing from the admissions table and are dropped")
    labels = data.groupby("hadm_id")["hospital_expire_flag"].first()
    log(f"   first-admission filter: {len(hadm_ids)} -> {keep.sum()} admissions "
        f"({(~keep).sum()} dropped: {(~keep & ~unknown).sum()} readmissions, {unknown.sum()} unknown) | "
        f"mortality {labels.mean():.3f} -> {labels.loc[hadm_ids[keep].values].mean():.3f}")
    return data[data["hadm_id"].isin(set(hadm_ids[keep]))]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def temporal_split(data, patients, args):
    """Assign each hadm_id to train/val/test using the anchor_year_group of its patient."""
    groups = {"train": args.train_groups, "val": args.val_groups, "test": args.test_groups}
    all_groups = sum(groups.values(), [])
    if len(all_groups) != len(set(all_groups)):
        raise ValueError("train/val/test anchor_year_groups must not overlap")

    admissions = data.groupby("hadm_id")["subject_id"].first().reset_index()
    admissions = admissions.merge(patients[["subject_id", "anchor_year_group"]], on="subject_id", how="left")
    admissions["split"] = None
    for split, split_groups in groups.items():
        admissions.loc[admissions["anchor_year_group"].isin(split_groups), "split"] = split

    n_unassigned = admissions["split"].isna().sum()
    if n_unassigned > 0:
        log(f"   WARNING: {n_unassigned} admissions have no anchor_year_group in the split and are excluded")
    return admissions.dropna(subset=["split"])


def extract_features_fast(data_list):
    """Mean (excluding zeros) of every numerical feature over the sequence, as in main.py."""
    result = []
    for x in data_list:
        data = np.array(x[0][1:])
        data_masked = np.where(data == 0, np.nan, data)
        result.append(np.nanmean(data_masked, axis=1))
    return np.array(result)


def nested_subsets(y_train, fractions, seed):
    """Stratified, nested subsets of the training indices: the 10% subset is contained in the 20% one, etc."""
    rng = np.random.RandomState(seed)
    pos = rng.permutation(np.where(y_train == 1)[0])
    neg = rng.permutation(np.where(y_train == 0)[0])
    subsets = {}
    for fraction in fractions:
        n_pos = max(1, int(round(fraction * len(pos))))
        n_neg = max(1, int(round(fraction * len(neg))))
        subsets[fraction] = np.sort(np.concatenate([pos[:n_pos], neg[:n_neg]]))
    return subsets


def compute_scores(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob > threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def predict_proba(model, loader):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            y_prob.append(model(x.to(device)).squeeze(1).cpu().numpy())
            y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(y_prob)

def val_scores(y_true, y_prob):
    """Validation AUPRC (early-stopping criterion) and macro F1 at 0.5."""
    return {"val_pr_auc": average_precision_score(y_true, y_prob),
            "val_f1": f1_score(y_true, (y_prob > 0.5).astype(int), average="macro")}


def fit(model, run_epoch, val_loader, num_epochs, patience, name, verbose=False):
    """Run `run_epoch` for at most num_epochs keeping the weights with the best validation AUPRC; stop after
    `patience` epochs without improvement."""
    best, best_state, best_epoch, count_stop = None, None, 0, 0
    # transient progress bar, shown only on a terminal (not when the output is redirected to a log file)
    epochs = tqdm(range(num_epochs), desc=f"      {name}", leave=False, ncols=100,
                  disable=verbose or not sys.stderr.isatty())
    for epoch in epochs:
        model.train()
        train_loss = run_epoch()
        scores = val_scores(*predict_proba(model, val_loader))
        epochs.set_postfix(loss=f"{train_loss:.4f}", val_auprc=f"{scores['val_pr_auc']:.4f}")
        if verbose:
            log(f"      [{name}] epoch {epoch+1:3d}/{num_epochs} | loss {train_loss:.4f} | "
                f"val AUPRC {scores['val_pr_auc']:.4f} | val F1 {scores['val_f1']:.4f}")
        if best is None or scores["val_pr_auc"] > best["val_pr_auc"]:
            best, best_epoch = scores, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            count_stop = 0
        else:
            count_stop += 1
            if count_stop >= patience:
                break
    epochs.close()
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "epochs_run": epoch + 1, **best}


def risk_mlp(sequence_length):
    # MLP hard-codes the input width of fc1 for a single horizon: resize it to window + 23 comorbidities + age
    model = MLP(sequence_length, 64)
    model.fc1 = torch.nn.Linear(sequence_length + 23 + 1, 64)
    return model


class SigmoidThreshold(torch.nn.Module):
    """Risk predicate sigmoid(slope * (value - threshold)) of a clinical variable (slope < 0 for "below" rules),
    with learnable threshold and slope initialised at the clinical cut-off. Only observed values are used (missing
    measurements and padding are 0 after scaling): `aggregate` "any" takes the max of the truths over the window,
    "all" the min, and windows without observations are false."""

    def __init__(self, threshold, slope, aggregate="any", trainable=True):
        super().__init__()
        self.threshold = torch.nn.Parameter(torch.tensor(float(threshold)), requires_grad=trainable)
        self.log_slope = torch.nn.Parameter(torch.tensor(float(np.log(abs(slope)))), requires_grad=trainable)
        self.sign = float(np.sign(slope))
        self.aggregate = aggregate

    def truth(self, values):
        return torch.sigmoid(self.sign * self.log_slope.exp() * (values - self.threshold))

    def forward(self, values):
        if values.dim() == 1:  # a single value per patient (age)
            return self.truth(values)
        seen = values != 0
        truth = self.truth(values)
        if self.aggregate == "any":
            truth = truth.masked_fill(~seen, 0.0).max(dim=1).values
        else:
            truth = truth.masked_fill(~seen, 1.0).min(dim=1).values
        return truth * seen.any(dim=1)


class SigmoidNotClearing(torch.nn.Module):
    """Sigmoid version of LactateNotClearing: product of a sigmoid of the smallest increase between consecutive
    observed values (non-decreasing lactate) and a sigmoid of the highest value (lactate above the threshold).
    Windows with fewer than two observations are false."""

    def __init__(self, threshold, slope, trainable=True):
        super().__init__()
        self.high = SigmoidThreshold(threshold, slope, "any", trainable)
        self.min_increase = torch.nn.Parameter(torch.tensor(0.0), requires_grad=trainable)
        self.log_slope = torch.nn.Parameter(torch.tensor(float(np.log(slope))), requires_grad=trainable)

    def forward(self, values):
        seen = values != 0
        n_seen = seen.sum(dim=1)
        # observed values moved to the front of the window, in temporal order
        positions = torch.arange(values.shape[1], device=values.device).expand_as(values)
        order = torch.where(seen, positions, positions + values.shape[1]).argsort(dim=1)
        observed = values.gather(1, order)
        diffs = observed[:, 1:] - observed[:, :-1]
        valid = positions[:, 1:] < n_seen.unsqueeze(1)
        smallest = diffs.masked_fill(~valid, float("inf")).min(dim=1).values
        smallest = torch.where(n_seen > 1, smallest, torch.zeros_like(smallest))
        increasing = torch.sigmoid(self.log_slope.exp() * (smallest - self.min_increase))
        return increasing * self.high(values) * (n_seen > 1)


class SigmoidComorbidities(torch.nn.Module):
    """Sigmoid version of CronicConditionsRisk: sigmoid(slope * (weights . comorbidities - 0.5)), with weights
    initialised to 1 for the chronic conditions of the rule and 0 for the others, so that one condition gives 0.88."""

    def __init__(self, n_comorbidities, chronic, trainable=True):
        super().__init__()
        weights = torch.zeros(n_comorbidities)
        weights[chronic] = 1.0
        self.weights = torch.nn.Parameter(weights, requires_grad=trainable)
        self.threshold = torch.nn.Parameter(torch.tensor(0.5), requires_grad=trainable)
        self.log_slope = torch.nn.Parameter(torch.tensor(float(np.log(4.0))), requires_grad=trainable)

    def forward(self, comorbidities):
        return torch.sigmoid(self.log_slope.exp() * (comorbidities @ self.weights - self.threshold))


def train_xgboost(ctx, train_idx, args, seed):
    """XGBoost early-stopped on the validation AUPRC; best_epoch/epochs_run count boosting rounds."""
    model = XGBClassifier(n_estimators=args.max_trees, eval_metric="aucpr", early_stopping_rounds=args.xgb_patience,
                          random_state=seed)
    model.fit(ctx["X_xgb"]["train"][train_idx], ctx["y"]["train"][train_idx],
              eval_set=[(ctx["X_xgb"]["val"], ctx["y"]["val"])], verbose=False)
    # predict_proba uses the trees up to best_iteration
    predict = lambda split: (ctx["y"][split], model.predict_proba(ctx["X_xgb"][split])[:, 1])
    info = {"best_epoch": model.best_iteration + 1, "epochs_run": model.get_booster().num_boosted_rounds()}
    return predict, {**info, **val_scores(*predict("val"))}


def train_random_forest(ctx, train_idx, args, seed):
    """Random Forest grown args.rf_step trees at a time (warm start) and truncated to the number of trees with
    the best validation AUPRC; best_epoch/epochs_run count trees."""
    X_train, y_train = ctx["X_xgb"]["train"][train_idx], ctx["y"]["train"][train_idx]
    model = RandomForestClassifier(n_estimators=0, warm_start=True, n_jobs=-1, random_state=seed)
    best, best_trees, count_stop = None, 0, 0
    for n_trees in range(args.rf_step, args.max_trees + 1, args.rf_step):
        model.set_params(n_estimators=n_trees)
        model.fit(X_train, y_train)
        scores = val_scores(ctx["y"]["val"], model.predict_proba(ctx["X_xgb"]["val"])[:, 1])
        if args.verbose:
            log(f"      [RF] trees {n_trees:4d}/{args.max_trees} | val AUPRC {scores['val_pr_auc']:.4f} | "
                f"val F1 {scores['val_f1']:.4f}")
        if best is None or scores["val_pr_auc"] > best["val_pr_auc"]:
            best, best_trees, count_stop = scores, n_trees, 0
        else:
            count_stop += 1
            if count_stop >= args.rf_patience:
                break
    # the forest averages its trees: keeping the first best_trees gives exactly the best validated model
    model.estimators_ = model.estimators_[:best_trees]
    model.n_estimators = best_trees
    predict = lambda split: (ctx["y"][split], model.predict_proba(ctx["X_xgb"][split])[:, 1])
    return predict, {"best_epoch": best_trees, "epochs_run": n_trees, **best}


def train_lstm(ctx, train_loader, val_loader, args):
    lstm = LSTMModel(ctx["vocab_sizes"], ctx["config"], 1, ctx["feature_names"]).to(device)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=ctx["config"].learning_rate)
    criterion = torch.nn.BCELoss()

    def run_epoch():
        losses = []
        for x, y, _ in train_loader:
            optimizer.zero_grad()
            loss = criterion(lstm(x.to(device)).squeeze(1).cpu(), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return np.mean(losses)

    return fit(lstm, run_epoch, val_loader, args.num_epochs, args.patience, "LSTM", args.verbose)


def train_ltn(ctx, train_loader, val_loader, args):
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    SatAgg = ltn.fuzzy_ops.SatAgg()

    lstm = LSTMModel(ctx["vocab_sizes"], ctx["config"], 1, ctx["feature_names"]).to(device)
    P = ltn.Predicate(lstm).to(device)
    optimizer = torch.optim.Adam(P.parameters(), lr=ctx["config"].learning_rate)

    def run_epoch():
        train_loss = 0.0
        for x, y, _ in train_loader:
            optimizer.zero_grad()
            x_D = ltn.Variable("x_D", x[y == 1])
            x_not_D = ltn.Variable("x_not_D", x[y == 0])
            formulas = []
            if x_D.value.numel() > 0:
                formulas.append(Forall(x_D, P(x_D)).value)
            if x_not_D.value.numel() > 0:
                formulas.append(Forall(x_not_D, Not(P(x_not_D))).value)
            loss = 1 - SatAgg(*formulas)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        return train_loss / len(train_loader)

    return fit(lstm, run_epoch, val_loader, args.num_epochs_nesy, args.patience, "LTN w/o K", args.verbose)


def train_ltn_knowledge(ctx, train_loader, val_loader, args):
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
    SatAgg = ltn.fuzzy_ops.SatAgg()

    seq_len = ctx["sequence_length"]
    features_dict = ctx["features_dict"]
    scalers = ctx["scalers"]

    def window(feature):
        i = features_dict[feature]
        return slice(i * seq_len - seq_len, i * seq_len)

    def scaled(feature, value):
        return scalers[feature].transform(np.array([[value]]))[0][0]

    lstm = LSTMModel(ctx["vocab_sizes"], ctx["config"], 1, ctx["feature_names"]).to(device)
    P = ltn.Predicate(lstm).to(device)

    # Functions extracting the clinical variables
    lactate = ltn.Function(func=lambda x: x[:, window("Lactate")])
    bilirubin = ltn.Function(func=lambda x: x[:, window("Total Bilirubin")])
    platelet = ltn.Function(func=lambda x: x[:, window("Platelet Count")])
    crp = ltn.Function(func=lambda x: x[:, window("C-Reactive Protein")])
    wbc = ltn.Function(func=lambda x: x[:, window("White Blood Cells")])
    gcs = ltn.Function(func=lambda x: x[:, window("gcs")])
    creatinine = ltn.Function(func=lambda x: x[:, window("Creatinine (serum)")])
    glucose = ltn.Function(func=lambda x: x[:, window("Glucose")])
    age = ltn.Function(func=lambda x: x[:, window("anchor_age")][:, 0])
    comorbidities = ltn.Function(func=lambda x: x[:, -23:])

    # Predicates modeling risk factors
    if args.predicates == "mlp":
        LactateRisk = ltn.Predicate(risk_mlp(seq_len)).to(device)
        HighBilirubin = ltn.Predicate(risk_mlp(seq_len)).to(device)
        PlateletLow = ltn.Predicate(risk_mlp(seq_len)).to(device)
        CRPRisk = ltn.Predicate(risk_mlp(seq_len)).to(device)
        WBCRisk = ltn.Predicate(risk_mlp(seq_len)).to(device)
        GCSRisk = ltn.Predicate(risk_mlp(seq_len)).to(device)
        CreatinineRisk = ltn.Predicate(risk_mlp(seq_len)).to(device)
        GlucoseRisk = ltn.Predicate(risk_mlp(seq_len)).to(device)
        LactateNotClearing = ltn.Predicate(SimpleMLP(seq_len, 64)).to(device)
        CronicConditionsRisk = ltn.Predicate(SimpleMLP(23, 64)).to(device)
        AgeRisk = ltn.Predicate(SimpleMLPAge(24, 64)).to(device)

        lactate_risk = lambda v: LactateRisk(lactate(v), comorbidities(v), age(v))
        bilirubin_risk = lambda v: HighBilirubin(bilirubin(v), comorbidities(v), age(v))
        platelet_risk = lambda v: PlateletLow(platelet(v), comorbidities(v), age(v))
        lactate_not_clearing_risk = lambda v: LactateNotClearing(lactate(v))
        crp_risk = lambda v: CRPRisk(crp(v), comorbidities(v), age(v))
        chronic_risk = lambda v: CronicConditionsRisk(comorbidities(v))
        wbc_risk = lambda v: WBCRisk(wbc(v), comorbidities(v), age(v))
        age_risk = lambda v: AgeRisk(age(v), comorbidities(v))
        gcs_risk = lambda v: GCSRisk(gcs(v), comorbidities(v), age(v))
        creatinine_risk = lambda v: CreatinineRisk(creatinine(v), comorbidities(v), age(v))
        glucose_risk = lambda v: GlucoseRisk(glucose(v), comorbidities(v), age(v))
    else:
        trainable = not args.freeze_sigmoid

        def sigmoid(feature, value, direction=1, aggregate="any"):
            # slope in clinical units (logit ±2 at ± sigmoid_width * value), converted to the min-max scaled units
            slope = direction * 2.0 / (args.sigmoid_width * value) / scalers[feature].scale_[0]
            return SigmoidThreshold(scaled(feature, value), slope, aggregate, trainable)

        LactateRisk = ltn.Predicate(sigmoid("Lactate", 4.0)).to(device)
        HighBilirubin = ltn.Predicate(sigmoid("Total Bilirubin", 2.0)).to(device)
        PlateletLow = ltn.Predicate(sigmoid("Platelet Count", 50.0, -1, "all")).to(device)
        CRPRisk = ltn.Predicate(sigmoid("C-Reactive Protein", 100.0)).to(device)
        WBCRisk = ltn.Predicate(sigmoid("White Blood Cells", 30.0)).to(device)
        GCSRisk = ltn.Predicate(sigmoid("gcs", 8.0, -1)).to(device)
        CreatinineRisk = ltn.Predicate(sigmoid("Creatinine (serum)", 1.5, 1, "all")).to(device)
        GlucoseRisk = ltn.Predicate(sigmoid("Glucose", GLUCOSE_THRESHOLD)).to(device)
        lactate_high = sigmoid("Lactate", 2.0)
        LactateNotClearing = ltn.Predicate(SigmoidNotClearing(lactate_high.threshold.item(),
                                                              lactate_high.log_slope.exp().item(),
                                                              trainable)).to(device)
        CronicConditionsRisk = ltn.Predicate(SigmoidComorbidities(23, CHRONIC_CONDITIONS, trainable)).to(device)
        AgeRisk = ltn.Predicate(sigmoid("anchor_age", 65.0)).to(device)

        # each risk factor depends only on its own clinical variable (and on the comorbidities for the chronic one)
        lactate_risk = lambda v: LactateRisk(lactate(v))
        bilirubin_risk = lambda v: HighBilirubin(bilirubin(v))
        platelet_risk = lambda v: PlateletLow(platelet(v))
        lactate_not_clearing_risk = lambda v: LactateNotClearing(lactate(v))
        crp_risk = lambda v: CRPRisk(crp(v))
        chronic_risk = lambda v: CronicConditionsRisk(comorbidities(v))
        wbc_risk = lambda v: WBCRisk(wbc(v))
        age_risk = lambda v: AgeRisk(age(v))
        gcs_risk = lambda v: GCSRisk(gcs(v))
        creatinine_risk = lambda v: CreatinineRisk(creatinine(v))
        glucose_risk = lambda v: GlucoseRisk(glucose(v))

    predicates = [P, LactateRisk, HighBilirubin, PlateletLow, LactateNotClearing, CRPRisk, CronicConditionsRisk,
                  WBCRisk, AgeRisk, GCSRisk, CreatinineRisk, GlucoseRisk]
    optimizer = torch.optim.Adam([p for pred in predicates for p in pred.parameters() if p.requires_grad],
                                 lr=ctx["config"].learning_rate)

    # Clinical thresholds grounding the risk predicates (same as main.py)
    lactate_above_threshold = lambda x: (x[:, window("Lactate")] > scaled("Lactate", 4.0)).any(axis=1)
    age_above_threshold = lambda x: x[:, window("anchor_age")][:, 0] > scaled("anchor_age", 65.0)
    bilirubin_above_threshold = lambda x: (x[:, window("Total Bilirubin")] >= scaled("Total Bilirubin", 2.0)).any(axis=1)
    platelet_below_threshold = lambda x: (x[:, window("Platelet Count")] < scaled("Platelet Count", 50.0)).all(axis=1)
    crp_above_threshold = lambda x: (x[:, window("C-Reactive Protein")] >= scaled("C-Reactive Protein", 100.0)).any(axis=1)
    wbc_above_threshold = lambda x: (x[:, window("White Blood Cells")] > scaled("White Blood Cells", 30.0)).any(axis=1)
    lactate_not_clearing = lambda x: torch.as_tensor(np.array([
        (np.all(np.diff(seq[seq != 0]) >= 0) and np.any(seq > scaled("Lactate", 2.0))) if np.sum(seq != 0) > 1 else False
        for seq in x[:, window("Lactate")].numpy()
    ], dtype=bool))
    # GCS, creatinine and glucose are tested on the observed values only: a missing measurement and the padding of
    # short sequences are 0 after scaling, which would otherwise read as a GCS < 8 or break the "every value" test
    def observed(x, feature):
        values = x[:, window(feature)]
        return values, values != 0
    def any_observed(x, feature, condition):
        values, seen = observed(x, feature)
        return (seen & condition(values)).any(axis=1)
    def all_observed(x, feature, condition):
        values, seen = observed(x, feature)
        return seen.any(axis=1) & (~seen | condition(values)).all(axis=1)
    gcs_below_threshold = lambda x: any_observed(x, "gcs", lambda v: v < scaled("gcs", 8.0))
    creatinine_above_threshold = lambda x: all_observed(x, "Creatinine (serum)",
                                                        lambda v: v >= scaled("Creatinine (serum)", 1.5))
    # 180 mg/dL (ICU hyperglycaemia): > 100 mg/dL flags ~80% of the patients with the cohort mortality, so it
    # carries no risk information
    glucose_above_threshold = lambda x: any_observed(x, "Glucose", lambda v: v > scaled("Glucose", GLUCOSE_THRESHOLD))
    has_cronic_condition = lambda x: (x[:, -23:][:, CHRONIC_CONDITIONS] == 1).any(axis=1)

    def run_epoch():
        train_loss = 0.0
        for x, y, _ in train_loader:
            optimizer.zero_grad()
            x_D = ltn.Variable("x_D", x[y == 1])
            x_not_D = ltn.Variable("x_not_D", x[y == 0])
            x_All = ltn.Variable("x_All", x)
            lnc = lactate_not_clearing(x)

            formulas = []
            if x_D.value.numel() > 0:
                formulas.append(Forall(x_D, P(x_D)).value)
            if x_not_D.value.numel() > 0:
                formulas.append(Forall(x_not_D, Not(P(x_not_D)), p=6).value)

            # patients satisfying a clinical threshold should satisfy the corresponding risk predicate
            groundings = [
                ("x_above_lactate", lactate_above_threshold(x), lactate_risk),
                ("x_below_platelet", platelet_below_threshold(x), platelet_risk),
                ("x_lactate_not_clear", lnc, lactate_not_clearing_risk),
                ("x_lactate_clearing", ~lnc, lambda v: Not(lactate_not_clearing_risk(v))),
                ("x_above_bilirubin", bilirubin_above_threshold(x), bilirubin_risk),
                ("x_above_crp", crp_above_threshold(x), crp_risk),
                ("x_chronic_condition", has_cronic_condition(x), chronic_risk),
                ("x_above_wbc", wbc_above_threshold(x), wbc_risk),
                ("x_above_age", age_above_threshold(x), age_risk),
                ("x_below_gcs", gcs_below_threshold(x), gcs_risk),
                ("x_above_creatinine", creatinine_above_threshold(x), creatinine_risk),
                ("x_above_glucose", glucose_above_threshold(x), glucose_risk),
            ]
            formulas_knowledge = []
            for var_name, mask, axiom in groundings:
                v = ltn.Variable(var_name, x[mask])
                if v.value.numel() > 0:
                    formulas_knowledge.append(Forall(v, axiom(v)).value)
            # every risk factor implies death
            for risk in [lactate_risk, bilirubin_risk, platelet_risk, lactate_not_clearing_risk, crp_risk,
                         chronic_risk, wbc_risk, age_risk, gcs_risk, creatinine_risk, glucose_risk]:
                formulas_knowledge.append(Forall(x_All, Implies(risk(x_All), P(x_All))).value)

            loss = 1 - (W_DATA * SatAgg(*formulas) + W_KNOWLEDGE * SatAgg(*formulas_knowledge))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        return train_loss / len(train_loader)

    return fit(lstm, run_epoch, val_loader, args.num_epochs_nesy_k, args.patience, "LTN w/ K", args.verbose)


def prepare_horizon(horizon, patients, args, first_admissions=None):
    log(f"-- Loading data ({horizon}h before death/discharge)")
    data = pd.read_csv(f"data/subset/events_{horizon}h_before_death_gcs.csv", dtype={"hadm_id": str, "subject_id": str})
    data_external = pd.read_csv(f"data/subset/eicu_final_sepsis3_dataset_lead_{horizon}h.csv", dtype={"hadm_id": str})
    data = data[~data["concept:name"].isin(COLUMNS_TO_DROP)]
    if first_admissions is not None:
        data = filter_first_admission(data, *first_admissions)

    admissions = temporal_split(data, patients, args)
    split_of = dict(zip(admissions["hadm_id"], admissions["split"]))
    train_ids = admissions.loc[admissions["split"] == "train", "hadm_id"].tolist()

    # scalers are fit on the training period only, so no information from the future leaks into training
    log("-- Preprocessing (scalers fit on the training period only)")
    start = time.time()
    with quiet():
        (X, y, feature_names), (X_external, y_external, _), vocab_sizes, scalers, sequence_length = \
            preprocess_eventlog_external(data, data_external, args.seed, False, fit_ids=train_ids)
    log(f"   done in {fmt_time(time.time() - start)} | sequence length {sequence_length} | {len(feature_names)} features")

    splits = ["train", "val", "test"] + ([] if args.skip_external else ["external"])
    samples = {s: [] for s in splits}
    labels = {s: [] for s in splits}
    for x, label in zip(X, y):
        split = split_of.get(x[2])
        if split is not None:
            samples[split].append(x)
            labels[split].append(label)
    if not args.skip_external:
        samples["external"], labels["external"] = X_external, y_external

    split_desc = {"train": ", ".join(args.train_groups), "val": ", ".join(args.val_groups),
                  "test": ", ".join(args.test_groups), "external": "eICU"}
    log(f"\n   {'split':<10}{'period':<28}{'admissions':>12}{'deaths':>9}{'mortality':>11}")
    for split in splits:
        log(f"   {split:<10}{split_desc[split]:<28}{len(labels[split]):>12}{int(np.sum(labels[split])):>9}"
            f"{np.mean(labels[split]):>11.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    kept_ids = {x[2] for split in ["train", "val", "test"] for x in samples[split]}
    admissions[admissions["hadm_id"].isin(kept_ids)].to_csv(
        os.path.join(args.out_dir, f"split_{horizon}h.csv"), index=False)

    # XGBoost/RF features must be extracted before SepsisDataset converts the sequences to tensors in place
    X_xgb = {s: extract_features_fast(samples[s]) for s in splits}
    datasets = {s: SepsisDataset(samples[s], labels[s], feature_names) for s in splits}

    config = ModelConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        sequence_length=sequence_length,
        num_epochs=args.num_epochs
    )

    return {
        "splits": splits,
        "X_xgb": X_xgb,
        "y": {s: np.array(labels[s]) for s in splits},
        "datasets": datasets,
        "feature_names": feature_names,
        "features_dict": {s: i for i, s in enumerate(feature_names, start=1)},
        "vocab_sizes": vocab_sizes,
        "scalers": scalers,
        "sequence_length": sequence_length,
        "config": config,
    }


def print_summary(results, horizon, n_runs):
    """AUROC / AUPRC (mean ± std over runs) per model and training fraction."""
    log(f"\n-- Summary {horizon}h (mean ± std over {n_runs} runs)")
    for eval_set in results["eval_set"].unique():
        for metric, metric_name in [("roc_auc", "AUROC"), ("pr_auc", "AUPRC")]:
            res = results[results["eval_set"] == eval_set]
            stats = res.groupby(["model", "fraction"])[metric].agg(["mean", "std"]).fillna(0)
            table = stats.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1).unstack("fraction")
            table = table.reindex([m for m in MODELS if m in table.index]).rename(index=MODEL_NAMES)
            table.columns = [f"{c:.0%}" for c in table.columns]
            table.index.name = None
            log(f"\n   {metric_name} on {eval_set}")
            log("   " + table.to_string().replace("\n", "\n   "))


def run_horizon(horizon, patients, args, first_admissions=None):
    section(f"HORIZON {horizon}h | cohort: {args.cohort}")
    start_horizon = time.time()
    ctx = prepare_horizon(horizon, patients, args, first_admissions)
    eval_splits = [s for s in ctx["splits"] if s not in ("train", "val")]
    val_loader = DataLoader(ctx["datasets"]["val"], batch_size=args.batch_size, shuffle=False)
    eval_loaders = {s: DataLoader(ctx["datasets"][s], batch_size=args.batch_size, shuffle=False) for s in eval_splits}
    tree_trainers = {"xgboost": train_xgboost, "rf": train_random_forest}
    nn_trainers = {"lstm": train_lstm, "ltn": train_ltn, "ltn_k": train_ltn_knowledge}

    results_path = os.path.join(args.out_dir, f"results_{horizon}h.csv")
    if os.path.exists(results_path):
        os.remove(results_path)

    n_steps, step = args.n_runs * len(args.fractions), 0
    for run in range(args.n_runs):
        seed = args.seed + run
        subsets = nested_subsets(ctx["y"]["train"], args.fractions, seed)
        for fraction in args.fractions:
            step += 1
            train_idx = subsets[fraction]
            n_train, n_train_pos = len(train_idx), int(ctx["y"]["train"][train_idx].sum())
            log(f"\n[{horizon}h | step {step}/{n_steps}] run {run+1}/{args.n_runs} (seed {seed}) | "
                f"train fraction {fraction:.0%} = {n_train} admissions ({n_train_pos} deaths)")
            header = "".join(f"{s + ' AUROC':>16}{s + ' AUPRC':>16}{s + ' F1':>12}" for s in eval_splits)
            log(f"   {'model':<11}{'best/run':>11}{'val AUPRC':>11}{'val F1':>8}{'time':>9}{header}")

            for model_name in args.models:
                set_seed(seed)
                start = time.time()
                if model_name in tree_trainers:
                    predict, info = tree_trainers[model_name](ctx, train_idx, args, seed)
                else:
                    train_loader = DataLoader(Subset(ctx["datasets"]["train"], train_idx),
                                              batch_size=args.batch_size, shuffle=True)
                    model, info = nn_trainers[model_name](ctx, train_loader, val_loader, args)
                    predict = lambda split: predict_proba(model, eval_loaders[split])
                train_time = time.time() - start

                rows, line = [], ""
                for split in eval_splits:
                    y_true, y_prob = predict(split)
                    scores = compute_scores(y_true, y_prob)
                    line += f"{scores['roc_auc']:>16.4f}{scores['pr_auc']:>16.4f}{scores['f1']:>12.4f}"
                    rows.append({"cohort": args.cohort, "predicates": args.predicates, "horizon": horizon, "fraction": fraction, "run": run,
                                 "seed": seed, "model": model_name, "eval_set": split, "n_train": n_train,
                                 "n_train_pos": n_train_pos, "n_eval": len(y_true), **scores, **info,
                                 "train_time_s": train_time})
                # epochs for the neural models, boosting rounds for XGBoost, trees for the Random Forest
                best_ep = f"{info['best_epoch']}/{info['epochs_run']}"
                log(f"   {MODEL_NAMES[model_name]:<11}{best_ep:>11}{info['val_pr_auc']:>11.4f}{info['val_f1']:>8.4f}"
                    f"{fmt_time(train_time):>9}{line}")
                pd.DataFrame(rows).to_csv(results_path, mode="a", header=not os.path.exists(results_path), index=False)

    results = pd.read_csv(results_path)
    summary = results.groupby(["horizon", "eval_set", "model", "fraction"])[METRICS + ["n_train"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.to_csv(os.path.join(args.out_dir, f"summary_{horizon}h.csv"))
    print_summary(results, horizon, args.n_runs)
    log(f"\n-- Horizon {horizon}h completed in {fmt_time(time.time() - start_horizon)} | results: {results_path}")


def main():
    args = get_args()
    start = time.time()
    section("TEMPORAL TRAIN-SIZE EXPERIMENT")
    log(f"   device      : {device}")
    log(f"   cohort      : {args.cohort}" + (f" (ranked on {args.admissions_path})" if args.first_admission else ""))
    log(f"   horizons    : {', '.join(f'{h}h' for h in args.horizons)}")
    log(f"   fractions   : {', '.join(f'{f:.0%}' for f in args.fractions)}")
    log(f"   models      : {', '.join(MODEL_NAMES[m] for m in args.models)}")
    log(f"   predicates  : {args.predicates} (risk predicates of LTN w/ K)" + (
        f", width {args.sigmoid_width}, {'frozen' if args.freeze_sigmoid else 'learnable'}"
        if args.predicates == "sigmoid" else ""))
    log(f"   runs        : {args.n_runs} (seeds {args.seed}..{args.seed + args.n_runs - 1})")
    log(f"   early stop  : validation AUPRC | neural: max {args.num_epochs}/{args.num_epochs_nesy}/"
        f"{args.num_epochs_nesy_k} epochs (LSTM/LTN/LTN w/ K), patience {args.patience} | XGBoost: max "
        f"{args.max_trees} rounds, patience {args.xgb_patience} | RF: max {args.max_trees} trees, "
        f"+{args.rf_step} per step, patience {args.rf_patience} steps")
    log(f"   split       : train {args.train_groups} | val {args.val_groups} | test {args.test_groups}")
    log(f"   external    : {'skipped' if args.skip_external else 'eICU'}")
    log(f"   output      : {args.out_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    patients = pd.read_csv(args.patients_path, dtype={"subject_id": str})
    first_admissions = None
    if args.first_admission:
        log(f"\n-- Ranking admissions of each patient by admittime ({args.admissions_path})")
        first_admissions = first_admission_ids(args.admissions_path)

    for horizon in args.horizons:
        run_horizon(horizon, patients, args, first_admissions)

    section(f"DONE in {fmt_time(time.time() - start)} | results in {args.out_dir}")


if __name__ == "__main__":
    main()
