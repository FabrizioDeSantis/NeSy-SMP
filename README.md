# KGNeSY: knowledge-guided neuro-symbolic mortality prediction in sepsis

KGNeSY predicts in-hospital mortality of ICU patients with sepsis from the events recorded before death or
discharge. It combines a neural sequence model with clinical knowledge through
[Logic Tensor Networks](https://github.com/tommasocarraro/LTNtorch) (LTN). An LSTM grounds the predicate
`P(x)`, "patient *x* dies". The model is trained to satisfy two kinds of axioms:

- **data axioms**: patients who died satisfy `P`, survivors satisfy `¬P`;
- **knowledge axioms**: clinical risk concepts (high lactate, low platelets, …) are learned as predicates, and
  each of them must imply death.

In the experiment scripts and the paper tables, the model with knowledge is called **NeSy-SMP** ("LTN w/
knowledge"). It is compared with an LSTM, an LTN without knowledge, XGBoost and Random Forest. Models are
trained on MIMIC-IV and validated externally on eICU, at four prediction horizons: 6, 12, 24 and 48 hours before
the outcome. The models are evaluated in two settings:

- **random splits**: a single train/validation/test split and 5-fold stratified cross-validation;
- **temporal split**: trained on earlier years of MIMIC-IV, tested on later years and on eICU, with learning
  curves over growing amounts of training data.

## Contents

- [Setup](#setup)
- [Data](#data)
- [Quick start](#quick-start)
- [How the model works](#how-the-model-works)
- [Temporal split and learning curves](#temporal-split-and-learning-curves)
- [Experiments](#experiments)
- [Repository layout](#repository-layout)

## Setup

The code needs Python 3.11 and the following versions, which are the ones it was tested with:

| Package | Version |
|---|---|
| torch | 2.8.0 |
| LTNtorch | 1.0.2 |
| numpy | 2.0.2 |
| pandas | 3.0.1 |
| scikit-learn | 1.4.2 |
| xgboost | 3.0.0 |
| matplotlib, seaborn, scipy, tqdm | any recent |

```bash
pip install torch==2.8.0 LTNtorch==1.0.2 numpy==2.0.2 pandas==3.0.1 scikit-learn==1.4.2 xgboost==3.0.0 \
    matplotlib seaborn scipy tqdm
```

Only the knowledge-graph and data-extraction scripts need these extra packages: `rdflib`, `pyvis` and `networkx`
for the knowledge graph, `PyClause` for rule mining, and `medspacy` for comorbidity extraction.

Run every script from the repository root. The scripts read their input through relative paths such as
`data/subset/...`.

## Data

The data comes from two databases on PhysioNet, both of which require credentialed access:
[MIMIC-IV](https://physionet.org/content/mimiciv/) for training and internal testing, and
[eICU-CRD](https://physionet.org/content/eicu-crd/) for external validation. The CSV files derived from them are
not distributed with the code. Each user must obtain access and rebuild them.

The models read one event log per horizon `h ∈ {6, 12, 24, 48}`:

| File | Content |
|---|---|
| `data/subset/events_<h>h_before_death_gcs.csv` | MIMIC-IV event log: one row per event (`hadm_id`, `time:timestamp`, `concept:name`, vital signs and lab values, GCS, admission data, 23 comorbidity flags, `hospital_expire_flag` label) |
| `data/subset/eicu_final_sepsis3_dataset_lead_<h>h.csv` | eICU Sepsis-3 cohort with the same columns |

For patients who died, each admission covers a 24-hour observation window that ends `h` hours before death. For
survivors, the 24-hour window is placed at random during the stay. See `data/extract_before_death.py`.

`data/preprocessing.py` (`preprocess_eventlog_external`) turns the logs into one sample per admission:

- every feature becomes a fixed-length time series, padded with zeros;
- numerical features are scaled with scalers fitted on MIMIC-IV, categorical ones are encoded;
- the 23 binary comorbidity flags are appended at the end of the sample.

`data/dataset.py` (`SepsisDataset`) flattens each sample into a single tensor laid out as
`[feature_1 window | feature_2 window | … | 23 comorbidities]`.

## Quick start

```bash
# single train/validation/test split (24 h horizon); XGBoost, LSTM, LTN w/o and w/ knowledge
python main.py

# 5-fold stratified cross-validation (6 h horizon); adds Random Forest, reports mean ± std
python stratified_main.py

# temporal split with learning curves, all horizons (long run), then the paper figures and tables
python temporal_train_size.py
python paper_results_temporal_train_size.py
```

To check that everything runs end to end, use one epoch per model:

```bash
python main.py --num_epochs 1 --num_epochs_nesy 1
```

`main.py` and `stratified_main.py` accept the same options (`temporal_train_size.py` has its own, see
[Temporal split and learning curves](#temporal-split-and-learning-curves)):

| Option | Default | Meaning |
|---|---|---|
| `--hidden_size` | `128` | LSTM hidden size |
| `--num_layers` | `2` | LSTM layers |
| `--dropout_rate` | `0.1` | LSTM dropout |
| `--num_epochs` | `25` (`main.py`), `50` (`stratified_main.py`) | Maximum epochs of the vanilla LSTM |
| `--num_epochs_nesy` | `50` | Maximum epochs of each LTN model |
| `--train_vanilla` | `True` | Train the vanilla LSTM (`True`/`False`) |
| `--train_nesy` | `True` | Train the two LTN models (`True`/`False`) |
| `--sampling` | `False` | Apply stratified sampling to the MIMIC-IV data |
| `--predicates` | `mlp` | Grounding of the risk predicates: `mlp` or `sigmoid` (see [Knowledge base](#knowledge-base-knowledgepy)) |
| `--sigmoid_width` | `0.25` | `sigmoid`: initial slope, so that truth goes from 0.12 to 0.88 within ± width × threshold |
| `--freeze_sigmoid` | off | `sigmoid`: keep thresholds, slopes and comorbidity weights fixed |

The horizon is set by `INTERNAL_DATA_PATH` and `EXTERNAL_DATA_PATH` at the top of each script.

**`main.py` outputs:**

- metrics on the internal test set and on eICU, printed to the terminal;
- `xgboost_feature_importance.png`, `confusion_matrix*.png` and `calibration_plots.pdf`;
- the IDs of deaths that only the model with knowledge caught;
- the checkpoints `lstm_best.pth`, `ltn_w_o_k.pth` and `ltn_w_k.pth`.

**`stratified_main.py` outputs:** `stratified_results.txt` (internal test folds) and
`stratified_results_external.txt` (eICU). Both report accuracy, macro F1, macro precision, macro recall and ROC
AUC as mean ± std over the folds. The script also prints the training and inference times and the parameter
counts.

## How the model works

### Neural backbone (`model/models.py`)

`LSTMModel` processes each sample as follows:

1. Categorical features are embedded, and numerical features are projected to the hidden size.
2. A 2-layer bidirectional LSTM with attention pooling reads the time series.
3. The pooled output is combined with a dense encoding of the comorbidity flags.
4. A sigmoid output gives the probability of death.

The same network is the vanilla LSTM baseline and, wrapped in `ltn.Predicate`, the mortality predicate `P`.
By default the risk predicates are small MLPs: `MLP`, `SimpleMLP` and `SimpleMLPAge`.

### Knowledge base (`knowledge.py`)

`ClinicalKnowledge` defines eleven risk concepts. Each one is a learnable predicate, grounded on the patients
that satisfy a crisp clinical rule:

| Predicate | Clinical rule used for grounding |
|---|---|
| `LactateRisk` | lactate > 4 mmol/L at least once |
| `HighBilirubin` | total bilirubin ≥ 2 mg/dL at least once |
| `PlateletLow` | platelets < 50 ×10³/µL throughout the window |
| `CRPRisk` | C-reactive protein ≥ 100 mg/L at least once |
| `WBCRisk` | white blood cells > 30 ×10³/µL at least once |
| `GCSRisk` | Glasgow Coma Scale < 8 at least once |
| `CreatinineRisk` | serum creatinine ≥ 1.5 mg/dL at every measurement |
| `GlucoseRisk` | glucose > 180 mg/dL at least once |
| `AgeRisk` | age > 65 |
| `ChronicConditionsRisk` | AKI, AIDS, CAD, cancer, cirrhosis, COPD, HIV, kidney disease or metastatic cancer |
| `LactateNotClearing` | at least two lactate measurements, never decreasing, above 2 mmol/L at some point (also grounded as false on the other patients) |

Thresholds are given in clinical units and converted to the scale of the fitted scalers. The GCS, creatinine and
glucose rules look only at observed values, because missing measurements and padding are 0 after scaling.

`--predicates` selects how the risk predicates are grounded:

- `mlp` (default): a neural network of the clinical variable, the comorbidity flags and the age. `AgeRisk` sees
  the age and the comorbidities, `ChronicConditionsRisk` the comorbidities, `LactateNotClearing` the lactate
  series.
- `sigmoid`: a sigmoid of the clinical variable alone, centered on its clinical threshold, with "at least once"
  and "every measurement" taken as the maximum and minimum over the observed values. Thresholds and slopes are
  learnable unless `--freeze_sigmoid` is given.

### Training objective (`training.py`)

The knowledge base contains three groups of axioms:

```text
data:         ∀ x_D  P(x_D)                      ∀ x_¬D  ¬P(x_¬D)
concepts:     ∀ x ∈ {rule holds}  Risk(x)        (weak anchoring of every risk predicate)
implications: ∀ x  Risk(x) → P(x)                for every risk concept
```

The quantifiers use the pMeanError aggregator (p = 2), the implication is Reichenbach's and the negation is the
standard one. The two LTN models are trained by maximizing satisfaction:

- **LTN w/o knowledge**: `loss = 1 − SatAgg(data axioms)`
- **LTN w/ knowledge**: `loss = 1 − (0.8 · SatAgg(data axioms) + 0.2 · SatAgg(knowledge axioms))`, with
  `p = 6` on the survivor axiom

The weights 0.8 and 0.2 are `W_DATA` and `W_KNOWLEDGE` in `training.py`.

All neural models (vanilla LSTM and both LTNs) are early-stopped on validation macro F1. After the first 10
epochs, training stops when F1 has not improved for more than 15 epochs (20 for the LTN w/ knowledge in
`stratified_main.py`). The best checkpoint is then restored.

The tree baselines use aggregated features: the mean of every feature over the window, ignoring missing values.
`stratified_main.py` also adds the standard deviation and the raw series of the first feature.

## Temporal split and learning curves

The cross-validation experiments mix admissions from all years. `temporal_train_size.py` instead tests how the
models generalize to later patients, and how much training data each model needs. It runs every horizon, every
training fraction, every run and every model, and writes the raw results. `paper_results_temporal_train_size.py`
then turns them into figures and LaTeX tables. The full guide, with every option and output file, is
[README_temporal_train_size.md](README_temporal_train_size.md).

### Protocol

- **Temporal split.** Each admission is assigned by its patient's `anchor_year_group` in `data/patients.csv`:
  2008–2013 for training, 2014–2016 for validation and 2017–2019 for testing. The scalers are fitted on the
  training period only. eICU is used as a second, external test set.
- **Training fractions.** The models are trained on 10, 20, 50, 70, 90 and 100% of the training period. The
  subsets are stratified, so the mortality rate is the same in each, and nested: each smaller subset is contained
  in the larger ones.
- **Repeated runs.** Every run uses a different seed and therefore different subsets (5 runs by default). All
  models in a run share the same subset and seed, so their results can be compared in pairs.
- **Model selection.** All models are early-stopped on validation AUPRC, which does not depend on a decision
  threshold:
  - LSTM and LTNs: at most 50 epochs, patience 10.
  - XGBoost: at most 1000 boosting rounds, patience 50.
  - Random Forest: grown 25 trees at a time.

  Metrics (including AUPRC) are reported on the test period and on eICU.
- **First-admission cohort.** With `--first_admission`, only each patient's first hospital admission is kept.
  The admissions are ranked with the full MIMIC-IV `admissions.csv` (`--admissions_path`).

### Knowledge base

The LTN w/ knowledge uses the same eleven risk concepts and the same `--predicates`, `--sigmoid_width` and
`--freeze_sigmoid` options as [`knowledge.py`](#knowledge-base-knowledgepy). The script keeps its own copy of the
knowledge base, so a change to the axioms has to be made in both places. Unlike `main.py` and
`stratified_main.py`, the default output folder gets a `_sigmoid` suffix with `--predicates sigmoid`.

### Running it

```bash
# all admissions -> results_temporal_train_size/
python temporal_train_size.py

# only the first admission of each patient -> results_temporal_train_size_first_admission/
python temporal_train_size.py --first_admission

# quick check (a few minutes)
python temporal_train_size.py --horizons 6 --fractions 0.05 0.1 --n_runs 1 \
    --num_epochs 1 --num_epochs_nesy 1 --num_epochs_nesy_k 1 --max_trees 50 --out_dir smoke_test

# figures and LaTeX tables from both cohorts -> paper_temporal_train_size/
python paper_results_temporal_train_size.py \
    --results_dirs results_temporal_train_size results_temporal_train_size_first_admission
```

`temporal_train_size.py` writes, for each horizon `h`, `split_<h>h.csv`, `results_<h>h.csv` and
`summary_<h>h.csv`. `paper_results_temporal_train_size.py` produces:

- learning curves of AUROC and AUPRC against the training fraction;
- mean ± std tables for each horizon, model and fraction;
- a paired comparison of NeSy-SMP with each baseline (same subset and seed), with 95% confidence intervals;
- the comparison between the two cohorts.

`summarize_results.py` prints mean ± std tables directly from `results_<h>h.csv` files.

## Experiments

| Script | Horizon | What it does | Output |
|---|---|---|---|
| `main.py` | 24 h | Single split (70/30, then 20% of training for validation), external validation, error analysis, calibration | see [Quick start](#quick-start) |
| `stratified_main.py` | 6 h | 5-fold stratified CV of all models | `stratified_results*.txt` |
| `stratified_main_CI.py` | 12 h | 5-fold CV that saves out-of-fold predictions and computes bootstrap confidence intervals of the differences between models | `res_oof/`, `stratified_results_CI.txt` |
| `stratified_main_time.py` | 6 h | 5-fold CV that measures training time, inference time and parameter counts | `stratified_training_times.txt`, `stratified_inference_times.txt`, `stratified_parameter_counts.txt` |
| `ablation_study.py` | 48 h | Ablation of weak anchoring (concept axioms) and of the weighted axiom loss | `ablation_stratified_results.txt` |
| `external_validation.py` | 6 h train, 6–48 h eICU | Bootstrap metrics on eICU at every horizon, calibration curves and Brier scores | `calibration_eicu.pdf` |
| `temporal_train_size.py` | 6–48 h | Temporal split and learning curves, see [above](#temporal-split-and-learning-curves) | `results_temporal_train_size*/` |
| `paper_results_temporal_train_size.py` | — | Figures and LaTeX tables from the learning-curve results | `paper_temporal_train_size/` |
| `summarize_results.py` | — | Mean ± std tables from `results_<h>h.csv` files | terminal, CSV |

`main.py` and `stratified_main.py` use the shared modules `knowledge.py` and `training.py`. The other experiment
scripts still define their own copy of the knowledge base and training loop, so a change to the axioms has to be
made in each of them.

### Clinical-score baselines

| Script | What it does |
|---|---|
| `scores.py` | Computes SOFA and APACHE II from an event log (default: the 24 h MIMIC-IV log) and writes `patient_scores_per_patient.csv` |
| `analyze_sofa_apache_scores.py` | Evaluates the scores as mortality predictors (thresholds SOFA ≥ 10, APACHE II ≥ 20) and writes the deaths both scores missed, the false alarms and `confusion_matrices.png` |
| `extract_patients.py` | Extracts LTN predictions and risk-predicate values per patient from a prediction log, to compare them with the clinical scores |

## Repository layout

Only source files and inputs are listed. Generated results, checkpoints and figures are left out.

```text
.
├── main.py                        # single-split experiment (entry point)
├── stratified_main.py             # 5-fold cross-validation (entry point)
├── knowledge.py                   # LTN operators, data axioms, ClinicalKnowledge
├── training.py                    # training loops, early stopping, evaluation helpers
├── metrics.py                     # metrics, confusion matrix and per-patient predictions of a model
├── utils.py                       # bootstrap confidence intervals and clinical-impact analysis
├── model/
│   └── models.py                  # LSTMModel and the MLPs of the risk predicates
├── data/
│   ├── dataset.py                 # SepsisDataset, ModelConfig
│   ├── preprocessing.py           # event log -> fixed-length samples, scaling, encoding
│   ├── labeling.py                # flags every event followed by death within 48 h
│   ├── extract_before_death.py    # cuts the observation window for a given horizon
│   ├── com_pivot.py               # comorbidity list -> one binary column per comorbidity
│   ├── merge_on_stay_id.py        # attaches hadm_id to ICU urine-output records
│   └── subset/                    # event logs used by the models (see Data)
├── extract_com.py                 # comorbidity extraction from discharge notes (medspacy)
├── create_ckg.py                  # clinical knowledge graph (RDF, SNOMED concepts)
├── extract_rules.py               # rule mining on the knowledge graph (AnyBURL via PyClause)
├── filter_rules.py                # keeps the mined rules about death with confidence ≥ 0.8
├── rules/                         # knowledge graph, mined and filtered rules, plot_kc.py
├── plot_facts.py                  # plots of predicate truth values and axiom satisfaction
├── scores.py, analyze_sofa_apache_scores.py, extract_patients.py
├── stratified_main_CI.py, stratified_main_time.py, ablation_study.py, external_validation.py
├── temporal_train_size.py        # temporal split and learning curves (entry point)
├── paper_results_temporal_train_size.py  # figures and LaTeX tables of the temporal experiment
├── summarize_results.py          # mean ± std tables of the temporal experiment
└── README_temporal_train_size.md # full guide to the temporal experiment
```

The clinical knowledge graph (`create_ckg.py`, `rules/ckg.txt`) and the rules mined from it
(`rules/clinical-rule-filtered.csv`) document the risk factors linked to death. The axioms in `knowledge.py` are
written by hand.
