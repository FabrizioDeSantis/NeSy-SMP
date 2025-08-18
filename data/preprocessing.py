from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from collections import Counter
import random
import torch

def stratified_sampling(data):
    df_non_null_labels = data.dropna(subset=["hospital_expire_flag"])
    group_labels = df_non_null_labels.groupby("hadm_id")["hospital_expire_flag"].first()
    unique_group_ids = group_labels.index
    strat = group_labels.values
    sampled_group_ids, _, _, _ = train_test_split(unique_group_ids, strat, test_size=0.90, stratify=strat, random_state=24)
    sampled_df = data[data["hadm_id"].isin(sampled_group_ids)].copy()
    return sampled_df

def contains_one(lst):
    return 1 if 1 in lst else 0

def fill_nan_values(group, column):
    group[column] = group[column].ffill()
    group[column] = group[column].fillna(0)
    return group

def create_training_data(data, window_size):
    samples_training = []
    labels_training = []
    samples_test = []
    labels_test = []
    # training_data = data[data["hadm_id"].isin(train_ids)]
    # test_data = data[data["hadm_id"].isin(test_ids)]

    # # Create n-grams for training data
    # for id_value, group in data.groupby('hadm_id'):
    #     group = group.drop(columns=["hadm_id"])
    #     group_comorbidities = group[['acute kidney injury', 'aids', 'atrial fibrillation', 'cad', 'cancer',
    #     'cerebrovascular accident', 'cirrhosis', 'copd', 'dementia', 'diabetes',
    #     'diabetes mellitus', 'heart failure', 'hiv', 'hypertension',
    #     'kidney disease', 'kidney failure', 'leukemia', 'lymphoma',
    #     'metastatic cancer', 'metastatic disease', 'peptic ulcer disease',
    #     'pneumonia', 'trauma']]
    #     # take only the first row of comorbidities
    #     group_comorbidities = group_comorbidities.iloc[0]
    #     # convert to list
    #     comorbidities_list = group_comorbidities.tolist()
    #     label = int(group['hospital_expire_flag'].dropna().iloc[-1])
    #     group = group.drop(columns=["hospital_expire_flag"])
    #     labels_training.append(label)
    #     list_of_lists = group.values.tolist()
    #     if len(list_of_lists) > window_size:
    #         list_of_lists = list_of_lists[-window_size:]
    #     cols = [list(col) for col in zip(*list_of_lists)]
    #     cols = [inner_list + [0] * (window_size-len(inner_list)) for inner_list in cols]
    #     samples_training.append((cols, comorbidities_list, id_value))

    # Create n-grams for test data
    for id_value, group in data.groupby('hadm_id'):
        group = group.drop(columns=["hadm_id"])
        group_comorbidities = group[['acute kidney injury', 'aids', 'atrial fibrillation', 'cad', 'cancer',
        'cerebrovascular accident', 'cirrhosis', 'copd', 'dementia', 'diabetes',
        'diabetes mellitus', 'heart failure', 'hiv', 'hypertension',
        'kidney disease', 'kidney failure', 'leukemia', 'lymphoma',
        'metastatic cancer', 'metastatic disease', 'peptic ulcer disease',
        'pneumonia', 'trauma']]
        # take only the first row of comorbidities
        group_comorbidities = group_comorbidities.iloc[0]
        # convert to list
        comorbidities_list = group_comorbidities.tolist()
        label = int(group['hospital_expire_flag'].dropna().iloc[-1])
        group = group.drop(columns=["hospital_expire_flag"])
        labels_test.append(label)
        list_of_lists = group.values.tolist()
        if len(list_of_lists) > window_size:
            list_of_lists = list_of_lists[-window_size:]
        cols = [list(col) for col in zip(*list_of_lists)]
        cols = [inner_list + [0] * (window_size-len(inner_list)) for inner_list in cols]
        samples_test.append((cols, comorbidities_list, id_value))

        feature_names = group.columns.tolist()

    return samples_test, labels_test, feature_names


def preprocess_eventlog(data, seed, sampling=False):

    # data = stratified_sampling(data)
    # seven_days_mortality = pd.read_csv("data/admissions_30_days_mortality.csv", dtype={"hadm_id": str, "subject_id": str})
    if sampling:
        data = stratified_sampling(data)

    data = data[~data["concept:name"].isin(["Urine output"])]

    grouped_labels = data.groupby("hadm_id")["hospital_expire_flag"].first()
    admission_ids = grouped_labels.index.tolist()
    labels = grouped_labels.values.astype(int).tolist()

    print(len(admission_ids))
    print("Number of patients: ", len(labels))

    # train_ids, test_ids, train_labels, test_labels = train_test_split(admission_ids, labels, test_size=0.3, stratify=labels, random_state=seed)

    # print("Number of patients in train set: ", len(train_ids))
    # print("Number of patients in test set: ", len(test_ids))
    # # print labels distribution
    # print("Train labels distribution: ", Counter(train_labels))
    # print("Test labels distribution: ", Counter(test_labels))

    scalers = {}
    vocab_sizes = {}
    
    data = data.drop(columns=['race', 'subject_id','time:timestamp', 'language', 'last_careunit', 'marital_status'])
    data = data[~data["concept:name"].isin(["Death", "Discharge from hospital"])]
    data = data[~data["concept:name"].str.contains("std", case=False, na=False)]
    data["concept:name"] = data["concept:name"].fillna(0)
    data["concept:name"] = pd.Categorical(data["concept:name"])
    print("Administration of vasopressor: ", data["concept:name"].cat.categories.get_loc("Administration of vasopressor") + 1)
    data["concept:name"] = data["concept:name"].cat.codes + 1
    vocab_sizes["concept:name"] = data["concept:name"].max()

    grouped = data.groupby("hadm_id")
    data["hospital_expire_flag"] = grouped["hospital_expire_flag"].transform(lambda x: x.ffill())
    data["hospital_expire_flag"] = grouped["hospital_expire_flag"].transform(lambda x: x.bfill())

    for column_cat in ['admission_type', 'admission_location', 'medication']:
        data[column_cat] = data[column_cat].ffill()
        data[column_cat] = data[column_cat].bfill()
        data[column_cat] = data[column_cat].fillna("unknown")
        data[column_cat] = pd.Categorical(data[column_cat])
        data[column_cat] = data[column_cat].cat.codes + 1
        vocab_sizes[column_cat] = data[column_cat].max()
    
    for column_num in ["Heart Rate", "Respiratory Rate", "Temperature Celsius", "Hemoglobin", "Platelet Count",
                       "Creatinine (serum)", "Total Bilirubin", "Potassium (serum)", "Albumin", "Arterial CO2 Pressure",
                       "Arterial Blood Pressure systolic", "Arterial Blood Pressure diastolic", "Arterial Blood Pressure mean",
                       "Daily Weight", "Brain Natiuretic Peptide (BNP)", "Direct Bilirubin", "C-Reactive Protein", "Creatinine (whole blood)", 
                       "Glucose", "Lactate", "Lymphocytes", "Neutrophils", "White Blood Cells", "Alanine Aminotransferase (ALT)", 
                       "Asparate Aminotransferase (AST)", "gcs", "anchor_age"]:
            data[column_num] = grouped[column_num].ffill()
            if column_num == "anchor_age":
                data[column_num] = grouped[column_num].bfill()
            data[column_num] = data[column_num].fillna(0)
            scaler = MinMaxScaler()
            data[column_num] = scaler.fit_transform(data[[column_num]])
            scalers[column_num] = scaler

    comorbidities = ['acute kidney injury', 'aids', 'atrial fibrillation', 'cad', 'cancer',
        'cerebrovascular accident', 'cirrhosis', 'copd', 'dementia', 'diabetes',
        'diabetes mellitus', 'heart failure', 'hiv', 'hypertension',
        'kidney disease', 'kidney failure', 'leukemia', 'lymphoma',
        'metastatic cancer', 'metastatic disease', 'peptic ulcer disease',
        'pneumonia', 'trauma']
    
    for comorbidity in comorbidities:
        data[comorbidity] = grouped[comorbidity].transform(lambda x: x.ffill())
        data[comorbidity] = grouped[comorbidity].transform(lambda x: x.bfill())
        data[comorbidity] = data[comorbidity].fillna(0)
        data[comorbidity] = data[comorbidity].astype(int)

    #data = data[['hadm_id', 'concept:name', 'admission_type', 'admission_location', 'insurance', 'Lactate', 'Albumin', 'Creatinine (serum)', 'C-Reactive Protein', 'Heart Rate mean', 'Heart Rate std', 'Respiratory Rate mean', 'Respiratory Rate std', 'Temperature Celsius mean', 'Temperature Celsius std', 'Hemoglobin', 'Platelet Count', 'Total Bilirubin', 'Potassium (serum)', 'Arterial CO2 Pressure', 'Arterial Blood Pressure systolic mean', 'Arterial Blood Pressure systolic std', 'Arterial Blood Pressure diastolic mean', 'Arterial Blood Pressure diastolic std', 'Daily Weight', 'Direct Bilirubin', 'Creatinine (whole blood)', 'medication', 'Glucose', 'Lymphocytes', 'Neutrophils', 'White Blood Cells', 'Alanine Aminotransferase (ALT)', 'Asparate Aminotransferase (AST)', 'hospital_expire_flag']]

    data = data[['hadm_id', 'concept:name', 'admission_type', 'admission_location', 'medication', "Heart Rate", "Respiratory Rate", "Temperature Celsius", "Hemoglobin", "Platelet Count",
                    "Creatinine (serum)", "Total Bilirubin", "Potassium (serum)", "Albumin", "Arterial CO2 Pressure",
                    "Arterial Blood Pressure systolic", "Arterial Blood Pressure diastolic", "Arterial Blood Pressure mean",
                    "Daily Weight", "Brain Natiuretic Peptide (BNP)", "Direct Bilirubin", "C-Reactive Protein", "Creatinine (whole blood)", 
                    "Glucose", "Lactate", "Lymphocytes", "Neutrophils", "White Blood Cells", "Alanine Aminotransferase (ALT)", 
                    "Asparate Aminotransferase (AST)", "gcs",
                    'anchor_age', 'acute kidney injury', 'aids', 'atrial fibrillation', 'cad', 'cancer',
                    'cerebrovascular accident', 'cirrhosis', 'copd', 'dementia', 'diabetes',
                    'diabetes mellitus', 'heart failure', 'hiv', 'hypertension',
                    'kidney disease', 'kidney failure', 'leukemia', 'lymphoma',
                    'metastatic cancer', 'metastatic disease', 'peptic ulcer disease',
                    'pneumonia', 'trauma', 'hospital_expire_flag']]
    # Group by 'id' and count the size of each group
    group_sizes = data.groupby('hadm_id').size()

    # Find the maximum group size
    max_group_size = group_sizes.max()
    min_group_size = group_sizes.min()

    print("Maximum group size:", max_group_size)
    print("Minimum group size:", min_group_size)

    data = data.groupby('hadm_id').filter(lambda x: len(x) > 4)

    return create_training_data(data, max_group_size), vocab_sizes, scalers, max_group_size

def preprocess_eventlog_mean(data, seed, sampling=False):

    # data = stratified_sampling(data)
    # seven_days_mortality = pd.read_csv("data/admissions_30_days_mortality.csv", dtype={"hadm_id": str, "subject_id": str})
    if sampling:
        data = stratified_sampling(data)

    grouped_labels = data.groupby("hadm_id")["hospital_expire_flag"].first()
    admission_ids = grouped_labels.index.tolist()
    labels = grouped_labels.values.astype(int).tolist()

    print(len(admission_ids))
    print("Number of patients: ", len(labels))

    train_ids, test_ids, train_labels, test_labels = train_test_split(admission_ids, labels, test_size=0.3, stratify=labels, random_state=seed)

    print("Number of patients in train set: ", len(train_ids))
    print("Number of patients in test set: ", len(test_ids))
    # print labels distribution
    print("Train labels distribution: ", Counter(train_labels))
    print("Test labels distribution: ", Counter(test_labels))

    scalers = {}
    vocab_sizes = {}
    
    data = data.drop(columns=['race', 'subject_id','time:timestamp', 'language', 'last_careunit', 'marital_status'])
    data = data[~data["concept:name"].isin(["Death", "Discharge from hospital"])]
    data = data[~data["concept:name"].str.contains("std", case=False, na=False)]
    data["concept:name"] = data["concept:name"].fillna(0)
    data["concept:name"] = pd.Categorical(data["concept:name"])
    print("Administration of vasopressor: ", data["concept:name"].cat.categories.get_loc("Administration of vasopressor") + 1)
    data["concept:name"] = data["concept:name"].cat.codes + 1
    vocab_sizes["concept:name"] = data["concept:name"].max()

    grouped = data.groupby("hadm_id")
    data["hospital_expire_flag"] = grouped["hospital_expire_flag"].transform(lambda x: x.ffill())
    data["hospital_expire_flag"] = grouped["hospital_expire_flag"].transform(lambda x: x.bfill())

    for column_cat in ['admission_location', 'insurance', 'medication']:
        data[column_cat] = data[column_cat].ffill()
        data[column_cat] = data[column_cat].bfill()
        data[column_cat] = data[column_cat].fillna("unknown")
        data[column_cat] = pd.Categorical(data[column_cat])
        data[column_cat] = data[column_cat].cat.codes + 1
        vocab_sizes[column_cat] = data[column_cat].max()
    
    for column_num in ['Hemoglobin mean', 'Hemoglobin std',
       'Platelet Count mean', 'Platelet Count std', 'Creatinine (serum) mean',
       'Creatinine (serum) std', 'Total Bilirubin mean', 'Total Bilirubin std',
       'Potassium (serum) mean', 'Potassium (serum) std', 'Albumin mean',
       'Albumin std', 'Arterial CO2 Pressure mean',
       'Arterial CO2 Pressure std', 'Glucose mean', 'Glucose std',
       'Lactate mean', 'Lactate std', 'Lymphocytes mean', 'Lymphocytes std',
       'Neutrophils mean', 'Neutrophils std', 'White Blood Cells mean',
       'White Blood Cells std', 'Heart Rate mean', 'Heart Rate std',
       'Arterial Blood Pressure systolic mean',
       'Arterial Blood Pressure systolic std',
       'Arterial Blood Pressure diastolic mean',
       'Arterial Blood Pressure diastolic std',
       'Arterial Blood Pressure mean mean', 'Arterial Blood Pressure mean std',
       'Respiratory Rate mean', 'Respiratory Rate std', 'Daily Weight mean',
       'Daily Weight std', 'Alanine Aminotransferase (ALT) mean',
       'Alanine Aminotransferase (ALT) std',
       'Asparate Aminotransferase (AST) mean',
       'Asparate Aminotransferase (AST) std', 'Temperature Celsius mean',
       'Temperature Celsius std', 'Direct Bilirubin mean',
       'Direct Bilirubin std', 'Creatinine (whole blood) mean',
       'Creatinine (whole blood) std', 'C-Reactive Protein mean',
       'C-Reactive Protein std', 'anchor_age']:
            data[column_num] = grouped[column_num].ffill()
            data[column_num] = data[column_num].fillna(0)
            scaler = MinMaxScaler()
            data[column_num] = scaler.fit_transform(data[[column_num]])
            scalers[column_num] = scaler

    comorbidities = ['acute kidney injury', 'aids', 'atrial fibrillation', 'cad', 'cancer',
        'cerebrovascular accident', 'cirrhosis', 'copd', 'dementia', 'diabetes',
        'diabetes mellitus', 'heart failure', 'hiv', 'hypertension',
        'kidney disease', 'kidney failure', 'leukemia', 'lymphoma',
        'metastatic cancer', 'metastatic disease', 'peptic ulcer disease',
        'pneumonia', 'trauma']
    
    for comorbidity in comorbidities:
        data[comorbidity] = grouped[comorbidity].transform(lambda x: x.ffill())
        data[comorbidity] = grouped[comorbidity].transform(lambda x: x.bfill())
        data[comorbidity] = data[comorbidity].fillna(0)
        data[comorbidity] = data[comorbidity].astype(int)

    #data = data[['hadm_id', 'concept:name', 'admission_type', 'admission_location', 'insurance', 'Lactate', 'Albumin', 'Creatinine (serum)', 'C-Reactive Protein', 'Heart Rate mean', 'Heart Rate std', 'Respiratory Rate mean', 'Respiratory Rate std', 'Temperature Celsius mean', 'Temperature Celsius std', 'Hemoglobin', 'Platelet Count', 'Total Bilirubin', 'Potassium (serum)', 'Arterial CO2 Pressure', 'Arterial Blood Pressure systolic mean', 'Arterial Blood Pressure systolic std', 'Arterial Blood Pressure diastolic mean', 'Arterial Blood Pressure diastolic std', 'Daily Weight', 'Direct Bilirubin', 'Creatinine (whole blood)', 'medication', 'Glucose', 'Lymphocytes', 'Neutrophils', 'White Blood Cells', 'Alanine Aminotransferase (ALT)', 'Asparate Aminotransferase (AST)', 'hospital_expire_flag']]

    # data = data[['hadm_id', 'concept:name', 'admission_type', 'admission_location', 'insurance', 'medication', 'Hemoglobin mean', 'Hemoglobin std',
    #    'Platelet Count mean', 'Platelet Count std', 'Creatinine (serum) mean',
    #    'Creatinine (serum) std', 'Total Bilirubin mean', 'Total Bilirubin std',
    #    'Potassium (serum) mean', 'Potassium (serum) std', 'Albumin mean',
    #    'Albumin std', 'Arterial CO2 Pressure mean',
    #    'Arterial CO2 Pressure std', 'Glucose mean', 'Glucose std',
    #    'Lactate mean', 'Lactate std', 'Lymphocytes mean', 'Lymphocytes std',
    #    'Neutrophils mean', 'Neutrophils std', 'White Blood Cells mean',
    #    'White Blood Cells std', 'Heart Rate mean', 'Heart Rate std',
    #    'Arterial Blood Pressure systolic mean',
    #    'Arterial Blood Pressure systolic std',
    #    'Arterial Blood Pressure diastolic mean',
    #    'Arterial Blood Pressure diastolic std',
    #    'Arterial Blood Pressure mean mean', 'Arterial Blood Pressure mean std',
    #    'Respiratory Rate mean', 'Respiratory Rate std', 'Daily Weight mean',
    #    'Daily Weight std', 'Alanine Aminotransferase (ALT) mean',
    #    'Alanine Aminotransferase (ALT) std',
    #    'Asparate Aminotransferase (AST) mean',
    #    'Asparate Aminotransferase (AST) std', 'Temperature Celsius mean',
    #    'Temperature Celsius std', 'Direct Bilirubin mean',
    #    'Direct Bilirubin std', 'Creatinine (whole blood) mean',
    #    'Creatinine (whole blood) std', 'C-Reactive Protein mean',
    #    'C-Reactive Protein std', 'anchor_age', 'acute kidney injury', 'aids', 'atrial fibrillation', 'cad', 'cancer',
    #     'cerebrovascular accident', 'cirrhosis', 'copd', 'dementia', 'diabetes',
    #     'diabetes mellitus', 'heart failure', 'hiv', 'hypertension',
    #     'kidney disease', 'kidney failure', 'leukemia', 'lymphoma',
    #     'metastatic cancer', 'metastatic disease', 'peptic ulcer disease',
    #     'pneumonia', 'trauma', 'hospital_expire_flag']]

    data = data[['hadm_id', 'concept:name', 'insurance', 'medication', 'Hemoglobin mean',
       'Platelet Count mean', 'Creatinine (serum) mean', 'Total Bilirubin mean',
       'Potassium (serum) mean', 'Albumin mean', 'Arterial CO2 Pressure mean', 'Glucose mean',
       'Lactate mean', 'Lymphocytes mean',
       'Neutrophils mean', 'White Blood Cells mean', 'Heart Rate mean',
       'Arterial Blood Pressure systolic mean',
       'Arterial Blood Pressure diastolic mean',
       'Arterial Blood Pressure mean mean',
       'Respiratory Rate mean', 'Daily Weight mean', 'Alanine Aminotransferase (ALT) mean',
       'Asparate Aminotransferase (AST) mean', 'Temperature Celsius mean', 'Direct Bilirubin mean', 'Creatinine (whole blood) mean', 'C-Reactive Protein mean',
       'anchor_age', 'acute kidney injury', 'aids', 'atrial fibrillation', 'cad', 'cancer',
        'cerebrovascular accident', 'cirrhosis', 'copd', 'dementia', 'diabetes',
        'diabetes mellitus', 'heart failure', 'hiv', 'hypertension',
        'kidney disease', 'kidney failure', 'leukemia', 'lymphoma',
        'metastatic cancer', 'metastatic disease', 'peptic ulcer disease',
        'pneumonia', 'trauma', 'hospital_expire_flag']]
    # Group by 'id' and count the size of each group
    group_sizes = data.groupby('hadm_id').size()

    # Find the maximum group size
    max_group_size = group_sizes.max()
    min_group_size = group_sizes.min()

    print("Maximum group size:", max_group_size)
    print("Minimum group size:", min_group_size)

    # Filter out groups with size less than or equal to 4
    data = data.groupby('hadm_id').filter(lambda x: len(x) > 4)

    # After filtering data, we need to update train_ids and test_ids as well
    valid_ids = set(data['hadm_id'].unique())
    train_ids = [id for id in train_ids if id in valid_ids]
    test_ids = [id for id in test_ids if id in valid_ids]

    return create_training_data(data, train_ids, test_ids, max_group_size), vocab_sizes, scalers, max_group_size