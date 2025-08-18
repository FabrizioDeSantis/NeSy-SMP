import medspacy
from medspacy.ner import TargetRule
from medspacy.visualization import visualize_ent, visualize_dep
from pathlib import Path
import pandas as pd

# Load medspacy model
nlp = medspacy.load(medspacy_enable=['medspacy_pyrush', 'medspacy_target_matcher', 'medspacy_context', 'medspacy_sectionizer'])
print(nlp.pipe_names)

# Add rules for target concept extraction
target_matcher = nlp.get_pipe("medspacy_target_matcher")
target_rules = [
    TargetRule("cancer", "PROBLEM"),
    TargetRule("pneumonia", "PROBLEM"),
    TargetRule("cirrhosis", "PROBLEM"),
    TargetRule("dementia", "PROBLEM"),
    TargetRule("kidney disease", "PROBLEM"),
    TargetRule("kidney failure", "PROBLEM"),
    TargetRule("leukemia", "PROBLEM"),
    TargetRule("hypertension", "PROBLEM"),
    TargetRule("HIV", "PROBLEM"),
    TargetRule("COPD", "PROBLEM"),
    TargetRule("Chronic Obstructive Pulmonary Disease", "PROBLEM"),
    TargetRule("diabetes", "PROBLEM"),
    TargetRule("diabetes mellitus", "PROBLEM"),
    TargetRule("trauma", "PROBLEM"),
    TargetRule("coronary artery disease", "PROBLEM"),
    TargetRule("Coronary Artery Disease", "PROBLEM"),
    TargetRule("cad", "PROBLEM"),
    TargetRule("heart failure", "PROBLEM"),
    TargetRule("atrial fibrillation", "PROBLEM"),
    TargetRule("acute kidney injury", "PROBLEM"),
    TargetRule("peptic ulcer disease", "PROBLEM"),
    TargetRule("cerebrovascular accident", "PROBLEM"),
    TargetRule("metastatic disease", "PROBLEM"),
    TargetRule("metastatic cancer", "PROBLEM"),
    TargetRule("lymphoma", "PROBLEM"),
    TargetRule("AIDS", "PROBLEM"),
    ##############################
    # TargetRule("past medical history", "PAST_MEDICAL_HISTORY"),
]
target_matcher.add(target_rules)

notes = pd.read_csv("path_to_note_file.csv")
patients = pd.read_csv("path_to_patients_table.csv")
patient_ids = patients["subject_id"].tolist()
ages = patients["anchor_age"].tolist()
admissions_ids = pd.read_csv("path_to_admissions_table.csv")
subject_ids = admissions_ids["subject_id"].tolist()
hadm_ids = admissions_ids["hadm_id"].tolist()
notes_hadm_ids = notes["hadm_id"].tolist()
texts = notes["text"].tolist()

with open("data/comorbidities.csv", "w", encoding="utf-8") as f:
    f.write("subject_id,hadm_id,age,comorbidity\n")
    for hadm_id, text in zip(notes_hadm_ids, texts):

        doc = nlp(text)
        index = hadm_ids.index(hadm_id)
        subject_id = subject_ids[index]
        patients_id = patient_ids.index(subject_id)
        age = ages[patients_id]

        entities = []
        for target, modifier in doc._.context_graph.edges:
            if modifier.rule.category != "NEGATED_EXISTENCE" and modifier.rule.category != "FAMILY":
                entities.append(target.text)
        entities = list(set(entities))
        for entity in entities:
            entity = entity.lower()
            if entity == "chronic obstructive pulmonary disease":
                entity = "copd"
            elif entity == "coronary artery disease":
                entity = "cad"
            f.write(f"{subject_id},{hadm_id},{age},{entity}\n")
                
# res = visualize_ent(doc, jupyter=False)
# output_path = Path("entity_visualization.html")
# # Save the HTML to the file
# output_path.open("w", encoding="utf-8").write(res)