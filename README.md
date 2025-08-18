# NeSy-SMP

This repository contains the code for NeSy-SMP, a project exploring the integration of Neuro-Symbolic AI for real-time sepsis mortality prediction.

## Files

*   **`main.py`**: Contains the code for evaluating the BiLSTM, LTN and NeSy-SMP models on a standard train-test split.
*   **`stratified_main.py`**: Contains the code for evaluating the Random Forest, XGBoost, BiLSTM, LTN and NeSy-SMP models on a 5-fold cross validation.
*   **`create_ckg.py`**: Contains the code for creating the clinical knowledge graph in RDF format.
*   **`extract_comorbidities.py`**: Contains the code for extracting the comorbidities from clinical textual notes.
*   **`extract_rules.py`**: Contains the code for extracting logical rules from a given knowledge graph in RDF format.
*   **`filter_rules.py`**: Contains the code for filtering the extracted rules given a threshold and the main outcome.
*   **`metrics.py`**: Contains the code for computing the metrics for the different configurations.
*   **`plot_facts.py`**: Script for plotting the truth values of predicates and implications.
*   **`data/dataset.py`**: Dataset class.
*   **`data/extract_before_death.py`**: Contains the code for creating the 24-hour observation windows given the desired lead time (6, 12, 24, and 48 hours).
*   **`model/preprocessing.py`**: Contains the code for preprocessing the data.