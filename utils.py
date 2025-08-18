import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from ltn.fuzzy_ops import AggregPMeanError
import ltn

def pad_lists(lists, num_features, max_length):
    return [tuple(lst[i] + [0] * (max_length - len(lst[i])) for i in range(num_features)) for lst in lists]

def print_dataset_info(training_list, validation_list, test_list):
    print("-- DATASET INFO")
    print(f"Training set size: {len(training_list)}")
    print(f"Validation set size: {len(validation_list)}")
    print(f"Test set size: {len(test_list)}")
    print("--- Label distribution")
    print("--- Training set")
    counts = Counter(training_list[1])
    print(counts)
    print("--- Validation set")
    counts = Counter(validation_list[1])
    print(counts)
    print("--- Test set")
    counts = Counter(test_list[1])
    print(counts)

def compute_truth_values(predicate, param, func_com, func_age, clinical_concept, loader):
    truth_values = []
    for x, y, c_id in loader:
        x_All = ltn.Variable("x_All", x)
        truth_values.append(predicate(param(x_All), func_com(x_All), func_age(x_All)).value.cpu().detach().numpy())
    # flatten the list of lists
    truth_values = [item for sublist in truth_values for item in sublist]
    print(truth_values)

def plot_compliance(compliance_dict):
    # Sort the lengths for proper ordering in the plot
    lengths = sorted(compliance_dict.keys())
    compliance_scores = [compliance_dict[length] for length in lengths]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lengths, compliance_scores, marker='o', linestyle='-', color='b')
    
    # Add labels and title
    ax.set_xlabel('Case Length', fontsize=12)
    ax.set_ylabel('Compliance Score', fontsize=12)
    ax.set_title('Compliance by Case Length', fontsize=14)
    
    # Set y-axis limits and grid
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.show()

class WeightedSatAgg:
    def __init__(self, agg_op=AggregPMeanError(p=2)):
        self.agg_op = agg_op
    def __call__(self, *closed_formulas):
        truth_values = list(closed_formulas)