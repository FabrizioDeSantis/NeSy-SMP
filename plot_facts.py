import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
import ltn

def plt_heatmap(df, vmin=None, vmax=None):
    plt.pcolor(df, vmin=vmin, vmax=vmax, cmap="RdYlBu_r")
    plt.yticks(np.arange(0.5,len(df.index), 1), df.index)
    plt.xticks(np.arange(0.5,len(df.columns), 1), df.columns)
    plt.colorbar()

def plot_facts(lactate_predicate, death_predicate, loader, features_dict, sequence_length, values_before_training):
    lactate = ltn.Function(func = lambda x: x[:,(features_dict["Lactate"]*sequence_length-sequence_length):(features_dict["Lactate"]*sequence_length)])
    creatinine = ltn.Function(func = lambda x: x[:,(features_dict["Creatinine (serum)"]*sequence_length-sequence_length):(features_dict["Creatinine (serum)"]*sequence_length)])
    bilirubin = ltn.Function(func = lambda x: x[:,(features_dict["Total Bilirubin"]*sequence_length-sequence_length):(features_dict["Total Bilirubin"]*sequence_length)])
    gcs = ltn.Function(func = lambda x: x[:,(features_dict["gcs"]*sequence_length-sequence_length):(features_dict["gcs"]*sequence_length)])
    age = ltn.Function(func = lambda x: x[:,(features_dict["anchor_age"]*sequence_length-sequence_length):(features_dict["anchor_age"]*sequence_length)][:, 0])
    abps = ltn.Function(func = lambda x: x[:,(features_dict["Arterial Blood Pressure systolic"]*sequence_length-sequence_length):(features_dict["Arterial Blood Pressure systolic"]*sequence_length)])
    respiratory_rate = ltn.Function(func = lambda x: x[:,(features_dict["Respiratory Rate"]*sequence_length-sequence_length):(features_dict["Respiratory Rate"]*sequence_length)])
    glucose = ltn.Function(func = lambda x: x[:,(features_dict["Glucose"]*sequence_length-sequence_length):(features_dict["Glucose"]*sequence_length)])
    platelet = ltn.Function(func = lambda x: x[:,(features_dict["Platelet Count"]*sequence_length-sequence_length):(features_dict["Platelet Count"]*sequence_length)])

    comorbidities = ltn.Function(func = lambda x: x[:, -23:])
    lactate_values_d = []
    lactate_values_not_d = []
    dead_values = []
    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    survivor_values = []
    for x, y, c_id in loader:
        x_D = ltn.Variable("x_D", x[y==1])
        x_not_D = ltn.Variable("x_not_D", x[y==0])
        #lactate_values_d.append(And(lactate_predicate(abps(x_D), comorbidities(x_D), age(x_D)), respiratory_rate_predicate(respiratory_rate(x_D), comorbidities(x_D), age(x_D))).value.cpu().detach().numpy())
        #lactate_values_not_d.append(And(lactate_predicate(abps(x_not_D), comorbidities(x_not_D), age(x_not_D)), respiratory_rate_predicate(respiratory_rate(x_not_D), comorbidities(x_not_D), age(x_not_D))).value.cpu().detach().numpy())
        lactate_values_d.append(lactate_predicate(bilirubin(x_D), comorbidities(x_D), age(x_D)).value.cpu().detach().numpy())
        lactate_values_not_d.append(lactate_predicate(bilirubin(x_not_D), comorbidities(x_not_D), age(x_not_D)).value.cpu().detach().numpy())
        # lactate_values_d.append(lactate_predicate(lactate(x_D)).value.cpu().detach().numpy())
        # lactate_values_not_d.append(lactate_predicate(lactate(x_not_D)).value.cpu().detach().numpy())
        dead_values.append(death_predicate(x_D).value.cpu().detach().numpy())
        survivor_values.append(death_predicate(x_not_D).value.cpu().detach().numpy())
    #average the values
    lactate_values_d = np.mean(np.concatenate(lactate_values_d, axis=0), axis=0)
    lactate_values_not_d = np.mean(np.concatenate(lactate_values_not_d, axis=0), axis=0)
    dead_values = np.mean(np.concatenate(dead_values, axis=0), axis=0)
    survivor_values = np.mean(np.concatenate(survivor_values, axis=0), axis=0)
    survivors = []
    survivors.append(lactate_values_not_d)
    # survivors.append(survivor_values)
    survivors.append(0.0)
    deaths = []
    deaths.append(lactate_values_d)
    #deaths.append(dead_values)
    deaths.append(1.0)
    
    print("BilirubinRisk (D):", lactate_values_d)
    print("BilirubinRisk (not D):", lactate_values_not_d)
    print("Death Risk (D):", dead_values)
    print("Death Risk (not D):", survivor_values)
    df_lactate_death_facts_before_training = pd.DataFrame(
        # [values_before_training[0], values_before_training[1]],
        [[values_before_training[0][0], 1.0], [values_before_training[1][0], 0.0]],
        columns=["BilirubinRisk(x)", "Death(x)"],
        index=["Deaths", "Survivors"]
    )
    df_lactate_death_facts = pd.DataFrame(
        [deaths, survivors],
        columns=["BilirubinRisk(x)", "Death(x)"],
        index=["Deaths", "Survivors"]
    )
    print(df_lactate_death_facts)
    plt.figure(figsize=(12, 3))
    plt.subplot(121)
    plt.title("BilirubinRisk(x) => Death(x) before training")
    plt_heatmap(df_lactate_death_facts_before_training, vmin=0, vmax=1)
    plt.subplot(122)
    plt.title("BilirubinRisk(x) => Death(x) after training")
    plt_heatmap(df_lactate_death_facts, vmin=0, vmax=1)
    plt.savefig('df_bilirubin_death_facts_gt.pdf')
    plt.show()

def plot_concept_level_verification(lactate_values_before_training, lactate_values_after_training, gt_lactate, bilirubin_values_before_training, bilirubin_values_after_training, gt_bilirubin, platelet_values_before_training, platelet_values_after_training, gt_platelet, qSOFA_values_before_training, qSOFA_values_after_training, gt_qSOFA):
    # lactate
    len_values = len(lactate_values_before_training)
    y_values = list(range(len_values))
    list_values_before_training = []
    for value, gt_value in zip(lactate_values_before_training, gt_lactate):
        list_values_before_training.append([value, gt_value])
    df_lactate_values_before_training = pd.DataFrame(
        list_values_before_training,
        columns=["HighLactate(x)", "Clinical Concept(x)"],
        index=y_values
    )
    len_values = len(lactate_values_after_training)
    y_values = list(range(len_values))
    list_values_after_training = []
    for value, gt_value in zip(lactate_values_after_training, gt_lactate):
        list_values_after_training.append([value, gt_value])
    df_lactate_values_after_training = pd.DataFrame(
        list_values_after_training,
        columns=["HighLactate(x)", "Clinical Concept(x)"],
        index=y_values
    )
    # bilirubin
    len_values = len(bilirubin_values_before_training)
    y_values = list(range(len_values))
    list_values_before_training = []
    for value, gt_value in zip(bilirubin_values_before_training, gt_bilirubin):
        list_values_before_training.append([value, gt_value])
    df_bilirubin_values_before_training = pd.DataFrame(
        list_values_before_training,
        columns=["HighBilirubin(x)", "Clinical Concept(x)"],
        index=y_values
    )
    len_values = len(bilirubin_values_after_training)
    y_values = list(range(len_values))
    list_values_after_training = []
    for value, gt_value in zip(bilirubin_values_after_training, gt_bilirubin):
        list_values_after_training.append([value, gt_value])
    df_bilirubin_values_after_training = pd.DataFrame(
        list_values_after_training,
        columns=["HighBilirubin(x)", "Clinical Concept(x)"],
        index=y_values
    )
    # platelet
    len_values = len(platelet_values_before_training)
    y_values = list(range(len_values))
    list_values_before_training = []
    for value, gt_value in zip(platelet_values_before_training, gt_platelet):
        list_values_before_training.append([value, gt_value])
    df_platelet_values_before_training = pd.DataFrame(
        list_values_before_training,
        columns=["LowPlatelets(x)", "Clinical Concept(x)"],
        index=y_values
    )
    len_values = len(platelet_values_after_training)
    y_values = list(range(len_values))
    list_values_after_training = []
    for value, gt_value in zip(platelet_values_after_training, gt_platelet):
        list_values_after_training.append([value, gt_value])
    df_platelet_values_after_training = pd.DataFrame(
        list_values_after_training,
        columns=["LowPlatelets(x)", "Clinical Concept(x)"],
        index=y_values
    )
    # qSOFA
    len_values = len(qSOFA_values_before_training)
    y_values = list(range(len_values))
    list_values_before_training = []
    for value, gt_value in zip(qSOFA_values_before_training, gt_qSOFA):
        list_values_before_training.append([value, gt_value])
    df_qSOFA_values_before_training = pd.DataFrame(
        list_values_before_training,
        columns=["HighqSOFA(x)", "Clinical Concept(x)"],
        index=y_values
    )
    len_values = len(qSOFA_values_after_training)
    y_values = list(range(len_values))
    list_values_after_training = []
    for value, gt_value in zip(qSOFA_values_after_training, gt_qSOFA):
        list_values_after_training.append([value, gt_value])
    df_qSOFA_values_after_training = pd.DataFrame(
        list_values_after_training,
        columns=["HighqSOFA(x)", "Clinical Concept(x)"],
        index=y_values
    )
    plt.figure(figsize=(16, 8))
    plt.subplot(241)  # Row 1, Col 1
    plt.title("HighLactate before training")
    plt_heatmap(df_lactate_values_before_training, vmin=0, vmax=1)
    plt.subplot(242)  # Row 1, Col 2
    plt.title("HighLactate after training")
    plt_heatmap(df_lactate_values_after_training, vmin=0, vmax=1)
    plt.subplot(243)  # Row 1, Col 3
    plt.title("HighBilirubin before training")
    plt_heatmap(df_bilirubin_values_before_training, vmin=0, vmax=1)
    plt.subplot(244)  # Row 1, Col 4
    plt.title("HighBilirubin after training")
    plt_heatmap(df_bilirubin_values_after_training, vmin=0, vmax=1)
    plt.subplot(245)  # Row 2, Col 1
    plt.title("LowPlatelets before training")
    plt_heatmap(df_platelet_values_before_training, vmin=0, vmax=1)
    plt.subplot(246)  # Row 2, Col 2
    plt.title("LowPlatelets after training")
    plt_heatmap(df_platelet_values_after_training, vmin=0, vmax=1)
    plt.subplot(247)  # Row 2, Col 3
    plt.title("HighqSOFA before training")
    plt_heatmap(df_qSOFA_values_before_training, vmin=0, vmax=1)
    plt.subplot(248)  # Row 2, Col 4
    plt.title("HighqSOFA after training")
    plt_heatmap(df_qSOFA_values_after_training, vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig('concepts/df_clinical_concepts_risk_gt.pdf')
    plt.show()

def plot_knowledge_axioms(
    lowmap_implies_death_values_d,
    lowmap_implies_death_values_s,
    lactate_implies_death_values_d,
    lactate_implies_death_values_s,
    platelet_implies_death_values_d,
    platelet_implies_death_values_s,
    bilirubin_implies_death_values_d,
    bilirubin_implies_death_values_s,
    lactate_not_clearing_values_d,
    lactate_not_clearing_values_s,
    wbc_implies_death_values_d,
    wbc_implies_death_values_s,
    crp_implies_death_values_d,
    crp_implies_death_values_s,
    cronic_conditions_implies_death_values_d,
    cronic_conditions_implies_death_values_s
):
    gt = True
    if gt:
        outcome_predicate = "Label(x)"
    else:
        outcome_predicate = "HighMortality(x)"
    df_lactate_death_facts = pd.DataFrame(
        [[lactate_implies_death_values_d, 1.0], [lactate_implies_death_values_s, 0.0]],
        columns=["HighLactate(x)", outcome_predicate],
        index=["Deaths", "Survivors"]
    )
    df_lactate_not_clearing_facts = pd.DataFrame(
        [[lactate_not_clearing_values_d, 1.0], [lactate_not_clearing_values_s, 0.0]],
        columns=["LactateNotClearing(x)", outcome_predicate],
        index=["Deaths", "Survivors"]
    )
    df_bilirubin_death_facts = pd.DataFrame(
        [[bilirubin_implies_death_values_d, 1.0], [bilirubin_implies_death_values_s, 0.0]],
        columns=["HighBilirubin(x)", outcome_predicate],
        index=["Deaths", "Survivors"]
    )
    df_platelet_death_facts = pd.DataFrame(
        [[platelet_implies_death_values_d, 1.0], [platelet_implies_death_values_s, 0.0]],
        columns=["LowPlatelets(x)", outcome_predicate],
        index=["Deaths", "Survivors"]
    )
    df_lowmap_death_facts = pd.DataFrame(
        [[lowmap_implies_death_values_d, 1.0], [lowmap_implies_death_values_s, 0.0]],
        columns=["LowMAP(x)", outcome_predicate],
        index=["Deaths", "Survivors"]
    )
    df_wbc_death_facts = pd.DataFrame(
        [[wbc_implies_death_values_d, 1.0], [wbc_implies_death_values_s, 0.0]],
        columns=["HighWBC(x)", outcome_predicate],
        index=["Deaths", "Survivors"]
    )
    df_crp_death_facts = pd.DataFrame(
        [[crp_implies_death_values_d, 1.0], [crp_implies_death_values_s, 0.0]],
        columns=["HighCRP(x)", outcome_predicate],
        index=["Deaths", "Survivors"]
    )
    df_cronic_conditions_death_facts = pd.DataFrame(
        [[cronic_conditions_implies_death_values_d, 1.0], [cronic_conditions_implies_death_values_s, 0.0]],
        columns=["ChronicConditionsRisk(x)", outcome_predicate],
        index=["Deaths", "Survivors"]
    )
    plt.figure(figsize=(24, 10))
    plt.subplot(241)
    plt.title("HighLactate(x) => " + outcome_predicate)
    plt_heatmap(df_lactate_death_facts, vmin=0, vmax=1)
    plt.subplot(242)
    plt.title("LactateNotClearing(x) => " + outcome_predicate)
    plt_heatmap(df_lactate_not_clearing_facts, vmin=0, vmax=1)
    plt.subplot(243)
    plt.title("HighBilirubin(x) => " + outcome_predicate)
    plt_heatmap(df_bilirubin_death_facts, vmin=0, vmax=1)
    plt.subplot(244)
    plt.title("LowPlatelets(x) => " + outcome_predicate)
    plt_heatmap(df_platelet_death_facts, vmin=0, vmax=1)
    plt.subplot(245)
    plt.title("LowMAP(x) => " + outcome_predicate)
    plt_heatmap(df_lowmap_death_facts, vmin=0, vmax=1)
    plt.tight_layout()
    plt.subplot(246)
    plt.title("HighWBC(x) => " + outcome_predicate)
    plt_heatmap(df_wbc_death_facts, vmin=0, vmax=1)
    plt.tight_layout()
    plt.subplot(247)
    plt.title("HighCRP(x) => " + outcome_predicate)
    plt_heatmap(df_crp_death_facts, vmin=0, vmax=1)
    plt.tight_layout()
    plt.subplot(248)
    plt.title("ChronicConditionsRisk(x) => " + outcome_predicate)
    plt_heatmap(df_cronic_conditions_death_facts, vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig('implies_plots/df_knowledge_axioms.pdf')
    plt.show()

# def plot_concept_level_verification_scatter(lactate_values_before_training, lactate_values_after_training, actual_lactate_values, gt_lactate,
#                                             bilirubin_values_before_training, bilirubin_values_after_training, actual_bilirubin_values, gt_bilirubin, 
#                                             platelet_values_before_training, platelet_values_after_training, actual_platelet_values, gt_platelet, 
#                                             qSOFA_values_before_training, qSOFA_values_after_training, gt_qSOFA,
#                                             labels):
#     lactate_threshold = 4.0
#     bilirubin_threshold = 2.0
#     platelet_threshold = 150.0
#     qSOFA_threshold = 2.0

#     indices_gt = np.where(labels == 1)[0]
#     indices_not_gt = np.where(labels == 0)[0]
#     patients_d_ids = [f"Patient_{i}" for i in indices_gt]
#     patients_not_d_ids = [f"Patient_{i}" for i in indices_not_gt]

#     # Lactate

#     data_d=  {
#         'Patient_ID': patients_d_ids,
#         'Lactate': actual_lactate_values[indices_gt],
#         'HighLactate Truth Values Before': lactate_values_before_training[indices_gt],
#         'HighLactate Truth Values After': lactate_values_after_training[indices_gt],
#     }
#     df = pd.DataFrame(data_d)

#     plt.figure(figsize=(18, 8))
#     sns.set_theme(style="whitegrid")
#     sns.scatterplot(
#         x = "Lactate",
#         y="HighLactate Truth Values Before",
#         data=df,
#         label='Before Training',
#         alpha=0.7,
#         s=100
#     )
#     sns.scatterplot(
#         x = "Lactate",
#         y="HighLactate Truth Values After",
#         data=df,
#         label='After Training',
#         alpha=0.7,
#         s=100
#     )

#     for i in range(len(df)):
#         plt.plot(
#             [df["Lactate"][i], df["Lactate"][i]],
#             [df["HighLactate Truth Values Before"][i], df["HighLactate Truth Values After"][i]],
#             'k--',
#             alpha=0.3,
#             linewidth=1
#         )

#     plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=2, label='Truth Value Midpoint (0.5)')
#     plt.axvline(x=lactate_threshold, color='r', linestyle='--', linewidth=2, label=f'Lactate Threshold ({lactate_threshold})')

#     plt.title("HighLactate Truth Values For Dead Patients", fontsize=16)
#     plt.xlabel("Raw Lactate Value", fontsize=14)
#     plt.ylabel("HighLactate Truth Values", fontsize=14)
#     plt.legend(title="Legend")
    
#     data_not_d = {
#         'Patient_ID': patients_not_d_ids,
#         'Lactate': actual_lactate_values[indices_not_gt],
#         'HighLactate Truth Values Before': lactate_values_before_training[indices_not_gt],
#         'HighLactate Truth Values After': lactate_values_after_training[indices_not_gt],
#     }
#     df_not_d = pd.DataFrame(data_not_d)

#     plt.figure(figsize=(18, 8))
#     sns.set_theme(style="whitegrid")

#     # Dead patients plot
#     plt.subplot(1, 2, 1)
#     sns.scatterplot(
#         x="Lactate",
#         y="HighLactate Truth Values Before",
#         data=df,
#         label='Before Training',
#         alpha=0.7,
#         s=100
#     )
#     sns.scatterplot(
#         x="Lactate",
#         y="HighLactate Truth Values After",
#         data=df,
#         label='After Training',
#         alpha=0.7,
#         s=100
#     )
#     for i in range(len(df)):
#         plt.plot(
#             [df["Lactate"][i], df["Lactate"][i]],
#             [df["HighLactate Truth Values Before"][i], df["HighLactate Truth Values After"][i]],
#             'k--',
#             alpha=0.3,
#             linewidth=1
#         )
#     plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=2, label='Truth Value Midpoint (0.5)')
#     plt.axvline(x=lactate_threshold, color='r', linestyle='--', linewidth=2, label=f'Lactate Threshold ({lactate_threshold})')
#     plt.title("HighLactate Truth Values For Dead Patients", fontsize=14)
#     plt.xlabel("Raw Lactate Value", fontsize=12)
#     plt.ylabel("HighLactate Truth Values", fontsize=12)
#     plt.legend(title="Legend")

#     # Survivor patients plot
#     plt.subplot(1, 2, 2)
#     sns.scatterplot(
#         x="Lactate",
#         y="HighLactate Truth Values Before",
#         data=df_not_d,
#         label='Before Training',
#         alpha=0.7,
#         s=100
#     )
#     sns.scatterplot(
#         x="Lactate",
#         y="HighLactate Truth Values After",
#         data=df_not_d,
#         label='After Training',
#         alpha=0.7,
#         s=100
#     )
#     for i in range(len(df_not_d)):
#         plt.plot(
#             [df_not_d["Lactate"][i], df_not_d["Lactate"][i]],
#             [df_not_d["HighLactate Truth Values Before"][i], df_not_d["HighLactate Truth Values After"][i]],
#             'k--',
#             alpha=0.3,
#             linewidth=1
#         )
#     plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=2, label='Truth Value Midpoint (0.5)')
#     plt.axvline(x=lactate_threshold, color='r', linestyle='--', linewidth=2, label=f'Lactate Threshold ({lactate_threshold})')
#     plt.title("HighLactate Truth Values For Survivor Patients", fontsize=14)
#     plt.xlabel("Raw Lactate Value", fontsize=12)
#     plt.ylabel("HighLactate Truth Values", fontsize=12)
#     plt.legend(title="Legend")

#     # Bilirubin

#     data_d=  {
#         'Patient_ID': patients_d_ids,
#         'Bilirubin': actual_bilirubin_values[indices_gt],
#         'HighBilirubin Truth Values Before': bilirubin_values_before_training[indices_gt],
#         'HighBilirubin Truth Values After': bilirubin_values_after_training[indices_gt],
#     }

#     df = pd.DataFrame(data_d)
    
#     plt.figure(figsize=(18, 8))
#     sns.set_theme(style="whitegrid")
#     sns.scatterplot(
#         x = "Bilirubin",
#         y="HighBilirubin Truth Values Before",
#         data=df,
#         label='Before Training',
#         alpha=0.7,
#         s=100
#     )
#     sns.scatterplot(
#         x = "Bilirubin",
#         y="HighBilirubin Truth Values After",
#         data=df,
#         label='After Training',
#         alpha=0.7,
#         s=100
#     )

#     for i in range(len(df)):
#         plt.plot(
#             [df["Bilirubin"][i], df["Bilirubin"][i]],
#             [df["HighBilirubin Truth Values Before"][i], df["HighBilirubin Truth Values After"][i]],
#             'k--',
#             alpha=0.3,
#             linewidth=1
#         )

#     plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=2, label='Truth Value Midpoint (0.5)')
#     plt.axvline(x=bilirubin_threshold, color='r', linestyle='--', linewidth=2, label=f'Bilirubin Threshold ({bilirubin_threshold})')

#     plt.title("HighBilirubin Truth Values For Dead Patients", fontsize=16)
#     plt.xlabel("Raw Bilirubin Value", fontsize=14)
#     plt.ylabel("HighBilirubin Truth Values", fontsize=14)
#     plt.legend(title="Legend")

#     data_not_d = {
#         'Patient_ID': patients_not_d_ids,
#         'Bilirubin': actual_bilirubin_values[indices_not_gt],
#         'HighBilirubin Truth Values Before': bilirubin_values_before_training[indices_not_gt],
#         'HighBilirubin Truth Values After': bilirubin_values_after_training[indices_not_gt],
#     }
#     df_not_d = pd.DataFrame(data_not_d)

#     plt.figure(figsize=(18, 8))
#     sns.set_theme(style="whitegrid")

#     # Dead patients plot
#     plt.subplot(1, 2, 1)
#     sns.scatterplot(
#         x="Bilirubin",
#         y="HighBilirubin Truth Values Before",
#         data=df,
#         label='Before Training',
#         alpha=0.7,
#         s=100
#     )
#     sns.scatterplot(
#         x="Bilirubin",
#         y="HighBilirubin Truth Values After",
#         data=df,
#         label='After Training',
#         alpha=0.7,
#         s=100
#     )
#     for i in range(len(df)):
#         plt.plot(
#             [df["Bilirubin"][i], df["Bilirubin"][i]],
#             [df["HighBilirubin Truth Values Before"][i], df["HighBilirubin Truth Values After"][i]],
#             'k--',
#             alpha=0.3,
#             linewidth=1
#         )
#     plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=2, label='Truth Value Midpoint (0.5)')
#     plt.axvline(x=bilirubin_threshold, color='r', linestyle='--', linewidth=2, label=f'Bilirubin Threshold ({bilirubin_threshold})')
#     plt.title("HighBilirubin Truth Values For Dead Patients", fontsize=14)
#     plt.xlabel("Raw Bilirubin Value", fontsize=12)
#     plt.ylabel("HighBilirubin Truth Values", fontsize=12)
#     plt.legend(title="Legend")

#     # Survivor patients plot
#     plt.subplot(1, 2, 2)
#     sns.scatterplot(
#         x="Bilirubin",
#         y="HighBilirubin Truth Values Before",
#         data=df_not_d,
#         label='Before Training',
#         alpha=0.7,
#         s=100
#     )
#     sns.scatterplot(
#         x="Bilirubin",
#         y="HighBilirubin Truth Values After",
#         data=df_not_d,
#         label='After Training',
#         alpha=0.7,
#         s=100
#     )
#     for i in range(len(df_not_d)):
#         plt.plot(
#             [df_not_d["Bilirubin"][i], df_not_d["Bilirubin"][i]],
#             [df_not_d["HighBilirubin Truth Values Before"][i], df_not_d["HighBilirubin Truth Values After"][i]],
#             'k--',
#             alpha=0.3,
#             linewidth=1
#         )
#     plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=2, label='Truth Value Midpoint (0.5)')
#     plt.axvline(x=bilirubin_threshold, color='r', linestyle='--', linewidth=2, label=f'Bilirubin Threshold ({bilirubin_threshold})')
#     plt.title("HighBilirubin Truth Values For Survivor Patients", fontsize=14)
#     plt.xlabel("Raw Bilirubin Value", fontsize=12)
#     plt.ylabel("HighBilirubin Truth Values", fontsize=12)
#     plt.legend(title="Legend")

#     plt.tight_layout()
#     plt.savefig('plots/truth_values_dead_vs_survivors.pdf', format='pdf', dpi=1000, bbox_inches='tight')
#     plt.show()

def plot_concept_level_verification_scatter(lactate_values_before_training, lactate_values_after_training, actual_lactate_values,
                                            bilirubin_values_before_training, bilirubin_values_after_training, actual_bilirubin_values, 
                                            platelet_values_before_training, platelet_values_after_training, actual_platelet_values, 
                                            wbc_values_before_training, wbc_values_after_training, actual_wbc_values,
                                            labels):

    # Thresholds
    lactate_threshold = 4.0
    bilirubin_threshold = 2.0
    respiratory_rate_threshold = 22
    platelet_threshold = 50
    glucose_threshold = 100
    creatinine_threshold = 1.5
    crp_threshold = 100  # Assuming a threshold for CRP
    map_threshold = 65  # Assuming a threshold for MAP
    blood_pressure_systolic_threshold = 100
    wbc_threshold = 30

    # Indices
    indices_gt = np.where(labels == 1)[0]
    indices_not_gt = np.where(labels == 0)[0]

    sns.set_theme(style="whitegrid")

    # 2 rows x 4 columns = 8 subplots
    fig, axs = plt.subplots(4, 2, figsize=(20, 25))  # Wider to fit all plots

    concepts = [
        {
            "name": "Lactate",
            "actual": actual_lactate_values,
            "before": lactate_values_before_training,
            "after": lactate_values_after_training,
            "threshold": lactate_threshold,
            "xlabel": "Raw Lactate Value",
            "title": "HighLactate"
        },
        {
            "name": "Bilirubin",
            "actual": actual_bilirubin_values,
            "before": bilirubin_values_before_training,
            "after": bilirubin_values_after_training,
            "threshold": bilirubin_threshold,
            "xlabel": "Raw Bilirubin Value",
            "title": "HighBilirubin"
        },
        {
            "name": "Platelets",
            "actual": actual_platelet_values,
            "before": platelet_values_before_training,
            "after": platelet_values_after_training,
            "threshold": platelet_threshold,
            "xlabel": "Raw Platelets Value",
            "title": "LowPlatelets"
        },
        {
            "name": "WBC",
            "actual": actual_wbc_values,
            "before": wbc_values_before_training,
            "after": wbc_values_after_training,
            "threshold": wbc_threshold,
            "xlabel": "Raw WBC Value",
            "title": "HighWBC"
        },
    ]

    for row, concept in enumerate(concepts):
        for col, (indices, group_label) in enumerate(zip([indices_gt, indices_not_gt], ["Deaths", "Survivors"])):
            ax = axs[row, col]
            df = pd.DataFrame({
                concept["name"]: concept["actual"][indices],
                "Before": concept["before"][indices],
                "After": concept["after"][indices],
            })

            sns.scatterplot(x=concept["name"], y="Before", data=df, ax=ax, label="Before Training", alpha=0.7, s=100)
            sns.scatterplot(x=concept["name"], y="After", data=df, ax=ax, label="After Training", alpha=0.7, s=100)

            # Connect points
            for i in range(len(df)):
                ax.plot(
                    [df[concept["name"]][i], df[concept["name"]][i]],
                    [df["Before"][i], df["After"][i]],
                    'k--', alpha=0.3, linewidth=1
                )

            ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=2)
            ax.axvline(x=concept["threshold"], color='r', linestyle='--', linewidth=2, label=f'Threshold ({concept["threshold"]})')

            ax.set_title(f"{concept['title']} Truth Values ({group_label})", fontsize=20)
            ax.set_xlabel(concept["xlabel"], fontsize=16)
            ax.set_ylabel("Truth Values", fontsize=16)
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            ax.legend()

    plt.tight_layout()
    plt.savefig("plots/concept_truth_values_grid_2x4.pdf", format='pdf', dpi=1000)
    plt.show()


def plot_global_axioms_violin(values, len_deaths, len_survivors):
    data = {
        'formula': ['HighLactate⇒HighMortality'] * (len_deaths+len_survivors) + ['LactateNotClearing⇒HighMortality'] * (len_deaths+len_survivors) + ['HighBilirubin⇒HighMortality'] * (len_deaths+len_survivors) + ['LowPlatelets⇒HighMortality'] * (len_deaths+len_survivors) 
        # + ['LowMAP⇒HighMortality'] * (len_deaths+len_survivors) +
        #            ['HighWBC⇒HighMortality'] * (len_deaths+len_survivors) + ['HighCRP⇒HighMortality'] * (len_deaths+len_survivors) + ['HasCronicConditions⇒HighMortality'] * (len_deaths+len_survivors),
        ,
        'truth_value': values,
        'Outcome': (
            ['Deaths'] * len_deaths + ['Survivors'] * len_survivors +
            ['Deaths'] * len_deaths + ['Survivors'] * len_survivors +
            ['Deaths'] * len_deaths + ['Survivors'] * len_survivors +
            # ['Deaths'] * len_deaths + ['Survivors'] * len_survivors +
            # ['Deaths'] * len_deaths + ['Survivors'] * len_survivors +
            # ['Deaths'] * len_deaths + ['Survivors'] * len_survivors +
            # ['Deaths'] * len_deaths + ['Survivors'] * len_survivors +
            ['Deaths'] * len_deaths + ['Survivors'] * len_survivors
        )
    }

    df = pd.DataFrame(data)

    # Violin plot with embedded box plot
    fig, ax = plt.subplots(figsize=(35, 20))
    sns.violinplot(
        x='formula', y='truth_value', hue='Outcome',
        data=df, split=True, inner='box',
        palette={'Deaths': 'lightcoral', 'Survivors': 'skyblue'},
        ax=ax
    )
    plt.title('Truth Values of Implication Formulas by Mortality Class', fontsize=22)
    ax.set_xlabel('Formula', fontsize=18)
    ax.set_ylabel('Truth Value', fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=16, title_fontsize=16)
    plt.ylim(0, 1)
    plt.savefig("plots/global-axioms.pdf", format='pdf', dpi=1000)
    plt.show()

def plot_global_axioms_density(values, len_deaths, len_survivors):
    formulas = [
        'HighLactate⇒HighMortality',
        'LactateNotClearing⇒HighMortality',
        'HighBilirubin⇒HighMortality',
        'LowPlatelets⇒HighMortality',
        'LowMAP⇒HighMortality',
        'HighWBC⇒HighMortality',
        'HighCRP⇒HighMortality',
        'HasCronicConditions⇒HighMortality'
    ]
    data = {
        'formula': ['HighLactate⇒HighMortality'] * (len_deaths+len_survivors) + ['LactateNotClearing⇒HighMortality'] * (len_deaths+len_survivors) + ['HighBilirubin⇒HighMortality'] * (len_deaths+len_survivors) + ['LowPlatelets⇒HighMortality'] * (len_deaths+len_survivors) 
        + ['LowMAP⇒HighMortality'] * (len_deaths+len_survivors) +
                   ['HighWBC⇒HighMortality'] * (len_deaths+len_survivors) + ['HighCRP⇒HighMortality'] * (len_deaths+len_survivors) + ['HasCronicConditions⇒HighMortality'] * (len_deaths+len_survivors),
        'truth_value': values,
        'Outcome': (
            ['Deaths'] * len_deaths + ['Survivors'] * len_survivors +
            ['Deaths'] * len_deaths + ['Survivors'] * len_survivors +
            ['Deaths'] * len_deaths + ['Survivors'] * len_survivors +
            ['Deaths'] * len_deaths + ['Survivors'] * len_survivors +
            ['Deaths'] * len_deaths + ['Survivors'] * len_survivors +
            ['Deaths'] * len_deaths + ['Survivors'] * len_survivors +
            ['Deaths'] * len_deaths + ['Survivors'] * len_survivors +
            ['Deaths'] * len_deaths + ['Survivors'] * len_survivors
        )
    }

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    axes = axes.flatten()
    for i, formula in enumerate(formulas):
        ax = axes[i]
        subset = df[df['formula'] == formula]
        sns.kdeplot(data=subset, x="truth_value", hue="Outcome", fill=True, ax=ax, common_norm=False)
        ax.set_title(formula)
        ax.set_xlim(0, 1)

    # Remove unused subplots if any (in case number of formulas < 8)
    for j in range(len(formulas), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("plots/global-axiom-density.pdf", format='pdf', dpi=1000)
    plt.show()
