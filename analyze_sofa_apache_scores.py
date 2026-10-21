#!/usr/bin/env python3
"""
Mortality Prediction Analyzer
Analyzes SOFA and APACHE II score performance in predicting mortality
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, 
    roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns


def get_failed_predictions(scores_file, sofa_threshold=10, apache_threshold=20):
    """
    Simple function that returns list of hadm_ids where both scores failed to identify deaths
    
    Parameters:
    -----------
    scores_file : str
        Path to CSV file with per-patient scores
    sofa_threshold : int
        SOFA score threshold for predicting death (default: 10)
    apache_threshold : int
        APACHE II score threshold for predicting death (default: 20)
    
    Returns:
    --------
    list : List of hadm_ids where patient died but both scores predicted survival
    
    Example:
    --------
    >>> failed_ids = get_failed_predictions('patient_scores.csv')
    >>> print(f"Failed predictions: {failed_ids}")
    """
    df = pd.read_csv(scores_file)
    
    # Filter to patients who died but both scores predicted survival
    failed = df[
        (df['hospital_expire_flag'] == 1) &  # Patient died
        (df['SOFA_score'] < sofa_threshold) &  # SOFA predicted survival
        (df['APACHE_II_score'] < apache_threshold)  # APACHE predicted survival
    ]
    
    return failed['hadm_id'].tolist()


def calculate_mortality_risk_sofa(sofa_score):
    """
    Calculate mortality risk based on SOFA score
    Returns 1 (predict death) or 0 (predict survival)
    
    SOFA Score Mortality Risk:
    - 0-6: Low risk (~10% mortality) -> Predict Survival
    - 7-9: Moderate risk (~15-20% mortality) -> Predict Survival  
    - 10-12: High risk (~40-50% mortality) -> Predict Death
    - 13+: Very high risk (>80% mortality) -> Predict Death
    """
    if pd.isna(sofa_score):
        return np.nan
    
    # Threshold: SOFA >= 10 predicts death
    return 1 if sofa_score >= 10 else 0


def calculate_mortality_risk_apache(apache_score):
    """
    Calculate mortality risk based on APACHE II score
    Returns 1 (predict death) or 0 (predict survival)
    
    APACHE II Mortality Risk:
    - 0-4: <4% mortality -> Predict Survival
    - 5-9: 4-8% mortality -> Predict Survival
    - 10-14: 15-20% mortality -> Predict Survival
    - 15-19: 25-30% mortality -> Predict Survival
    - 20-24: 40-50% mortality -> Predict Death
    - 25-29: 50-75% mortality -> Predict Death
    - 30-34: 75-85% mortality -> Predict Death
    - 35+: >85% mortality -> Predict Death
    """
    if pd.isna(apache_score):
        return np.nan
    
    # Threshold: APACHE II >= 20 predicts death
    return 1 if apache_score >= 20 else 0


def analyze_mortality_predictions(scores_file, sofa_threshold=10, apache_threshold=20):
    """
    Analyze mortality prediction performance of SOFA and APACHE II scores
    
    Parameters:
    -----------
    scores_file : str
        Path to CSV file with per-patient scores
    sofa_threshold : int
        SOFA score threshold for predicting death (default: 10)
    apache_threshold : int
        APACHE II score threshold for predicting death (default: 20)
    
    Returns:
    --------
    dict : Dictionary containing analysis results and failure cases
    """
    print(f"Loading patient scores from: {scores_file}")
    df = pd.read_csv(scores_file)
    
    print(f"Total patients: {len(df)}")
    print(f"Columns available: {df.columns.tolist()}\n")
    
    # Check required columns
    required_cols = ['hadm_id', 'SOFA_score', 'APACHE_II_score', 'hospital_expire_flag']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove rows with missing scores or outcomes
    df_clean = df.dropna(subset=['SOFA_score', 'APACHE_II_score', 'hospital_expire_flag']).copy()
    print(f"Patients with complete data: {len(df_clean)}")
    
    # Calculate predictions using custom thresholds
    df_clean['sofa_predicts_death'] = df_clean['SOFA_score'] >= sofa_threshold
    df_clean['apache_predicts_death'] = df_clean['APACHE_II_score'] >= apache_threshold
    
    # Convert to integer for easier comparison
    df_clean['sofa_predicts_death'] = df_clean['sofa_predicts_death'].astype(int)
    df_clean['apache_predicts_death'] = df_clean['apache_predicts_death'].astype(int)
    
    # Actual outcome
    df_clean['actually_died'] = df_clean['hospital_expire_flag'].astype(int)
    
    # Find cases where scores got it wrong
    df_clean['sofa_correct'] = df_clean['sofa_predicts_death'] == df_clean['actually_died']
    df_clean['apache_correct'] = df_clean['apache_predicts_death'] == df_clean['actually_died']
    df_clean['both_correct'] = df_clean['sofa_correct'] & df_clean['apache_correct']
    
    # Find cases where both scores FAILED to identify deaths
    # (Patient died but both scores predicted survival)
    both_failed_deaths = df_clean[
        (df_clean['actually_died'] == 1) & 
        (df_clean['sofa_predicts_death'] == 0) & 
        (df_clean['apache_predicts_death'] == 0)
    ].copy()
    
    # Find cases where both scores FALSELY predicted death
    # (Patient survived but both scores predicted death)
    both_false_alarms = df_clean[
        (df_clean['actually_died'] == 0) & 
        (df_clean['sofa_predicts_death'] == 1) & 
        (df_clean['apache_predicts_death'] == 1)
    ].copy()
    
    # Calculate metrics
    results = {
        'total_patients': len(df_clean),
        'actual_deaths': df_clean['actually_died'].sum(),
        'actual_survivors': (df_clean['actually_died'] == 0).sum(),
        'mortality_rate': df_clean['actually_died'].mean() * 100,
        
        # SOFA performance
        'sofa_threshold': sofa_threshold,
        'sofa_predicted_deaths': df_clean['sofa_predicts_death'].sum(),
        'sofa_accuracy': df_clean['sofa_correct'].mean() * 100,
        'sofa_sensitivity': None,  # Will calculate below
        'sofa_specificity': None,
        'sofa_ppv': None,
        'sofa_npv': None,
        
        # APACHE II performance
        'apache_threshold': apache_threshold,
        'apache_predicted_deaths': df_clean['apache_predicts_death'].sum(),
        'apache_accuracy': df_clean['apache_correct'].mean() * 100,
        'apache_sensitivity': None,
        'apache_specificity': None,
        'apache_ppv': None,
        'apache_npv': None,
        
        # Combined performance
        'both_correct': df_clean['both_correct'].sum(),
        'both_correct_pct': df_clean['both_correct'].mean() * 100,
        
        # Failure cases
        'both_failed_deaths_count': len(both_failed_deaths),
        'both_failed_deaths_hadm_ids': both_failed_deaths['hadm_id'].tolist(),
        'both_false_alarms_count': len(both_false_alarms),
        'both_false_alarms_hadm_ids': both_false_alarms['hadm_id'].tolist(),
        
        # DataFrames for detailed analysis
        'df_clean': df_clean,
        'both_failed_deaths_df': both_failed_deaths,
        'both_false_alarms_df': both_false_alarms
    }
    
    # Calculate detailed metrics for SOFA
    if results['actual_deaths'] > 0:
        sofa_cm = confusion_matrix(df_clean['actually_died'], df_clean['sofa_predicts_death'])
        tn, fp, fn, tp = sofa_cm.ravel()
        
        results['sofa_sensitivity'] = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0  # True Positive Rate
        results['sofa_specificity'] = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0  # True Negative Rate
        results['sofa_ppv'] = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0  # Positive Predictive Value
        results['sofa_npv'] = tn / (tn + fn) * 100 if (tn + fn) > 0 else 0  # Negative Predictive Value
        results['sofa_confusion_matrix'] = sofa_cm
        
        try:
            results['sofa_auc'] = roc_auc_score(df_clean['actually_died'], df_clean['SOFA_score'])
        except:
            results['sofa_auc'] = None
    
    # Calculate detailed metrics for APACHE II
    if results['actual_deaths'] > 0:
        apache_cm = confusion_matrix(df_clean['actually_died'], df_clean['apache_predicts_death'])
        tn, fp, fn, tp = apache_cm.ravel()
        
        results['apache_sensitivity'] = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        results['apache_specificity'] = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        results['apache_ppv'] = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        results['apache_npv'] = tn / (tn + fn) * 100 if (tn + fn) > 0 else 0
        results['apache_confusion_matrix'] = apache_cm
        
        try:
            results['apache_auc'] = roc_auc_score(df_clean['actually_died'], df_clean['APACHE_II_score'])
        except:
            results['apache_auc'] = None
    
    return results


def print_analysis_report(results):
    """
    Print a comprehensive analysis report
    """
    print("\n" + "="*70)
    print("MORTALITY PREDICTION ANALYSIS REPORT")
    print("="*70)
    
    print(f"\n{'DATASET OVERVIEW':^70}")
    print("-"*70)
    print(f"Total patients: {results['total_patients']}")
    print(f"Actual deaths: {results['actual_deaths']} ({results['mortality_rate']:.1f}%)")
    print(f"Actual survivors: {results['actual_survivors']} ({100-results['mortality_rate']:.1f}%)")
    
    print(f"\n{'SOFA SCORE PERFORMANCE':^70}")
    print("-"*70)
    print(f"Threshold used: SOFA >= {results['sofa_threshold']} predicts death")
    print(f"Predicted deaths: {results['sofa_predicted_deaths']}")
    print(f"\nAccuracy: {results['sofa_accuracy']:.1f}%")
    if results['sofa_sensitivity'] is not None:
        print(f"Sensitivity (recall): {results['sofa_sensitivity']:.1f}%  <- % of deaths correctly identified")
        print(f"Specificity: {results['sofa_specificity']:.1f}%  <- % of survivors correctly identified")
        print(f"PPV (precision): {results['sofa_ppv']:.1f}%  <- % of death predictions that were correct")
        print(f"NPV: {results['sofa_npv']:.1f}%  <- % of survival predictions that were correct")
        if results['sofa_auc'] is not None:
            print(f"AUC-ROC: {results['sofa_auc']:.3f}")
    
    print(f"\n{'APACHE II SCORE PERFORMANCE':^70}")
    print("-"*70)
    print(f"Threshold used: APACHE II >= {results['apache_threshold']} predicts death")
    print(f"Predicted deaths: {results['apache_predicted_deaths']}")
    print(f"\nAccuracy: {results['apache_accuracy']:.1f}%")
    if results['apache_sensitivity'] is not None:
        print(f"Sensitivity (recall): {results['apache_sensitivity']:.1f}%  <- % of deaths correctly identified")
        print(f"Specificity: {results['apache_specificity']:.1f}%  <- % of survivors correctly identified")
        print(f"PPV (precision): {results['apache_ppv']:.1f}%  <- % of death predictions that were correct")
        print(f"NPV: {results['apache_npv']:.1f}%  <- % of survival predictions that were correct")
        if results['apache_auc'] is not None:
            print(f"AUC-ROC: {results['apache_auc']:.3f}")
    
    print(f"\n{'COMBINED ANALYSIS':^70}")
    print("-"*70)
    print(f"Both scores correct: {results['both_correct']} ({results['both_correct_pct']:.1f}%)")
    
    print(f"\n{'FAILURE CASES - MISSED DEATHS':^70}")
    print("-"*70)
    print(f"Patients who died but BOTH scores predicted survival: {results['both_failed_deaths_count']}")
    if results['both_failed_deaths_count'] > 0:
        print(f"\nhadm_id list of missed deaths:")
        for hadm_id in results['both_failed_deaths_hadm_ids']:
            print(f"  - {hadm_id}")
        
        # Show details of missed deaths
        print(f"\nDetailed information on missed deaths:")
        failed_df = results['both_failed_deaths_df']
        cols_to_show = ['hadm_id', 'SOFA_score', 'APACHE_II_score', 'anchor_age']
        if 'last_careunit' in failed_df.columns:
            cols_to_show.append('last_careunit')
        print(failed_df[cols_to_show].to_string(index=False))
    
    print(f"\n{'FAILURE CASES - FALSE ALARMS':^70}")
    print("-"*70)
    print(f"Patients who survived but BOTH scores predicted death: {results['both_false_alarms_count']}")
    if results['both_false_alarms_count'] > 0:
        print(f"\nhadm_id list of false alarms:")
        for hadm_id in results['both_false_alarms_hadm_ids']:
            print(f"  - {hadm_id}")
        
        # Show details of false alarms
        print(f"\nDetailed information on false alarms:")
        false_df = results['both_false_alarms_df']
        cols_to_show = ['hadm_id', 'SOFA_score', 'APACHE_II_score', 'anchor_age']
        if 'last_careunit' in false_df.columns:
            cols_to_show.append('last_careunit')
        print(false_df[cols_to_show].to_string(index=False))
    
    print("\n" + "="*70)


def save_failure_cases(results, output_dir=''):
    """
    Save detailed information about failure cases
    """
    # Save missed deaths
    if results['both_failed_deaths_count'] > 0:
        missed_file = f'{output_dir}missed_deaths_by_both_scores.csv'
        results['both_failed_deaths_df'].to_csv(missed_file, index=False)
        print(f"\nSaved missed deaths to: {missed_file}")
    
    # Save false alarms
    if results['both_false_alarms_count'] > 0:
        false_file = f'{output_dir}false_alarms_by_both_scores.csv'
        results['both_false_alarms_df'].to_csv(false_file, index=False)
        print(f"Saved false alarms to: {false_file}")
    
    # Save complete analysis
    analysis_file = f'{output_dir}mortality_prediction_analysis.csv'
    results['df_clean'].to_csv(analysis_file, index=False)
    print(f"Saved complete analysis to: {analysis_file}")
    
    return missed_file if results['both_failed_deaths_count'] > 0 else None


def plot_confusion_matrices(results, output_dir=''):
    """
    Create confusion matrix visualizations
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # SOFA confusion matrix
    if 'sofa_confusion_matrix' in results:
        cm = results['sofa_confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=['Predict Survive', 'Predict Death'],
                   yticklabels=['Actually Survived', 'Actually Died'])
        axes[0].set_title(f'SOFA Score (threshold={results["sofa_threshold"]})\n'
                         f'Accuracy: {results["sofa_accuracy"]:.1f}%')
        axes[0].set_ylabel('Actual Outcome')
        axes[0].set_xlabel('Predicted Outcome')
    
    # APACHE II confusion matrix
    if 'apache_confusion_matrix' in results:
        cm = results['apache_confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                   xticklabels=['Predict Survive', 'Predict Death'],
                   yticklabels=['Actually Survived', 'Actually Died'])
        axes[1].set_title(f'APACHE II Score (threshold={results["apache_threshold"]})\n'
                         f'Accuracy: {results["apache_accuracy"]:.1f}%')
        axes[1].set_ylabel('Actual Outcome')
        axes[1].set_xlabel('Predicted Outcome')
    
    plt.tight_layout()
    output_file = f'{output_dir}confusion_matrices.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved confusion matrices to: {output_file}")
    plt.close()


def main():
    """
    Main execution function
    """
    import sys
    
    # Check if file path provided
    if len(sys.argv) > 1:
        scores_file = sys.argv[1]
    else:
        # Default to the output from calculate_clinical_scores.py
        scores_file = 'patient_scores_per_patient.csv'
        print(f"No file specified, using default: {scores_file}")
    
    # Optional: custom thresholds
    sofa_threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    apache_threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    try:
        # Analyze predictions
        results = analyze_mortality_predictions(
            scores_file, 
            sofa_threshold=sofa_threshold,
            apache_threshold=apache_threshold
        )
        
        # Print report
        print_analysis_report(results)
        
        # Save failure cases
        save_failure_cases(results)
        
        # Create visualizations
        try:
            plot_confusion_matrices(results)
        except Exception as e:
            print(f"\nNote: Could not create plots: {e}")
        
        # Return the list of hadm_ids where both scores failed
        if results['both_failed_deaths_count'] > 0:
            print(f"\n{'SUMMARY':^70}")
            print("-"*70)
            print(f"List of {results['both_failed_deaths_count']} hadm_id(s) where BOTH scores failed to identify deaths:")
            print(results['both_failed_deaths_hadm_ids'])
        else:
            print(f"\n{'SUMMARY':^70}")
            print("-"*70)
            print("No cases where both scores failed to identify deaths.")
        
    except FileNotFoundError:
        print(f"Error: File not found: {scores_file}")
        print("\nUsage: python analyze_mortality_predictions.py <path_to_scores_csv> [sofa_threshold] [apache_threshold]")
        print("Example: python analyze_mortality_predictions.py patient_scores_per_patient.csv 10 20")
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing predictions: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()