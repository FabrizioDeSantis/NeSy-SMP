import re
import pandas as pd
import numpy as np


def extract_patient_data_from_file(file_path):
    """
    Extract patient IDs and predictions from LTN prediction text file
    
    Parameters:
    -----------
    file_path : str
        Path to the text file containing LTN predictions
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with columns:
        - patient_id: Extracted patient ID
        - ltn_mortality_risk: LTN prediction value
        - lactate_risk: LactateRisk output
        - high_bilirubin: HighBilirubin output
        - platelet_low: PlateletLow output
        - lactate_not_clearing: LactateNotClearing output
        - map_risk: MeanArterialPressureRisk output
        - chronic_conditions_risk: CronicConditionsRisk output
        - crp_risk: CRPRisk output
        - wbc_risk: WBCRisk output
        - age_risk: AgeRisk output
    """
    print(f"Reading file: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split by "Patient ID:" to get individual patient blocks
    patient_blocks = content.split('Patient ID:')[1:]  # Skip the first empty split
    
    print(f"Found {len(patient_blocks)} patient records")
    
    patients_data = []
    
    for block in patient_blocks:
        # Extract patient ID (first line of each block)
        patient_id_match = re.search(r'^(\d+)', block.strip())
        if not patient_id_match:
            continue
        
        patient_id = patient_id_match.group(1)
        
        # Extract LTN Prediction (HighMortalityRisk)
        ltn_match = re.search(r'LTN Prediction \(HighMortalityRisk\): \[\[([0-9.]+)\]\]', block)
        ltn_value = float(ltn_match.group(1)) if ltn_match else np.nan
        
        # Extract all risk outputs
        lactate_risk_match = re.search(r'LactateRisk Output: \[\[([0-9.e-]+)\]\]', block)
        lactate_risk = float(lactate_risk_match.group(1)) if lactate_risk_match else np.nan
        
        bilirubin_match = re.search(r'HighBilirubin Output: \[\[([0-9.e-]+)\]\]', block)
        bilirubin_risk = float(bilirubin_match.group(1)) if bilirubin_match else np.nan
        
        platelet_match = re.search(r'PlateletLow Output: \[\[([0-9.e-]+)\]\]', block)
        platelet_risk = float(platelet_match.group(1)) if platelet_match else np.nan
        
        lactate_clearing_match = re.search(r'LactateNotClearing Output: \[\[([0-9.e-]+)\]\]', block)
        lactate_clearing = float(lactate_clearing_match.group(1)) if lactate_clearing_match else np.nan
        
        map_match = re.search(r'MeanArterialPressureRisk Output: \[\[([0-9.e-]+)\]\]', block)
        map_risk = float(map_match.group(1)) if map_match else np.nan
        
        chronic_match = re.search(r'CronicConditionsRisk Output: \[\[([0-9.e-]+)\]\]', block)
        chronic_risk = float(chronic_match.group(1)) if chronic_match else np.nan
        
        crp_match = re.search(r'CRPRisk Output: \[\[([0-9.e-]+)\]\]', block)
        crp_risk = float(crp_match.group(1)) if crp_match else np.nan
        
        wbc_match = re.search(r'WBCRisk Output: \[\[([0-9.e-]+)\]\]', block)
        wbc_risk = float(wbc_match.group(1)) if wbc_match else np.nan
        
        age_match = re.search(r'AgeRisk Output: \[\[([0-9.e-]+)\]\]', block)
        age_risk = float(age_match.group(1)) if age_match else np.nan
        
        # Extract age value
        age_value_match = re.search(r'Age Value: \[([0-9.]+)\]', block)
        age_value = float(age_value_match.group(1)) if age_value_match else np.nan
        
        patients_data.append({
            'patient_id': patient_id,
            'ltn_mortality_risk': ltn_value,
            'lactate_risk': lactate_risk,
            'high_bilirubin': bilirubin_risk,
            'platelet_low': platelet_risk,
            'lactate_not_clearing': lactate_clearing,
            'map_risk': map_risk,
            'chronic_conditions_risk': chronic_risk,
            'crp_risk': crp_risk,
            'wbc_risk': wbc_risk,
            'age_risk': age_risk,
            'age_value': age_value
        })
    
    df = pd.DataFrame(patients_data)
    
    # Sort by patient ID
    df = df.sort_values('patient_id').reset_index(drop=True)
    
    return df


def extract_patient_ids_only(file_path):
    """
    Extract only patient IDs from the file (simpler version)
    
    Parameters:
    -----------
    file_path : str
        Path to the text file containing LTN predictions
    
    Returns:
    --------
    list : List of patient IDs as strings
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all patient IDs using regex
    patient_ids = re.findall(r'Patient ID: (\d+)', content)
    
    return patient_ids


def classify_mortality_risk(ltn_score, threshold=0.5):
    """
    Classify mortality risk based on LTN score
    
    Parameters:
    -----------
    ltn_score : float
        LTN mortality risk score (0-1)
    threshold : float
        Threshold for high mortality risk (default: 0.5)
    
    Returns:
    --------
    str : 'High' or 'Low' risk classification
    """
    if pd.isna(ltn_score):
        return 'Unknown'
    return 'High' if ltn_score >= threshold else 'Low'


def save_extracted_data(df, output_file='/mnt/user-data/outputs/ltn_predictions.csv'):
    """
    Save extracted patient data to CSV
    """
    df.to_csv(output_file, index=False)
    print(f"\nSaved extracted data to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total patients extracted: {len(df)}")
    
    if 'ltn_mortality_risk' in df.columns:
        print(f"\nLTN Mortality Risk Statistics:")
        print(f"  Mean: {df['ltn_mortality_risk'].mean():.4f}")
        print(f"  Median: {df['ltn_mortality_risk'].median():.4f}")
        print(f"  Range: {df['ltn_mortality_risk'].min():.4f} - {df['ltn_mortality_risk'].max():.4f}")
        print(f"  Std Dev: {df['ltn_mortality_risk'].std():.4f}")
        
        # Classify patients
        df['risk_category'] = df['ltn_mortality_risk'].apply(classify_mortality_risk)
        high_risk = (df['risk_category'] == 'High').sum()
        low_risk = (df['risk_category'] == 'Low').sum()
        
        print(f"\nRisk Classification (threshold=0.5):")
        print(f"  High Risk: {high_risk} patients ({high_risk/len(df)*100:.1f}%)")
        print(f"  Low Risk: {low_risk} patients ({low_risk/len(df)*100:.1f}%)")
    
    # Show top risk factors
    risk_columns = ['lactate_risk', 'high_bilirubin', 'platelet_low', 
                   'lactate_not_clearing', 'map_risk', 'chronic_conditions_risk',
                   'crp_risk', 'wbc_risk', 'age_risk']
    
    print(f"\nTop Risk Factor Contributions (Mean):")
    for col in risk_columns:
        if col in df.columns:
            mean_val = df[col].mean()
            print(f"  {col}: {mean_val:.4f}")
    
    print("\n" + "="*60)
    
    return output_file


def main():
    """
    Main execution function
    """
    import sys
    
    # Check if file path provided
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # Default to looking in uploads directory
        input_file = 'interesting_case_details_final.txt'
        print(f"No file specified, using default: {input_file}")
    
    try:
        # Extract patient data
        df = extract_patient_data_from_file(input_file)
        
        # Save results
        # output_file = save_extracted_data(df)
        
        # Also print patient IDs list
        print(f"\nPatient IDs:")
        print(df['patient_id'].tolist())
        df["patient_id"] = df["patient_id"].astype(str)
        patients = df['patient_id'].tolist()
        
        # Show first few records
        print(f"\nFirst 5 records:")
        print(df.head().to_string(index=False))
        # print column types
        print("\nColumn Data Types:")
        print(df.dtypes)
        
        print("\n" + "="*60)
        print("EXTRACTION COMPLETE")
        print("="*60)

        missed_deaths_by_both_scores = pd.read_csv('missed_deaths_by_both_scores.csv')
        missed_deaths_by_both_scores['hadm_id'] = missed_deaths_by_both_scores['hadm_id'].astype(str)
        missed_deaths_by_both_scores = missed_deaths_by_both_scores[missed_deaths_by_both_scores['hadm_id'].isin(patients)]
        df = df[df["patient_id"].isin(missed_deaths_by_both_scores['hadm_id'].tolist())]

        # find patients with lactate_not_clearing > 0.5, chronic_conditions_risk > 0.5, age_risk > 0.5
        filtered_patients = df[(df['chronic_conditions_risk'] > 0.5)]
        print("\nPatients with high risk factors (lactate_not_clearing, chronic_conditions_risk, age_risk > 0.5):")
        print(filtered_patients[['patient_id', 'lactate_not_clearing', 'chronic_conditions_risk', 'age_risk']].to_string(index=False))
        
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
        print("\nUsage: python extract_patient_ids.py <path_to_txt_file>")
        print("Example: python extract_patient_ids.py ltn_predictions.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error extracting data: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()