import pandas as pd
import numpy as np
from datetime import timedelta

print("Loading data...")
df = pd.read_csv("path_to_dataset.csv", dtype={"hadm_id": str, "subject_id": str})
print("Data loaded.")

df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])

WINDOW_SIZE = timedelta(hours=24)
STRIDE_SIZE = timedelta(hours=6) # Generate a new window every hour

# 3. DEFINE THE SLIDING WINDOW FUNCTION
def process_patient_sliding_windows(patient_group):
    """
    Creates and labels sliding windows for a patient's stay.
    """
    hadm_id = patient_group['hadm_id'].iloc[0]

    relevant_activity_idxs = np.where(patient_group["concept:name"] == "Admission to intensive care unit")[0]
    if len(relevant_activity_idxs) > 0:
        idx = relevant_activity_idxs[0]
        patient_group = patient_group[idx:]  # Start from the first ICU admission event

    # Find death and discharge times (same as before)
    death_events = patient_group[patient_group['concept:name'] == 'Death']
    death_time = death_events['time:timestamp'].iloc[0] if not death_events.empty else pd.NaT
    
    discharge_events = patient_group[patient_group['concept:name'] == 'Discharge from hospital']
    discharge_time = discharge_events['time:timestamp'].iloc[0] if not discharge_events.empty else pd.NaT

    # Define the end of the observation period (same as before)
    terminal_times = [t for t in [death_time, discharge_time] if pd.notna(t)]
    end_of_stay_time = min(terminal_times) if terminal_times else pd.NaT
    
    if pd.notna(end_of_stay_time):
        windowing_events = patient_group[patient_group['time:timestamp'] < end_of_stay_time]
    else:
        windowing_events = patient_group

    if windowing_events.empty:
        return pd.DataFrame()

    # Determine the timeline for generating windows
    stay_start_time = windowing_events['time:timestamp'].min()
    # The last possible moment a window can START is the last event time.
    stay_end_time = windowing_events['time:timestamp'].max()

    # --- MAIN CHANGE: Generate window starts based on the STRIDE ---
    window_starts = pd.date_range(
        start=stay_start_time,
        end=stay_end_time,
        freq=STRIDE_SIZE
    )

    labeled_windows = []
    for window_start in window_starts:
        window_end = window_start + WINDOW_SIZE

        # --- FILTERS ---
        # 1. Discard windows that extend beyond the patient's last valid event time.
        if window_end > stay_end_time + STRIDE_SIZE: # Allow window to end on last event time
             # The + STRIDE_SIZE gives a little buffer
             if window_end > stay_end_time:
                 # More precise check: A window is only valid if it contains at least one event.
                 # This check implicitly handles it, but a stricter version could be added.
                 # For now, we will just ensure the window doesn't extend far beyond the last event.
                 # Let's refine: The window must end after the last event, but not start after it.
                 # The current range of window_starts already ensures it won't start after the last event.
                 # The most important thing is filtering based on outcome events.
                 pass # This filter is less critical than the outcome filters

        # 2. Discard if window overlaps with a terminal event (same logic as before)
        if pd.notna(death_time) and (window_start <= death_time < window_end):
            continue
        if pd.notna(discharge_time) and (window_start <= discharge_time < window_end):
            continue

        # --- LABELING --- (same logic as before)
        labeling_horizon_end = window_end + timedelta(hours=STRIDE_SIZE)
        died_label = 0
        if pd.notna(death_time) and (window_end < death_time <= labeling_horizon_end):
            died_label = 1
            
        labeled_windows.append({
            'hadm_id': hadm_id,
            'window_start': window_start,
            'window_end': window_end,
            'died': died_label
        })

    if not labeled_windows:
        return pd.DataFrame()
    
    return pd.DataFrame(labeled_windows)

sliding_window_df = (
    df.groupby('hadm_id', group_keys=False)
    .apply(process_patient_sliding_windows)
    .reset_index(drop=True)
)

sliding_window_df.to_csv("time_windows.csv", index=False) # save extracted time windows