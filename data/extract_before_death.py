import pandas as pd
import numpy as np

df = pd.read_csv("dataset_gcs.csv", dtype={"subject_id": str, "hadm_id": str})
df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])

# 3. Define a function to filter events 12 hours before death
def get_events_before_death(group):
    """
    For a group of events for a single patient, finds the 'Death' event
    and returns all events that occurred in the 12 hours leading up to and
    including the death event.
    """
    # Find the row with the 'Death' event
    death_event = group[group['concept:name'] == 'Death']

    # Proceed only if a 'Death' event exists for the group
    if not death_event.empty:
        # Get the timestamp of the first 'Death' event
        death_time = death_event['time:timestamp'].iloc[0]

        # Calculate the start of the 6-hour window
        start_time = death_time - pd.Timedelta(hours=12) # CAMBIA SOLO QUESTO
    
        # Filter the group to get events within the 6-hour window
        group_to_filter_out = group[(group['time:timestamp'] >= start_time) & (group['time:timestamp'] <= death_time)]
        group_to_filter_out2 = group[group["time:timestamp"] >= death_time]
    
        # take out the rows of group_to_filter_out from the group
        group_filtered = group[~group.index.isin(group_to_filter_out.index)]
        group_filtered = group_filtered[~group_filtered.index.isin(group_to_filter_out2.index)]

        if group_filtered.empty:
            return group_filtered

        last_event_time = group_filtered['time:timestamp'].max()
        start_time_12h = last_event_time - pd.Timedelta(hours=24)
        return group_filtered[group_filtered['time:timestamp'] >= start_time_12h]
    else:
        # Handle survivors: take a random 24-hour window of events
        if len(group) > 1:
            # Determine the total time span of the events
            min_time = group['time:timestamp'].min()
            max_time = group['time:timestamp'].max()
            total_duration = max_time - min_time

            # Define the window size
            window_size = pd.Timedelta(hours=24)

            # If the total duration is less than the window size, return the whole group
            if total_duration <= window_size:
                return group

            # Calculate the latest possible start time for a full 24h window
            latest_start_time = max_time - window_size

            # Generate a random start time
            # We can do this by finding the total possible range for the start time in seconds
            # and picking a random second within that range.
            time_range_seconds = (latest_start_time - min_time).total_seconds()
            random_offset = pd.to_timedelta(np.random.uniform(0, time_range_seconds), unit='s')
            start_time = min_time + random_offset
            end_time = start_time + window_size

            # Filter events within the random window
            return group[(group['time:timestamp'] >= start_time) & (group['time:timestamp'] <= end_time)]
        else:
            # If there's only one or zero events, return the group as is (or empty)
            return group

    # If no 'Death' event, return an empty DataFrame for this group
    return pd.DataFrame()

hadm_ids = df['hadm_id'].unique()
print(f"Number of unique HADM IDs in the DataFrame: {len(hadm_ids)}")
events_before_death_df = df.groupby('hadm_id').apply(get_events_before_death).reset_index(drop=True)
# fill nan values in the column "hospital_expire_flag" with ffill and bfill methods grouped by 'hadm_id'
events_before_death_df['hospital_expire_flag'] = events_before_death_df.groupby('hadm_id')['hospital_expire_flag'].transform(lambda x: x.ffill().bfill())
events_before_death_df = events_before_death_df.sort_values(by=['subject_id', 'hadm_id', 'time:timestamp'])
# drop rows where 'hospital_expire_flag' is NaN
events_before_death_df = events_before_death_df.dropna(subset=['hospital_expire_flag'])
events_before_death_df['hospital_expire_flag'] = events_before_death_df['hospital_expire_flag'].astype(int)
hadm_ids = events_before_death_df['hadm_id'].unique()
print(f"Number of unique HADM IDs in the filtered DataFrame: {len(hadm_ids)}")
events_before_death_df.to_csv("events_12h_before_death_gcs.csv", index=False)