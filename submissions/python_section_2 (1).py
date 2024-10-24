import pandas as pd
import numpy as np

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Initialize an empty distance matrix DataFrame
    locations = pd.concat([df['id_start'], df['id_end']]).unique()
    distance_matrix = pd.DataFrame(0, index=locations, columns=locations, dtype=float)

    # Populate the matrix with distances
    for _, row in df.iterrows():
        start, end, dist = row['id_start'], row['id_end'], row['distance']
        distance_matrix.at[start, end] = dist
        distance_matrix.at[end, start] = dist  # Ensure symmetry

    # Handle cumulative distances (e.g., A -> B -> C)
    for k in locations:
        for i in locations:
            for j in locations:
                distance_matrix.at[i, j] = min(
                    distance_matrix.at[i, j], 
                    distance_matrix.at[i, k] + distance_matrix.at[k, j]
                )
    
        return df


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
        records = []
    
    # Iterate over the DataFrame and store all (i, j, distance) combinations
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:  # Exclude diagonal entries
                records.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance': distance_matrix.at[id_start, id_end]
                })

      return df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    # Filter rows for the given reference_id
    reference_distances = unrolled_df[unrolled_df['id_start'] == reference_id]['distance']
    reference_avg = reference_distances.mean()

    # Define the 10% threshold range
    lower_bound = reference_avg * 0.9
    upper_bound = reference_avg * 1.1

    # Find IDs within the threshold range
    result = unrolled_df.groupby('id_start')['distance'].mean()
    ids_within_threshold = result[(result >= lower_bound) & (result <= upper_bound)].index.tolist()

       return df


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
   
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Create toll rate columns for each vehicle type
    for vehicle, rate in rates.items():
        unrolled_df[vehicle] = unrolled_df['distance'] * rate

        return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
from datetime import time

    time_ranges = [
        (time(0, 0, 0), time(10, 0, 0), 0.8),   # 00:00 - 10:00
        (time(10, 0, 0), time(18, 0, 0), 1.2),  # 10:00 - 18:00
        (time(18, 0, 0), time(23, 59, 59), 0.8) # 18:00 - 23:59
    ]
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    records = []

    for _, row in toll_rate_df.iterrows():
        for day in days_of_week:
            is_weekend = day in ['Saturday', 'Sunday']
            discount_factor = 0.7 if is_weekend else None

            for start_time, end_time, factor in time_ranges:
                # Apply weekday/weekend-specific factor
                applied_factor = discount_factor if is_weekend else factor

                record = row.to_dict()
                record.update({
                    'start_day': day,
                    'end_day': day,
                    'start_time': start_time,
                    'end_time': end_time,
                })

                for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                    record[vehicle] *= applied_factor

                records.append(record)
  
           return df
