import pandas as pd
import numpy as np
from datetime import datetime

COLUMNS_MAPPING = {
    'elint_notation':'elnot',
    'radio_frequency_value':'rf',
    'scan_rate_value':'sr',
    'pulse_duration':'pd',
    'pulse_repetition_interval':'pri',
    'lat':'lat',
    'lon':'lon',
    'info.production_name':'name',
    'info.primary_function':'pfc',
    'orientation_value':'bearing', 
    'major_axis_value':'major_axis',
    'minor_axis_value':'minor_axis', 
    '@timestamp':'time'
}

latlon_cols = ['lat', 'lon']
ellipses_cols = ['major_axis', 'minor_axis', 'bearing']

def load_data(filename):
    df = pd.read_csv(filename)
    df = df.rename(columns=COLUMNS_MAPPING)
    df['time'] = get_timestamps(df)
    df['elnot'] = df['elnot'].astype(str)
    df_processed, df_rejected = process_latlon_ellipse(df)
    df_processed = latlon_ellipse_float32(df_processed)
    return df_processed, df_rejected
    
def all_elnots(df):
    return sorted(set(df['elnot']))
    
def iter_elnots(df, yield_elnot=False):
    elnots = all_elnots(df)
    for elnot in elnots:
        res = df[df['elnot'] == elnot]
        if yield_elnot:
            yield elnot, res
        else:
            yield res

def parse_time(time_str):
    return datetime.strptime(time_str, "%b %d, %Y @ %H:%M:%S.%f")

def get_timestamps(df):
    timestamps = df['time'].to_numpy().astype(str)
    return np.array(list(map(parse_time, timestamps)), dtype='datetime64[m]')

def process_latlon_ellipse(df):
    
    # Store original unprocessed data for later
    df_original = df.copy()
    
    # Work with subset but keep track of the index
    subset_df = df[latlon_cols + ellipses_cols].copy()
    subset_df = subset_df.apply(pd.to_numeric, errors='coerce')
    
    # Identify rows that will be dropped due to NaN values
    nan_mask = subset_df.isna().any(axis=1)
    rejected_indices = df.index[nan_mask]
    
    # Keep track of valid rows (non-NaN) before dropping
    valid_mask = subset_df.dropna().index
    subset_df = subset_df.dropna()
    
    # Transform the data
    latlon = subset_df[latlon_cols].to_numpy(dtype=np.float32)
    ellipses = subset_df[ellipses_cols].to_numpy(dtype=np.float32)
    
    # Convert full-axes in km (original data) to semi-axes in meters (internal representation)
    ellipses[:, :2] *= 500.
    ellipses[:, :2] = np.clip(ellipses[:,:2], 5, None)
    
    # Update the processed DataFrame
    # First, set all transformed columns to NaN (to handle dropped rows)
    df.loc[:, latlon_cols + ellipses_cols] = np.nan
    
    # Then update only the valid rows with transformed data
    df.loc[valid_mask, latlon_cols] = latlon
    df.loc[valid_mask, ellipses_cols] = ellipses
    
    # Create processed dataframe (only rows without NaN)
    df_processed = df.loc[valid_mask].copy()
    
    # Create rejected dataframe (original unprocessed data for rejected rows)
    df_rejected = df_original.loc[rejected_indices].copy()
    
    return df_processed, df_rejected

def latlon_ellipse_float32(df):
    dtypes = {col:np.float32 for col in latlon_cols+ellipses_cols}
    return df.astype(dtypes)
    
def extract_arrays(df, names=['latlons', 'ellipses']):
    output = []
    assert len(names) == len(set(names)), "Duplicate values queried"
    for name in names:
        name = name.lower()
        assert name in ['latlons', 'ellipses', 'time'], f"Unknwon array query {name}"
        
        if name == "latlons":
            output.append(df[latlon_cols].values)
            
        if name == 'ellipses':
            output.append(df[ellipses_cols].values)
            
        if name == 'time':
            output.append(df['time'].values)
            
    return tuple(output)
        