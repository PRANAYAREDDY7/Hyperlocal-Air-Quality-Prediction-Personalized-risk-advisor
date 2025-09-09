import pandas as pd
import glob
import os

# Set the path to your dataset folder (use os.getcwd() if in same folder)
path = os.getcwd()

# Find all CSV files matching 'TG*.csv'
all_files = glob.glob(os.path.join(path, "TG*.csv"))

# Create an empty list to hold DataFrames
df_list = []

for file in all_files:
    df = pd.read_csv(file)
    # Optional: Add a column indicating source filename
    base_name = os.path.splitext(os.path.basename(file))[0]
    df['file_name'] = base_name
    df_list.append(df)

# Concatenate all DataFrames into one
data = pd.concat(df_list, ignore_index=True)

print(f"Combined dataset shape: {data.shape}")
print(data.head())





# Show all columns
print(data.columns)

# 1. Drop columns you do NOT need (edit this list as desired)
# For example: drop Benzene, Toluene, Xylene, if not required
columns_to_drop = ['Benzene (ug/m3)', 'Toluene (ug/m3)', 'Xylene (ug/m3)']
data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

# 2. Convert date columns to datetime
data['From Date'] = pd.to_datetime(data['From Date'], errors='coerce')
data['To Date'] = pd.to_datetime(data['To Date'], errors='coerce')

# 3. Handle missing values
# Remove rows where key features are missing (e.g., PM2.5, PM10, NO2)
key_features = ['PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'NO2 (ug/m3)', 'SO2 (ug/m3)', 'CO (mg/m3)', 'Ozone (ug/m3)']
for col in key_features:
    if col in data.columns:
        data = data[data[col].notnull()]

# 4. Fill remaining missing values (optional)
data = data.fillna(method='ffill')  # Forward fill

# 5. Reset index after filtering
data = data.reset_index(drop=True)

# 6. Check final clean shape and missing values
print(f"After cleaning, shape: {data.shape}")
print("Missing values per column:")
print(data.isnull().sum())
print(data.head())





# Select relevant columns for modeling
useful_columns = [
    'From Date', 'PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'NO2 (ug/m3)',
    'NOx (ppb)', 'SO2 (ug/m3)', 'CO (mg/m3)', 'Ozone (ug/m3)', 'NH3 (ug/m3)',
    'Temp (degree C)', 'RH (%)', 'WS (m/s)', 'RF (mm)', 'file_name'
]

# Keep only columns that exist in data
data = data[[col for col in useful_columns if col in data.columns]]

# Convert 'From Date' to datetime if not already
data['From Date'] = pd.to_datetime(data['From Date'], errors='coerce')

# Extract datetime features
data['year'] = data['From Date'].dt.year
data['month'] = data['From Date'].dt.month
data['day'] = data['From Date'].dt.day
data['hour'] = data['From Date'].dt.hour
data['weekday'] = data['From Date'].dt.weekday

# Reset index after slicing
data = data.reset_index(drop=True)

print(f"Dataset shape after feature selection: {data.shape}")
print(data.head())

# Save cleaned and feature-engineered dataset to CSV
data.to_csv('cleaned_air_quality_data.csv', index=False)
print("Cleaned dataset saved as 'cleaned_air_quality_data.csv'")



