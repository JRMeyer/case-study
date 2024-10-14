import argparse
import pandas as pd
import numpy as np
import sqlite3
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ArgParse setup
parser = argparse.ArgumentParser(description='Process data for diabetes prediction model')
parser.add_argument('--database', type=str, default='hospital_data.db', help='Path to SQLite database')
parser.add_argument('--train_output', type=str, default='train_data.csv', help='Path to output train CSV file')
parser.add_argument('--test_output', type=str, default='test_data.csv', help='Path to output test CSV file')
args = parser.parse_args()

print("Starting data processing for diabetes prediction model...")

print("\nLoading data from SQLite database...")
# Connect to the SQLite database and load data
conn = sqlite3.connect(args.database)
lab_results = pd.read_sql("SELECT * FROM lab_test_results", conn)
patient_info = pd.read_sql("SELECT * FROM patient_info", conn)

print("Merging and preprocessing data...")
# Merge the dataframes
merged_data = lab_results.merge(patient_info, on='patient_id')

# Convert 'report_date_utc' to datetime, handling errors
merged_data['report_date_utc'] = pd.to_datetime(merged_data['report_date_utc'], errors='coerce')

# Drop rows with invalid dates
merged_data = merged_data.dropna(subset=['report_date_utc'])

# Sort the data
merged_data = merged_data.sort_values(['patient_id', 'report_date_utc']).reset_index(drop=True)

print("Categorizing diabetes status over time...")
tqdm.pandas()

def categorize_diabetes_over_time(df_patient):
    last_known_status = None
    statuses = []
    for idx, row in df_patient.iterrows():
        if not pd.isnull(row['hba1c']) or not pd.isnull(row['fasting_blood_glucose']):
            # Use current lab results to determine status
            if row['hba1c'] >= 6.5 or row['fasting_blood_glucose'] >= 7.0:
                status = "detected_diabetes"
            elif (5.7 <= row['hba1c'] < 6.5) or (5.6 <= row['fasting_blood_glucose'] < 7.0):
                status = "detected_pre-diabetes"
            else:
                status = "undetected"
            last_known_status = status
        else:
            # No lab results; use last known status
            if last_known_status:
                status = last_known_status
            else:
                status = "unknown"
        statuses.append(status)
    df_patient['diabetes'] = statuses
    return df_patient

# Apply the function to each patient group
merged_data = merged_data.groupby('patient_id', group_keys=False).progress_apply(categorize_diabetes_over_time)

print(f"Merged data shape: {merged_data.shape}")
print(f"Unique patients: {merged_data['patient_id'].nunique()}")
print(f"Date range: {merged_data['report_date_utc'].min()} to {merged_data['report_date_utc'].max()}")
print(f"Diabetes status distribution:\n{merged_data['diabetes'].value_counts(dropna=False)}")

# Define features
continuous_features = [
    'glomerular_filtration_rate', 'alanine_aminotransferase', 'aspartate_aminotransferase',
    'CRP', 'cholesterol', 'creatinine', 'triglycerids', 'erythrocytes', 'leukocytes',
    'hba1c', 'fasting_blood_glucose', 'hemoglobin'
]

categorical_features = ['sex', 'hepatitis', 'high_blood_pressure', 'hyperlipidemia']

print("\nPreprocessing data...")

# Impute missing values for continuous features
print("Imputing and scaling continuous features...")
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

X_continuous = merged_data[continuous_features]
X_continuous_imputed = imputer.fit_transform(X_continuous)
X_continuous_scaled = scaler.fit_transform(X_continuous_imputed)

# Create a new dataframe with processed continuous features
processed_data = pd.DataFrame(X_continuous_scaled, columns=continuous_features)

# Impute missing values for categorical features
print("Imputing missing values for categorical features...")
imputer_cat = SimpleImputer(strategy='most_frequent')
X_categorical = merged_data[categorical_features]
X_categorical_imputed = pd.DataFrame(imputer_cat.fit_transform(X_categorical), columns=categorical_features)

# Encode categorical features
print("Encoding categorical features...")
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_cat = encoder.fit_transform(X_categorical_imputed)
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_features))

# Add encoded categorical features to processed_data
processed_data = pd.concat([processed_data.reset_index(drop=True), encoded_cat_df.reset_index(drop=True)], axis=1)

# Add other necessary columns
processed_data['patient_id'] = merged_data['patient_id'].values
processed_data['report_date_utc'] = merged_data['report_date_utc'].values
processed_data['birth_date'] = merged_data['birth_date'].values

# Convert 'birth_date' to datetime, handling errors
processed_data['birth_date'] = pd.to_datetime(processed_data['birth_date'], errors='coerce')

# Calculate age at the time of each measurement
processed_data['age'] = (processed_data['report_date_utc'] - processed_data['birth_date']).dt.days / 365.25

# Drop birth_date as we now have age
processed_data = processed_data.drop(columns=['birth_date'])

# Map 'diabetes' column to binary labels
print("\nMapping diabetes status to binary labels...")
label_mapping = {
    'detected_diabetes': 1,
    'detected_pre-diabetes': 1,
    'undetected': 0,
    'unknown': 0  # Map 'unknown' to 0 for simplicity
}
processed_data['diabetes_label'] = merged_data['diabetes'].map(label_mapping)

# Handle missing values if any
print("\nHandling missing values...")
processed_data = processed_data.fillna(0)

# Exclude patients who only have 'unknown' diabetes status
print("\nFiltering out patients who only have 'unknown' diabetes status...")
valid_patients = merged_data[merged_data['diabetes'] != 'unknown']['patient_id'].unique()
initial_patient_count = processed_data['patient_id'].nunique()
processed_data = processed_data[processed_data['patient_id'].isin(valid_patients)].reset_index(drop=True)
filtered_patient_count = processed_data['patient_id'].nunique()
print(f"Patients before filtering: {initial_patient_count}")
print(f"Patients after filtering: {filtered_patient_count}")

# Ensure 'patient_id' and 'report_date_utc' are of the correct type
processed_data['patient_id'] = processed_data['patient_id'].astype(int)
processed_data['report_date_utc'] = pd.to_datetime(processed_data['report_date_utc'])

# Sort data by 'patient_id' and 'report_date_utc'
processed_data = processed_data.sort_values(['patient_id', 'report_date_utc']).reset_index(drop=True)
print("\nData sorted by 'patient_id' and 'report_date_utc'.")

# Create a stratification variable based on the last diabetes label for each patient
stratify_variable = processed_data.groupby('patient_id')['diabetes_label'].last()

# Split the data into train and test sets
print("\nSplitting data into train and test sets...")
train_patients, test_patients = train_test_split(
    stratify_variable.index,
    test_size=0.2,
    random_state=42,
    stratify=stratify_variable
)

train_data = processed_data[processed_data['patient_id'].isin(train_patients)]
test_data = processed_data[processed_data['patient_id'].isin(test_patients)]

print(f"\nSaving processed train data to {args.train_output}")
train_data.to_csv(args.train_output, index=False)

print(f"Saving processed test data to {args.test_output}")
test_data.to_csv(args.test_output, index=False)

print("Data processing completed.")
conn.close()
