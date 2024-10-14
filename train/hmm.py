import argparse
import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ArgParse setup
parser = argparse.ArgumentParser(description='Train HMM model for diabetes prediction')
parser.add_argument('--input', type=str, default='processed_data.csv', help='Path to the input CSV file')
args = parser.parse_args()

print("Starting diabetes prediction model training...")

# Load processed data from CSV
print("\nLoading processed data from CSV file...")
data = pd.read_csv(args.input)

print("Preprocessing data...")
# Map 'diabetes' column to binary labels
label_mapping = {
    'detected_diabetes': 1,
    'detected_pre-diabetes': 1,
    'undetected': 0,
    'unknown': 0  # Map 'unknown' to 0 for simplicity
}
data['diabetes_label'] = data['diabetes'].map(label_mapping)

# Check for missing values in the data
print("Checking for missing values in the data...")
missing_values = data.isnull().sum()
print(missing_values)

if missing_values.any():
    print("\nHandling missing values...")
    # Identify columns with missing values
    missing_columns = missing_values[missing_values > 0].index.tolist()
    print(f"Columns with missing values: {missing_columns}")

    # Fill missing values in feature columns with zero or another appropriate value
    exclude_columns = ['patient_id', 'report_date_utc', 'diabetes', 'diabetes_label']
    feature_columns = [col for col in data.columns if col not in exclude_columns]
    data[feature_columns] = data[feature_columns].fillna(0)

    # Handle missing values in 'age' separately if needed
    if 'age' in missing_columns:
        print("Handling missing values in 'age' column...")
        data['age'] = data['age'].fillna(data['age'].mean())

    # Handle missing values in 'diabetes_label' if any
    if 'diabetes_label' in missing_columns:
        print("Missing values found in 'diabetes_label'. Dropping these rows.")
        data = data.dropna(subset=['diabetes_label'])

    # Ensure 'report_date_utc' is datetime and handle missing dates
    data['report_date_utc'] = pd.to_datetime(data['report_date_utc'], errors='coerce')

    # Drop rows with missing 'report_date_utc' if necessary
    if data['report_date_utc'].isnull().any():
        print("Missing or invalid 'report_date_utc' detected. Dropping these rows.")
        data = data.dropna(subset=['report_date_utc'])

    # Re-check for missing values
    missing_values_after = data.isnull().sum()
    if missing_values_after.any():
        print("Warning: Missing values still present after handling:")
        print(missing_values_after)
else:
    print("No missing values detected.")

# Sort data by 'patient_id' and 'report_date_utc'
data = data.sort_values(['patient_id', 'report_date_utc']).reset_index(drop=True)

# Define feature columns
exclude_columns = ['patient_id', 'report_date_utc', 'diabetes', 'diabetes_label']
feature_columns = [col for col in data.columns if col not in exclude_columns]

print(f"Feature columns: {feature_columns}")

# Create sequences for each patient
print("\nCreating sequences for each patient...")
sequences = []
lengths = []
labels = []
patient_ids = data['patient_id'].unique()

for pid in tqdm(patient_ids, desc="Processing patients"):
    patient_data = data[data['patient_id'] == pid]
    sequence = patient_data[feature_columns].values
    label = patient_data['diabetes_label'].iloc[-1]  # Use the last known diabetes status
    sequences.append(sequence)
    lengths.append(len(sequence))
    labels.append(label)

# Combine all sequences into one array for HMM
X = np.vstack(sequences)
lengths = np.array(lengths)

# Split data into train and test sets
print("\nSplitting data into train and test sets...")
X_train_list, X_test_list, y_train, y_test = train_test_split(
    sequences, labels, test_size=0.2, random_state=42, stratify=labels
)

# Prepare training data
X_train = np.vstack(X_train_list)
lengths_train = [len(seq) for seq in X_train_list]

# Prepare testing data
X_test = np.vstack(X_test_list)
lengths_test = [len(seq) for seq in X_test_list]

# Initialize and train HMM models for each class
print("\nTraining HMM models...")
n_components = 3  # Number of hidden states, you can adjust this

# Separate sequences by class
X_train_class0_list = [seq for seq, label in zip(X_train_list, y_train) if label == 0]
lengths_class0 = [len(seq) for seq in X_train_class0_list]
X_train_class0 = np.vstack(X_train_class0_list) if X_train_class0_list else np.empty((0, X_train.shape[1]))

X_train_class1_list = [seq for seq, label in zip(X_train_list, y_train) if label == 1]
lengths_class1 = [len(seq) for seq in X_train_class1_list]
X_train_class1 = np.vstack(X_train_class1_list) if X_train_class1_list else np.empty((0, X_train.shape[1]))

# Check if we have data for both classes
if X_train_class0.shape[0] == 0 or X_train_class1.shape[0] == 0:
    print("Insufficient data for one of the classes after splitting. Please check the class distribution.")
    exit()

# Create and train HMMs for each class
model_0 = hmm.GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=1000)
model_0.fit(X_train_class0, lengths_class0)

model_1 = hmm.GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=1000)
model_1.fit(X_train_class1, lengths_class1)

# Evaluation
print("\nEvaluating on test data...")
y_pred = []
for seq in X_test_list:
    # Compute log likelihood for each model
    log_likelihood_0 = model_0.score(seq)
    log_likelihood_1 = model_1.score(seq)
    # Predict class with higher likelihood
    if log_likelihood_1 > log_likelihood_0:
        y_pred.append(1)
    else:
        y_pred.append(0)

# Calculate and print metrics
print('\nEvaluation Results:')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

print("\nScript execution completed.")
