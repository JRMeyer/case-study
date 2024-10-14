import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ArgParse setup
parser = argparse.ArgumentParser(description='Train LSTM model for diabetes prediction')
parser.add_argument('--input', type=str, default='data/FORMATTED.csv', help='Path to the input CSV file')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size of LSTM')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
args = parser.parse_args()

print("Starting diabetes prediction model training...")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load processed data from CSV
print("\nLoading processed data from CSV file...")
data = pd.read_csv(args.input)
print(f"Data shape: {data.shape}")

print("\nPreprocessing data...")
# Map 'diabetes' column to binary labels
label_mapping = {
    'detected_diabetes': 1,
    'detected_pre-diabetes': 1,
    'undetected': 0,
    'unknown': 0  # Map 'unknown' to 0 for simplicity
}
data['diabetes_label'] = data['diabetes'].map(label_mapping)
print("Diabetes label mapping applied.")
print(f"Unique diabetes labels: {data['diabetes_label'].unique()}")

# Handle missing values if any
print("\nChecking for missing values in the dataset...")
missing_values = data.isnull().sum()
print(missing_values)

if missing_values.any():
    print("\nHandling missing values...")
    # Fill missing values in feature columns with zero or mean
    exclude_columns = ['patient_id', 'report_date_utc', 'diabetes', 'diabetes_label']
    feature_columns = [col for col in data.columns if col not in exclude_columns]
    data[feature_columns] = data[feature_columns].fillna(0)
    print("Missing values in feature columns filled with zeros.")
    
    # Handle missing values in 'diabetes_label' if any
    if 'diabetes_label' in missing_values[missing_values > 0].index:
        data = data.dropna(subset=['diabetes_label'])
        print("Rows with missing 'diabetes_label' dropped.")
    
    # Convert 'report_date_utc' to datetime
    data['report_date_utc'] = pd.to_datetime(data['report_date_utc'], errors='coerce')
    # Drop rows with missing 'report_date_utc'
    data = data.dropna(subset=['report_date_utc'])
    print("Rows with invalid 'report_date_utc' dropped.")
else:
    print("No missing values detected.")

# Exclude patients who only have 'unknown' diabetes status
print("\nFiltering out patients who only have 'unknown' diabetes status...")
valid_patients = data[data['diabetes'] != 'unknown']['patient_id'].unique()
initial_patient_count = data['patient_id'].nunique()
data = data[data['patient_id'].isin(valid_patients)].reset_index(drop=True)
filtered_patient_count = data['patient_id'].nunique()
print(f"Patients before filtering: {initial_patient_count}")
print(f"Patients after filtering: {filtered_patient_count}")

# Sort data by 'patient_id' and 'report_date_utc'
data = data.sort_values(['patient_id', 'report_date_utc']).reset_index(drop=True)
print("\nData sorted by 'patient_id' and 'report_date_utc'.")

# Define feature columns
exclude_columns = ['patient_id', 'report_date_utc', 'diabetes', 'diabetes_label']
feature_columns = [col for col in data.columns if col not in exclude_columns]
print(f"\nFeature columns ({len(feature_columns)}): {feature_columns}")

# Create sequences for each patient
print("\nCreating sequences for each patient...")

class PatientDataset(Dataset):
    def __init__(self, data, feature_columns):
        self.data = data
        self.feature_columns = feature_columns
        self.patient_ids = data['patient_id'].unique()
        print(f"Total patients: {len(self.patient_ids)}")
        self.sequences = []
        self.labels = []
        self.lengths = []
        self._create_sequences()

    def _create_sequences(self):
        for pid in tqdm(self.patient_ids, desc="Processing patients"):
            patient_data = self.data[self.data['patient_id'] == pid]
            sequence = patient_data[self.feature_columns].values
            label = patient_data['diabetes_label'].iloc[-1]  # Use the last known diabetes status
            self.sequences.append(torch.tensor(sequence, dtype=torch.float32))
            self.labels.append(torch.tensor(label, dtype=torch.float32))
            self.lengths.append(len(sequence))
        print(f"Total sequences created: {len(self.sequences)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.lengths[idx]

def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True)
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return sequences_padded, labels, lengths

# Create the dataset
print("\nInitializing dataset...")
dataset = PatientDataset(data, feature_columns)

# Split dataset into training and testing
print("\nSplitting data into train and test sets...")
train_indices, test_indices = train_test_split(
    range(len(dataset)),
    test_size=0.2,
    random_state=42,
    stratify=dataset.labels
)
print(f"Total sequences: {len(dataset)}")
print(f"Training sequences: {len(train_indices)}")
print(f"Testing sequences: {len(test_indices)}")

train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
print("\nData loaders created.")

# Define the LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()  # For binary classification

    def forward(self, x, lengths):
        # Pack the sequences
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(packed_input)
        # Use the last hidden state
        out = self.fc(hidden[-1])
        out = self.sigmoid(out)
        return out.squeeze()

print("\nInitializing model...")
input_size = len(feature_columns)
model = LSTMClassifier(input_size, args.hidden_size).to(device)
print(f"Model initialized with input size {input_size} and hidden size {args.hidden_size}.")
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
print("Loss function and optimizer set up.")

# Training loop
print(f"\nStarting training for {args.epochs} epochs...")
for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for sequences, labels, lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
        sequences = sequences.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        optimizer.zero_grad()
        outputs = model(sequences, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}')

print("\nTraining completed.")

# Evaluation
print("\nStarting evaluation on test data...")
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for sequences, labels, lengths in tqdm(test_loader, desc="Evaluating"):
        sequences = sequences.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        outputs = model(sequences, lengths)
        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Binarize predictions
predictions = (np.array(all_preds) > 0.5).astype(int)
true_labels = np.array(all_labels).astype(int)

# Calculate metrics
print('\nEvaluation Results:')
print('Confusion Matrix:')
print(confusion_matrix(true_labels, predictions))
print('\nClassification Report:')
print(classification_report(true_labels, predictions, target_names=['No Diabetes', 'Diabetes']))

print("\nScript execution completed.")
