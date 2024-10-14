import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ArgParse setup
parser = argparse.ArgumentParser(description='Train LSTM model for diabetes prediction')
parser.add_argument('--train_input', type=str, default='train_data.csv', help='Path to the input train CSV file')
parser.add_argument('--test_input', type=str, default='test_data.csv', help='Path to the input test CSV file')
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
print("\nLoading processed data from CSV files...")
train_data = pd.read_csv(args.train_input)
test_data = pd.read_csv(args.test_input)
print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Define feature columns
exclude_columns = ['patient_id', 'report_date_utc', 'diabetes_label']
feature_columns = [col for col in train_data.columns if col not in exclude_columns]
print(f"\nFeature columns ({len(feature_columns)}): {feature_columns}")

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
            label = patient_data['diabetes_label'].iloc[-1]
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

# Create the datasets
print("\nInitializing datasets...")
train_dataset = PatientDataset(train_data, feature_columns)
test_dataset = PatientDataset(test_data, feature_columns)

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
