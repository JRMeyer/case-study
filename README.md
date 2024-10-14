# Diabetes Prediction Project

This project implements a machine learning model to predict the onset of diabetes using historical laboratory test data. It includes data preprocessing, model training using LSTMs and HMMs, and evaluation of the models' performance.

## Project Structure

```
.
├── Dockerfile
├── data
│   ├── processed
│   │   └── README.txt
│   └── raw
│       └── README.txt
├── exploration
│   ├── convert_db_to_csvs.py
│   ├── generate_report.py
│   └── viz.py
├── pre-processing
│   └── process_data.py
├── requirements.txt
├── run.sh
└── train
    ├── hmm.py
    └── lstm.py
```

- `run.sh`: Main script to run the entire pipeline
- `pre-processing/process_data.py`: Script for data preprocessing
- `train/lstm.py`: Script for training and evaluating the LSTM model
- `train/hmm.py`: Script for training and evaluating the HMM model
- `Dockerfile`: Configuration for creating a Docker container for the project
- `requirements.txt`: List of Python dependencies
- `data/`: Directory containing raw and processed data
  - `raw/`: Contains the raw SQLite database and a README
  - `processed/`: Directory for storing processed data files and a README
- `exploration/`: Directory containing scripts for data exploration and visualization
  - `convert_db_to_csvs.py`: Script to convert SQLite database to CSV files
  - `generate_report.py`: Script to generate a report on the data
  - `viz.py`: Script for data visualization

## Prerequisites

- Python 3.9+
- Docker (optional, for containerized execution)

## Setup and Execution

### Using Docker

1. Build the Docker image:
   ```
   docker build -t diabetes-prediction .
   ```

2. Run the container:
   ```
   docker run diabetes-prediction
   ```

### Manual Setup

1. Create a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the pipeline:
   ```
   ./run.sh
   ```

## Data

The project expects a SQLite database file in the `data/raw/` directory. This database should contain two tables:
- `lab_test_results`: Laboratory test results for patients
- `patient_info`: Demographic information for patients

## Pipeline Steps

1. **Data Preprocessing** (`process_data.py`):
   - Loads data from the SQLite database
   - Merges lab results with patient information
   - Handles missing values and encodes categorical features
   - Splits data into training and testing sets

2. **Model Training**:
   - LSTM (`lstm.py`): Implements an LSTM neural network for sequence classification
   - HMM (`hmm.py`): Implements a Hidden Markov Model for sequence classification
   - Trains the models on preprocessed data
   - Evaluates the models' performance on the test set

3. **Data Exploration** (optional):
   - `convert_db_to_csvs.py`: Converts the SQLite database to CSV files for easier exploration
   - `generate_report.py`: Generates a report summarizing the data
   - `viz.py`: Creates visualizations of the data

## Output

The pipeline generates several outputs:
1. Processed CSV files for training and testing data (in `data/processed/`)
2. Evaluation metrics including confusion matrix and classification report for both LSTM and HMM models
3. Data exploration reports and visualizations (when running the exploration scripts)

## Customization

You can modify the hyperparameters of the models by editing the arguments in `run.sh` or by passing them directly to `lstm.py` or `hmm.py` when running manually.
