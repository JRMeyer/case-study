#!/bin/bash

python3 -m venv venv

source venv/bin/activate

pip install -U pip

pip install -r requirements.txt

python pre-processing/process_data.py --database data/raw/CSO_case_study_data.db --train_output data/processed/train.csv --test_output data/processed/test.csv

python train/lstm.py --train_input data/processed/train.csv --test_input data/processed/test.csv
