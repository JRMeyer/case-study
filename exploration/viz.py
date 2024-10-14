import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
import numpy as np

def explore_table(df, table_name):
    print(f"\n--- Exploring table: {table_name} ---")
    print("\nDataset Info:")
    print(df.info())

    print("\nSample Data (first 5 rows):")
    print(df.head())

    print("\nSummary Statistics:")
    print(df.describe(include='all'))

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nData Types:")
    print(df.dtypes)

    print("\nUnique Values in Each Column:")
    for column in tqdm(df.columns, desc="Counting unique values"):
        print(f"{column}: {df[column].nunique()}")

def safe_plot(df, column, table_name, plot_func):
    try:
        plt.figure(figsize=(10, 6))
        plot_func(df, column, table_name)
        plt.close()
    except Exception as e:
        print(f"Error plotting {column}: {str(e)}")
        plt.close()

def plot_histogram(df, column, table_name):
    sample = df[column].dropna().sample(n=min(1000, len(df[column])))
    sns.histplot(data=sample, kde=True)
    plt.title(f'Distribution of {column} - {table_name} (Sample)')
    plt.show()

def plot_countplot(df, column, table_name):
    value_counts = df[column].value_counts()
    sns.barplot(x=value_counts.index, y=value_counts.values)
    plt.title(f'Distribution of {column} - {table_name}')
    plt.xticks(rotation=45)
    plt.show()

def visualize_table(df, table_name):
    print(f"\n--- Visualizing table: {table_name} ---")
    
    # Correlation heatmap for numerical columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(num_cols) > 1:
        print("Generating correlation heatmap...")
        corr = df[num_cols].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=False, cmap='coolwarm')
        plt.title(f'Correlation Heatmap - {table_name}')
        plt.tight_layout()
        plt.show()
        plt.close()

    # Distribution of numerical features
    for column in tqdm(num_cols, desc="Plotting numerical distributions"):
        safe_plot(df, column, table_name, plot_histogram)

    # Bar plots for categorical features
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for column in tqdm(cat_cols, desc="Plotting categorical distributions"):
        if df[column].nunique() < 10:
            safe_plot(df, column, table_name, plot_countplot)

def main(db_file):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)

    # Get the list of tables in the database
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    print("Tables in the database:")
    for table in tables:
        print(table[0])

    # Explore each table
    for table in tqdm(tables, desc="Processing tables"):
        table_name = table[0]
        print(f"\nLoading table: {table_name}")
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        
        explore_table(df, table_name)
        visualize_table(df, table_name)

    # Close the database connection
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore and visualize data from a SQLite database.")
    parser.add_argument("db_file", help="Path to the SQLite database file")
    args = parser.parse_args()

    main(args.db_file)
