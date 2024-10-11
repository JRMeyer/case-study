import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

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
    for column in df.columns:
        print(f"{column}: {df[column].nunique()}")

def visualize_table(df, table_name):
    print(f"\n--- Visualizing table: {table_name} ---")
    
    # Correlation heatmap for numerical columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(num_cols) > 1:
        plt.figure(figsize=(12, 10))
        sns.heatmap(df[num_cols].corr(), annot=False, cmap='coolwarm')
        plt.title(f'Correlation Heatmap - {table_name}')
        plt.tight_layout()
        plt.show()

    # Distribution of numerical features
    for column in num_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=column, kde=True)
        plt.title(f'Distribution of {column} - {table_name}')
        plt.show()

    # Bar plots for categorical features
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for column in cat_cols:
        if df[column].nunique() < 10:  # Only plot if there are fewer than 10 unique values
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df, x=column)
            plt.title(f'Distribution of {column} - {table_name}')
            plt.xticks(rotation=45)
            plt.show()

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
    for table in tables:
        table_name = table[0]
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
