import sqlite3
import pandas as pd
import numpy as np
import argparse
from scipy import stats
from tqdm import tqdm

def generate_table_summary(df, table_name):
    report = f"\n\n{'='*50}\n"
    report += f"Table: {table_name}\n"
    report += f"{'='*50}\n\n"

    # Basic info
    report += f"Number of rows: {len(df)}\n"
    report += f"Number of columns: {len(df.columns)}\n\n"

    # Sparsity and missing value information
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    sparsity = missing_cells / total_cells
    report += f"Overall sparsity: {sparsity:.2%}\n"
    report += f"Total missing values: {missing_cells} ({missing_cells/total_cells:.2%} of all cells)\n\n"

    # Column details
    report += "Column Details:\n"
    for col in tqdm(df.columns, desc=f"Analyzing columns in {table_name}", leave=False):
        report += f"\n- {col}:\n"
        report += f"  Data type: {df[col].dtype}\n"
        report += f"  Number of unique values: {df[col].nunique()}\n"
        null_count = df[col].isnull().sum()
        null_percentage = null_count / len(df) * 100
        report += f"  Number of null values: {null_count} ({null_percentage:.2f}%)\n"
        
        if df[col].dtype in ['int64', 'float64']:
            report += f"  Mean: {df[col].mean():.2f}\n"
            report += f"  Median: {df[col].median():.2f}\n"
            report += f"  Standard deviation: {df[col].std():.2f}\n"
            report += f"  Min: {df[col].min()}\n"
            report += f"  Max: {df[col].max()}\n"
            
            # Add information about zero values for numeric columns
            zero_count = (df[col] == 0).sum()
            zero_percentage = zero_count / len(df) * 100
            report += f"  Number of zero values: {zero_count} ({zero_percentage:.2f}%)\n"
        elif df[col].dtype == 'object':
            report += f"  Most common value: {df[col].mode().values[0]}\n"
            report += f"  Frequency of most common value: {df[col].value_counts().iloc[0]}\n"

    return report

def generate_correlation_summary(df):
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        report = "\nCorrelation Summary:\n"
        for col1 in tqdm(num_cols, desc="Calculating correlations", leave=False):
            for col2 in num_cols:
                if col1 != col2 and abs(corr.loc[col1, col2]) > 0.5:
                    report += f"- Strong correlation between {col1} and {col2}: {corr.loc[col1, col2]:.2f}\n"
        return report
    return "\nNo numerical columns to compute correlations.\n"

def generate_diabetes_specific_analysis(df):
    report = "\nDiabetes-Specific Analysis:\n"
    
    if 'diabetes' in df.columns:
        diabetes_col = df['diabetes']
        report += f"Diabetes column found.\n"
        report += f"- Number of diabetes cases: {diabetes_col.sum()}\n"
        report += f"- Percentage of diabetes cases: {(diabetes_col.sum() / len(diabetes_col)) * 100:.2f}%\n"
        
        # Analyze relationship with other columns
        for col in tqdm(df.columns, desc="Analyzing diabetes relationships", leave=False):
            if col != 'diabetes' and df[col].dtype in ['int64', 'float64']:
                # Handle missing values by dropping them for this analysis
                valid_data = df[[col, 'diabetes']].dropna()
                if len(valid_data) > 0:
                    t_stat, p_value = stats.ttest_ind(valid_data[valid_data['diabetes'] == 1][col], 
                                                      valid_data[valid_data['diabetes'] == 0][col])
                    if p_value < 0.05:
                        report += f"- Significant difference in {col} between diabetes and non-diabetes groups (p-value: {p_value:.4f})\n"
                else:
                    report += f"- Unable to analyze {col} due to insufficient valid data\n"
    else:
        report += "No column named 'diabetes' found in this table.\n"
    
    return report

def main(db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    full_report = "Comprehensive Database Report\n"
    full_report += "============================\n\n"

    for table in tqdm(tables, desc="Processing tables"):
        table_name = table[0]
        print(f"\nLoading table: {table_name}")
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        
        full_report += generate_table_summary(df, table_name)
        full_report += generate_correlation_summary(df)
        full_report += generate_diabetes_specific_analysis(df)

    conn.close()

    # Write report to file
    with open('database_report.txt', 'w') as f:
        f.write(full_report)

    print("Report generated and saved as 'database_report.txt'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a comprehensive report from a SQLite database.")
    parser.add_argument("db_file", help="Path to the SQLite database file")
    args = parser.parse_args()

    main(args.db_file)
