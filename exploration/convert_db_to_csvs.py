import sqlite3
import csv
import os
import argparse

def convert_sqlite_to_csv(db_file, output_dir):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Get the list of tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through each table and create a CSV file
    for table in tables:
        table_name = table[0]
        
        # Query all data from the table
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()

        # Get the column names
        column_names = [description[0] for description in cursor.description]

        # Write data to CSV file
        csv_file_path = os.path.join(output_dir, f"{table_name}.csv")
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(column_names)  # Write header
            csv_writer.writerows(rows)  # Write data

        print(f"Table '{table_name}' exported to {csv_file_path}")

    # Close the database connection
    conn.close()

    print("Conversion completed successfully.")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert SQLite database to CSV files")
    parser.add_argument("db_file", help="Path to the SQLite database file")
    parser.add_argument("output_dir", help="Directory to save the CSV files")

    # Parse arguments
    args = parser.parse_args()

    # Call the conversion function with parsed arguments
    convert_sqlite_to_csv(args.db_file, args.output_dir)

if __name__ == "__main__":
    main()
