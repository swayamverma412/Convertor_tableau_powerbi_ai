import pandas as pd
import os
import graphviz
from concurrent.futures import ThreadPoolExecutor
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the directory containing data
directory = 'bike store data'

# Get a list of all csv files in the directory
csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

# Function to find the potential primary keys for a dataframe
def find_potential_primary_keys(df):
    return [column for column in df.columns if df[column].nunique() == len(df)]

# Function to validate the dataframe
def validate_dataframe(df, file):
    if df.empty:
        raise ValueError(f"{file} is empty.")
    if df.isnull().values.any():
        raise ValueError(f"{file} contains null values.")
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_string_dtype(df[column]):
            raise ValueError(f"{file} contains non-numeric and non-string data in column {column}.")
    return df

# Function to read CSV and process it
def read_and_process(file):
    file_path = os.path.join(directory, file)
    try:
        df = pd.read_csv(file_path)
        df = validate_dataframe(df, file)
        primary_keys = find_potential_primary_keys(df)
        return file, df, primary_keys, None
    except Exception as e:
        logging.error(f"Error reading {file}: {e}")
        return file, None, None, str(e)

# Use ThreadPoolExecutor to read files concurrently
with ThreadPoolExecutor() as executor:
    results = executor.map(read_and_process, csv_files)

# Dictionaries to store results
dataframes = {}
primary_keys = {}
errors = []

for file, df, keys, error in results:
    if error:
        errors.append((file, error))
    else:
        dataframes[file] = df
        primary_keys[file] = keys
        print(f"File: {file}")
        print(f"Potential Primary keys: {keys}\n")

# Log errors if any
if errors:
    logging.warning(f"Errors encountered in the following files: {errors}")

# Dictionary to store foreign keys
foreign_keys = {}

# Check if primary keys are foreign keys in other tables
for file, keys in primary_keys.items():
    for key in keys:
        for other_file, other_df in dataframes.items():
            if file != other_file and key in other_df.columns:
                if key not in foreign_keys:
                    foreign_keys[key] = []
                foreign_keys[key].append((file, other_file))

# Dictionary to store table types
table_types = {file: 'Dimension' for file in csv_files}

# Determine table types
for file, keys in primary_keys.items():
    if any(key in foreign_keys for key in keys):
        table_types[file] = 'Fact'

# Print table types
for file, table_type in table_types.items():
    print(f"Table: {file}")
    print(f"Type: {table_type}\n")

# Create a graph to represent relationships using Graphviz
dot = graphviz.Digraph(comment='ER Diagram')

# Add nodes for each table
for file in csv_files:
    label = f"{file}\n({table_types[file]})"
    dot.node(file, label=label, shape='box', style='filled', color='lightgray')

# Add edges for primary key to foreign key relationships
for key, tables in foreign_keys.items():
    for (source_file, target_file) in tables:
        dot.edge(source_file, target_file, label=key)

# Render the output
output_path = 'er_diagram'
dot.render(output_path, view=True)
