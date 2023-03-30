import os
import pandas as pd

# Set the directory containing the XLS files
directory = "C:\\Users\\joshu\\Project-Ajax\\Josh Dataset\\Jumping\\"

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.xls'):
        # Load the Excel file into a pandas DataFrame
        xls_file = os.path.join(directory, filename)
        df = pd.read_excel(xls_file)

        # Create a new filename for the CSV file
        csv_file = os.path.join(directory, f'{os.path.splitext(filename)[0]}.csv')

        # Save the DataFrame to a CSV file
        df.to_csv(csv_file, index=False)
