import os
import shutil
import matplotlib.pyplot as plt

# Set the directory containing the folders with CSV files
directory = "C:\\Users\\joshu\\Project-Ajax\\Harrison Dataset\\Jumping"

# Loop through all folders in the directory
for foldername in os.listdir(directory):
    folderpath = os.path.join(directory, foldername)
    if os.path.isdir(folderpath):
        # Loop through all files in the folder
        for filename in os.listdir(folderpath):
            if filename.endswith('.csv'):
                # Create a new filename for the CSV file
                new_filename = f'{foldername}.csv'
                old_filepath = os.path.join(folderpath, filename)
                new_filepath = os.path.join(directory, new_filename)

                # Move the CSV file to the directory and rename it
                shutil.move(old_filepath, new_filepath)