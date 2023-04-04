import os
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Step 1: Read from csv files
def read_csv_files(folder_path):
    """
    Extracts data from a given folder path.
    
    Args:
    - folder_path: a string that contains the folders "walking" and "running" each containing compatible csv files,
    
    Returns:
    - dataframe_collection: a Dataframe collection with key for it's fileanme and a value containing raw data formated as Dataframes
    """
    print("Converting files to data frames from " + folder_path + " ...")
    dataframe_collection = {}

    # go through each activty, designate state
    for activity, state in [('walking', 1), ('jumping', 2)]:

        # indicate folder
        activity_path = folder_path + '/' + activity

        # gove through each file and check if they are csv
        for file_name in os.listdir(activity_path):
            #print(file_name)
            if file_name.endswith('.csv'):
                file_path = activity_path + '/' + file_name
                df = pd.read_csv(file_path, header="infer")
                # add additional header
                df['state'] = state

                dataframe_collection[file_name]=pd.DataFrame(df)
    return dataframe_collection

# Step 2: For each dataframe extract rolling window (and features?)
# Feature Extraction here probably?

# NOTE: Apparently we cannot store a rolling window object into hdf5 BUT we can store the "features" of the rolling window instead into that file
# TO DO: Feature extraction guy, make it so we add each rolling window feature into rolling_window_collection
def extract_rolling_window(data_dict):
    """
    Extracts rolling window features from a dictionary of DataFrames.
    
    Args:
    - data_dict (dict): a dictionary of DataFrames, where the keys are the names of the DataFrames
    
    Returns:
    - features_df (DataFrame): a DataFrame containing the extracted features
    """
    rolling_window_collection = {}
    for key in data_dict.keys():
        df = data_dict[key]
        df = df.astype('float64')
        rolling_window = df.rolling(window=5).sum()
        rolling_window_collection[key + "_window"] = rolling_window 
#Jacintha
#Josh
        # TO DO: Figure out how to extract features here

        # feature_dict = {
        #     'max': rolling_window.max(),
        #     'min': rolling_window.min(),
        #     'mean': rolling_window.mean(),
        #     'median': rolling_window.median(),
        #     'range': rolling_window.apply(lambda x: x.max() - x.min()),
        #     'std': rolling_window.std(),
        #     'var': rolling_window.var(),
        #     'kurt': rolling_window.kurt(),
        #     'skew': rolling_window.skew(),
        #     # Add additional features as needed
        # }
    return rolling_window_collection

# Stage 3: sort data frames into groups
def write_dataframes_to_hdf5_group(dataframes_dict, file_path, group):
    print("Storing original files into HDF5...")
    with h5py.File(file_path, 'a') as f:
        for key, value in dataframes_dict.items():
            group.create_dataset(key, data = value)
            
def write_windows_to_hdf5_group(dataframes_dict, file_path, group):
    print("Storing rolling windows into HDF5...")
    with h5py.File(file_path, 'w') as f:
        for key, value in dataframes_dict.items():
            rolling_window = value
            dataset = group.create_dataset(key, shape=dataframes_dict.shape, dtype='f')
            dataset.write(rolling_window.to_numpy())

# Stage ?: preprocessing
def preprocess():
    print("Preprocessing...")
    # cut off low-freq noise


# Main Function
def main(): 
    folder_paths = [
        'C:\\Users\\joshu\\Project-Ajax\\Harrison Dataset',
        'C:\\Users\\joshu\\Project-Ajax\\Josh Dataset', 
        'C:\\Users\\joshu\\Project-Ajax\\Jacintha Dataset']

    dataMem1 = {}
    dataMem2 = {}
    dataMem3 = {}

    # dictionary containing data frames of raw data
    dataMem1 = read_csv_files(folder_paths[0])
    dataMem2 = read_csv_files(folder_paths[1])
    dataMem3 = read_csv_files(folder_paths[2])

    dataMem1Window = extract_rolling_window(dataMem1)
    dataMem2Window = extract_rolling_window(dataMem2)
    dataMem3Window = extract_rolling_window(dataMem3)

    # TO DO: Add dictionary for windows of data frames

    output_file = 'data.h5'
    output = h5py.File(output_file, 'w')
    mem1_group = output.create_group("Harrison Dataset")
    mem2_group = output.create_group("Josh Dataset")
    mem3_group = output.create_group("Jacintha Dataset")
    window_group = output.create_group("Window")

    # TO DO: Figure out how to get rolling window into hdf5 
    # CURR ERROR: Object dtype dtype('O') has no native HDF5 equivalent

    #window_group = output.create_group("Windows") 
    write_dataframes_to_hdf5_group(dataMem1, output_file, mem1_group)
    write_dataframes_to_hdf5_group(dataMem2, output_file, mem2_group)
    write_dataframes_to_hdf5_group(dataMem3, output_file, mem3_group)

    write_dataframes_to_hdf5_group(dataMem1Window, output_file, window_group)

    # TO DO: Add Preprocessing

    # TO DO: Add Feature Extraction
    # Prints HDF5 structure
    with h5py.File(output_file, 'r') as hdf:
        items = list(hdf.items())
        print(items)
        mem1_items=list(mem1_group.items())
        mem2_items=list(mem2_group.items())
        mem3_items=list(mem3_group.items())
        window_items = list(window_group.items())
        print("\n#######     Harrison    #######")
        for item in mem1_items:
            print(item)
        print("\n#######       Josh      #######")
        for item in mem2_items:
            print(item)
        print("\n#######     Jacintha    #######")
        for item in mem3_items:
            print(item)

        print("\n#######     Windows    #######")
        for item in window_items:
            print(item)

if __name__ == "__main__":
    main()

    #Harrison