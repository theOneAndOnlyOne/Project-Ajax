import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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

        # go through each file and check if they are csv
        for file_name in os.listdir(activity_path):
            #print(file_name)
            if file_name.endswith('.csv'):
                file_path = activity_path + '/' + file_name
                df = pd.read_csv(file_path, header="infer")
                # add additional header
                df['state'] = state

                dataframe_collection[file_name]=pd.DataFrame(df)
                print(file_name)
    return dataframe_collection



current_dir = os.getcwd()

folder_paths = [
        os.path.join(current_dir, 'Harrison Dataset'),
        os.path.join(current_dir, 'Josh Dataset'),
        os.path.join(current_dir, 'Jacintha Dataset'),

    ]

dataMem1 = {}
dataMem2 = {}
dataMem3 = {}

# dictionary containing data frames of raw data
# original
dataMem1 = read_csv_files(folder_paths[0])
dataMem2 = read_csv_files(folder_paths[1])
dataMem3 = read_csv_files(folder_paths[2])

data_list = []
    
for d in [dataMem1, dataMem2, dataMem3]:
    for df in d.values():
        data_list.append(df)

            
raw_dataframe = pd.concat(data_list,ignore_index=True)

# Divide the dataframe based on 'state' values
state_1_dataframe = raw_dataframe[raw_dataframe.iloc[:,-1] == 1]
state_2_dataframe = raw_dataframe[raw_dataframe.iloc[:,-1] == 2]

# Histogram State 1
plt.figure()
sns.histplot(data=state_1_dataframe, x="Linear Acceleration x (m/s^2)", kde=True)
plt.title("Histogram of Linear Acceleration x (Walking)")
plt.show()

# Violin chart State 1
plt.figure()
sns.violinplot(data=state_1_dataframe[["Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)", "Linear Acceleration z (m/s^2)"]])
plt.title("Violin Chart of Linear Acceleration x, y, z (Walking)")
plt.show()

# Density chart State 1
plt.figure()
sns.kdeplot(data=state_1_dataframe, x="Linear Acceleration x (m/s^2)", label="Linear Acceleration x")
sns.kdeplot(data=state_1_dataframe, x="Linear Acceleration y (m/s^2)", label="Linear Acceleration y")
sns.kdeplot(data=state_1_dataframe, x="Linear Acceleration z (m/s^2)", label="Linear Acceleration z")
plt.title("Density Chart of Linear Acceleration x, y, z (Walking)")
plt.legend()
plt.show()


# Histogram State 1
plt.figure()
sns.histplot(data=state_2_dataframe, x="Linear Acceleration x (m/s^2)", kde=True)
plt.title("Histogram of Linear Acceleration x (Jumping)")
plt.show()

# Violin chart State 1
plt.figure()
sns.violinplot(data=state_2_dataframe[["Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)", "Linear Acceleration z (m/s^2)"]])
plt.title("Violin Chart of Linear Acceleration x, y, z (Jumping)")
plt.show()

# Density chart State 1
plt.figure()
sns.kdeplot(data=state_2_dataframe, x="Linear Acceleration x (m/s^2)", label="Linear Acceleration x")
sns.kdeplot(data=state_2_dataframe, x="Linear Acceleration y (m/s^2)", label="Linear Acceleration y")
sns.kdeplot(data=state_2_dataframe, x="Linear Acceleration z (m/s^2)", label="Linear Acceleration z")
plt.title("Density Chart of Linear Acceleration x, y, z (Jumping)")
plt.legend()
plt.show()

