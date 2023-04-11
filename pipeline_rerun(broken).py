import os
import sys
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA

def main():
    print("starting...")

    current_dir = os.getcwd()

    if os.path.exists(os.path.join(current_dir, 'Harrison Dataset')) and os.path.isdir(os.path.join(current_dir, 'Harrison Dataset')):
        print('Found the "Harrison Dataset" folder in the current directory.')
    else:
        print('Could not find the "Harrison Dataset" folder in the current directory.')

    if os.path.exists(os.path.join(current_dir, 'Josh Dataset')) and os.path.isdir(os.path.join(current_dir, 'Josh Dataset')):
        print('Found the "Josh Dataset" folder in the current directory.')
    else:
        print('Could not find the "Josh Dataset" folder in the current directory.')

    if os.path.exists(os.path.join(current_dir, 'Jacintha Dataset')) and os.path.isdir(os.path.join(current_dir, 'Jacintha Dataset')):
        print('Found the "Jacintha Dataset" folder in the current directory.')
    else:
        print('Could not find the "Jacintha Dataset" folder in the current directory.')

    folder_paths = [
        os.path.join(current_dir, 'Harrison Dataset'),
        os.path.join(current_dir, 'Josh Dataset'),
        os.path.join(current_dir, 'Jacintha Dataset'),

    ]

    dataMem1 = {}
    dataMem2 = {}
    dataMem3 = {}

    #----------------------------------------------------------------------------------------------------------------#
    ############################################## Step 1: Read from csv files #######################################

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
        for activity, state in [('walking', 0), ('jumping', 1)]:

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
        return dataframe_collection
    # dictionary containing data frames of raw data
    dataMem1 = read_csv_files(folder_paths[0])
    dataMem2 = read_csv_files(folder_paths[1])
    dataMem3 = read_csv_files(folder_paths[2])

    main_data = []

    for d in [dataMem1, dataMem2, dataMem3]:
        for df in d.values():
            main_data.append(df)
    main_dataframe = pd.concat(main_data, ignore_index=True)

    #----------------------------------------------------------------------------------------------------------------#
    ############################################## Step 2: Segment data ##############################################

    main_dataframe['Time (s)'] = pd.to_datetime(main_dataframe['Time (s)'], unit='s')
    main_dataframe.set_index('Time (s)', inplace=True)
    print(main_dataframe)

    window_size = 5 # Window Size
    sample_rate = 100 # Recordings made per second

    def segment_data(df, window_size, sample_rate):
        n_samples = len(df)
        window_length = window_size * sample_rate
        segments = []
        for i in range(0, n_samples, window_length):
            segment = df.iloc[i:i + window_length].copy()
            if len(segment) == window_length:
                segments.append(segment)
        return segments

    print("segmenting data...")
    segments = segment_data(main_dataframe, window_size, sample_rate)
    

    #print(segmented_dataframe.to_string())

    #------------------------------------------------------------------------------------------------------#
    #################################### Step 3: Shuffle and split data ####################################

    #X_train, X_test, Y_train, Y_test = train_test_split(segmented_dataframe[['Absolute acceleration (m/s^2)']], segmented_dataframe['state'], test_size=0.1, shuffle=True)
    train_data, test_data = train_test_split(segments, test_size=0.1, random_state=42)
    train_data = pd.concat(train_data).reset_index(drop=True)
    test_data = pd.concat(test_data).reset_index(drop=True)
    print("Train Data:")
    print(train_data)
    print("Test Data:")
    print(test_data)
    print("splitting data...")
    #print(train_data)

    #-----------------------------------------------------------------------------------------------------#
    #################################### Step 4: Preprocess test and train data ###########################
    def preprocess_dataframe(df):
        print("Preprocessing...")
        df.dropna(inplace=True)
        # Applying moving average
        # Select the filtered data with window size 50
        # Applying exponential moving average
        # Removing outliers
        sc = preprocessing.StandardScaler()
        df = pd.DataFrame(data=sc.fit_transform(df))
        return df

    #train_data_preprocessed = preprocess_dataframe(train_data)
    print("preprocessing...")
    train_data_preprocessed = train_data
    test_data_preprocessed = test_data

    #-----------------------------------------------------------------------------------------------------#
    #################################### Step 5: feature extraction #######################################
    def extract_rolling_window(df):
        """
        Extracts rolling window features from a dictionary of DataFrames.

        Args:
        - data_dict (dict): a dictionary of DataFrames, where the keys are the names of the DataFrames

        Returns:
        - features_df (DataFrame): a DataFrame containing the extracted features
        """
        features = pd.DataFrame(columns = ['max', 'min', 'mean', 'median', 'range', 'std', 'var', 'kurt', 'skew', 'state'])
        df = df.astype('float64')
        df_abs = df.iloc[:,3]
        max = df_abs.rolling(window=5).max()                 
        min = df_abs.rolling(window=5).min()
        mean = df_abs.rolling(window=5).mean()
        median =df_abs.rolling(window=5).median()
        range = df_abs.rolling(window=5).apply(lambda x: x.max() - x.min())
        std = df_abs.rolling(window=5).std()
        var = df_abs.rolling(window=5).var()
        kurt = df_abs.rolling(window=5).kurt()
        skew = df_abs.rolling(window=5).skew()
        state = df.iloc[:,4].rolling(window=5).mean() # idk even know what im doing here lmao
        #print(max)
        features['max'] = max                
        features['min'] = min
        features['mean'] = mean
        features['median'] = median
        features['range'] = range
        features['std'] = std
        features['var'] = var
        features['kurt']= kurt
        features['skew'] = skew
        features['state'] = state # idk even know what im doing here lmao
        features = features.dropna()
        return features

    print("extracting features...")
    train_data_preprocessed_features = extract_rolling_window(train_data_preprocessed)
    test_data_preprocessed_features = extract_rolling_window(test_data_preprocessed)
    print(train_data_preprocessed_features)

    def classifier(train_data, test_data):

        lab = preprocessing.LabelEncoder()

        train_data = train_data.dropna()
        test_data = test_data.dropna()
        X_train_data = train_data.iloc[:,:9]
        Y_train_data = train_data.iloc[:,9]
        Y_train_data.astype(int)
        print(Y_train_data)
        X_test_data = test_data.iloc[:,:9]
        Y_test_data = test_data.iloc[:,9]
        Y_test_data.astype(int)
        print(Y_train_data)
        print("Splitting dataframe to test and train")
        # create the classifier
        l_reg = LogisticRegression(max_iter = 10000)
        scaler = StandardScaler()
        clf = make_pipeline(scaler,l_reg)

        print("fitting to pipeline")
        clf.fit(X_train_data, Y_train_data)
        print("\n### COMPLETE ###\n")
        accuracy = clf.score(X_test_data, Y_test_data)
        print('Accuracy:', accuracy)
        
        #check if model works with a sample
        testing_dataframe = pd.read_csv('C:\\Users\\joshu\Project-Ajax\\Josh Dataset\\Walking\\josh_back_fastwalking.csv')
        def get_features_from_csv(df):
            features = pd.DataFrame(columns = ['max', 'min', 'mean', 'median', 'range', 'std', 'var', 'kurt', 'skew'])
            df = df.astype('float64')
            df_preprocessed = preprocess_dataframe(df)
            df_abs = df_preprocessed.iloc[:,4]
            max = df_abs.rolling(window=5).max()                 
            min = df_abs.rolling(window=5).min()
            mean = df_abs.rolling(window=5).mean()
            median =df_abs.rolling(window=5).median()
            range = df_abs.rolling(window=5).apply(lambda x: x.max() - x.min())
            std = df_abs.rolling(window=5).std()
            var = df_abs.rolling(window=5).var()
            kurt = df_abs.rolling(window=5).kurt()
            skew = df_abs.rolling(window=5).skew()
            #print(max)
            features['max'] = max                
            features['min'] = min
            features['mean'] = mean
            features['median'] = median
            features['range'] = range
            features['std'] = std
            features['var'] = var
            features['kurt']= kurt
            features['skew'] = skew
            return features
        testing_dataframe = get_features_from_csv(testing_dataframe)
        testing_dataframe = testing_dataframe.dropna()
        y_pred = clf.predict(testing_dataframe)
        #print(y_pred)
        num_walk_flag = np.count_nonzero(y_pred == 0)
        num_run_flag = np.count_nonzero(y_pred == 1)
        ratio_walk = num_walk_flag / len(y_pred)
        print("Walk confidence = ", ratio_walk)

    classifier(train_data_preprocessed_features,test_data_preprocessed_features)
#
if __name__ == "__main__":
    main()