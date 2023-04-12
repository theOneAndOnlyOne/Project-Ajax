import os
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pickle

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
#from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

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
    return dataframe_collection

# Step 2: For each dataframe extract rolling window and features

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
        features = pd.DataFrame(columns = [
            'x_max', 'x_min', 'x_mean', 'x_median', 'x_range', 'x_std', 'x_var', 'x_kurt', 'x_skew',
            'y_max', 'y_min', 'y_mean', 'y_median', 'y_range', 'y_std', 'y_var', 'y_kurt', 'y_skew',
            'z_max', 'z_min', 'z_mean', 'z_median', 'z_range', 'z_std', 'z_var', 'z_kurt', 'z_skew', 'state'])
        print("extracting " + key)
        df = data_dict[key]
        df = df.astype('float64')
        df_preprocessed = preprocess_dataframe(df)

        df_dim = df_preprocessed.iloc[:,1]
        max = df_dim.rolling(window= 500).max()                 
        min = df_dim.rolling(window= 500).min()
        mean = df_dim.rolling(window= 500).mean()
        median =df_dim.rolling(window= 500).median()
        range = df_dim.rolling(window= 500).apply(lambda x: x.max() - x.min())
        std = df_dim.rolling(window= 500).std()
        var = df_dim.rolling(window= 500).var()
        kurt = df_dim.rolling(window= 500).kurt()
        skew = df_dim.rolling(window= 500).skew()
        #print(max)
        features['x_max'] = max                
        features['x_min'] = min
        features['x_mean'] = mean
        features['x_median'] = median
        features['x_range'] = range
        features['x_std'] = std
        features['x_var'] = var
        features['x_kurt']= kurt
        features['x_skew'] = skew

        df_dim = df_preprocessed.iloc[:,2]
        max = df_dim.rolling(window= 500).max()                 
        min = df_dim.rolling(window= 500).min()
        mean = df_dim.rolling(window= 500).mean()
        median =df_dim.rolling(window= 500).median()
        range = df_dim.rolling(window= 500).apply(lambda x: x.max() - x.min())
        std = df_dim.rolling(window= 500).std()
        var = df_dim.rolling(window= 500).var()
        kurt = df_dim.rolling(window= 500).kurt()
        skew = df_dim.rolling(window= 500).skew()
        #print(max)
        features['y_max'] = max                
        features['y_min'] = min
        features['y_mean'] = mean
        features['y_median'] = median
        features['y_range'] = range
        features['y_std'] = std
        features['y_var'] = var
        features['y_kurt']= kurt
        features['y_skew'] = skew

        df_dim = df_preprocessed.iloc[:,3]
        max = df_dim.rolling(window= 500).max()                 
        min = df_dim.rolling(window= 500).min()
        mean = df_dim.rolling(window= 500).mean()
        median =df_dim.rolling(window= 500).median()
        range = df_dim.rolling(window= 500).apply(lambda x: x.max() - x.min())
        std = df_dim.rolling(window= 500).std()
        var = df_dim.rolling(window= 500).var()
        kurt = df_dim.rolling(window= 500).kurt()
        skew = df_dim.rolling(window= 500).skew()
        #print(max)
        features['z_max'] = max                
        features['z_min'] = min
        features['z_mean'] = mean
        features['z_median'] = median
        features['z_range'] = range
        features['z_std'] = std
        features['z_var'] = var
        features['z_kurt']= kurt
        features['z_skew'] = skew

        state = df.iloc[:,5].rolling(window= 500).mean() # idk even know what im doing here lmao
        features['state'] = state # idk even know what im doing here lmao

        rolling_window_collection[key + "_features"] = features

    return rolling_window_collection

# Helper method, used to get features without a specific label
def get_features_from_csv(df):
    features = pd.DataFrame(columns = ['x_max', 'x_min', 'x_mean', 'x_median', 'x_range', 'x_std', 'x_var', 'x_kurt', 'x_skew',
            'y_max', 'y_min', 'y_mean', 'y_median', 'y_range', 'y_std', 'y_var', 'y_kurt', 'y_skew',
            'z_max', 'z_min', 'z_mean', 'z_median', 'z_range', 'z_std', 'z_var', 'z_kurt', 'z_skew'])
    
    df = df.astype('float64')
    df_preprocessed = preprocess_dataframe(df)
    df_dim = df_preprocessed.iloc[:,1]
    max = df_dim.rolling(window= 500).max()                 
    min = df_dim.rolling(window= 500).min()
    mean = df_dim.rolling(window= 500).mean()
    median =df_dim.rolling(window= 500).median()
    range = df_dim.rolling(window= 500).apply(lambda x: x.max() - x.min())
    std = df_dim.rolling(window= 500).std()
    var = df_dim.rolling(window= 500).var()
    kurt = df_dim.rolling(window= 500).kurt()
    skew = df_dim.rolling(window= 500).skew()
    #print(max)
    features['x_max'] = max                
    features['x_min'] = min
    features['x_mean'] = mean
    features['x_median'] = median
    features['x_range'] = range
    features['x_std'] = std
    features['x_var'] = var
    features['x_kurt']= kurt
    features['x_skew'] = skew

    df_dim = df_preprocessed.iloc[:,2]
    max = df_dim.rolling(window= 500).max()                 
    min = df_dim.rolling(window= 500).min()
    mean = df_dim.rolling(window= 500).mean()
    median =df_dim.rolling(window= 500).median()
    range = df_dim.rolling(window= 500).apply(lambda x: x.max() - x.min())
    std = df_dim.rolling(window= 500).std()
    var = df_dim.rolling(window= 500).var()
    kurt = df_dim.rolling(window= 500).kurt()
    skew = df_dim.rolling(window= 500).skew()
    #print(max)
    features['y_max'] = max                
    features['y_min'] = min
    features['y_mean'] = mean
    features['y_median'] = median
    features['y_range'] = range
    features['y_std'] = std
    features['y_var'] = var
    features['y_kurt']= kurt
    features['y_skew'] = skew

    df_dim = df_preprocessed.iloc[:,3]
    max = df_dim.rolling(window= 500).max()                 
    min = df_dim.rolling(window= 500).min()
    mean = df_dim.rolling(window= 500).mean()
    median =df_dim.rolling(window= 500).median()
    range = df_dim.rolling(window= 500).apply(lambda x: x.max() - x.min())
    std = df_dim.rolling(window= 500).std()
    var = df_dim.rolling(window= 500).var()
    kurt = df_dim.rolling(window= 500).kurt()
    skew = df_dim.rolling(window= 500).skew()
    #print(max)
    features['z_max'] = max                
    features['z_min'] = min
    features['z_mean'] = mean
    features['z_median'] = median
    features['z_range'] = range
    features['z_std'] = std
    features['z_var'] = var
    features['z_kurt']= kurt
    features['z_skew'] = skew
    return features

# Step 2.1 : Preprocess each dataset before extracting features
def preprocess_dataframe(df):
    print("Preprocessing...")
    window_size = 51
    alpha = 0.1
    #df = df.rolling(window_size).mean()
    df = df.ewm(alpha, min_periods=window_size).mean()
    sc = preprocessing.StandardScaler()
    df = pd.DataFrame(data=sc.fit_transform(df))
    return df

# Stage 3: sort data frames into groups
def write_dataframes_to_hdf5_group(dataframes_dict, file_path, group):
    """
    Writes dataframe dictionary to a given group stored in a file path
    
    Args:
    - dataframes_dict (dict): a dictionary of DataFrames, where the keys are the names of the DataFrames
    - file_path (string) : file path of hdf5 file
    - group (hdf5 group) : specified group we want to store our info in
    
    Returns:
    - none
    """
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

# Step 4: run classifier to train and test dataset
def classifier(df):
    df = df.dropna()
    print("Splitting dataframe to test and train")
    X_train, X_test, Y_train, Y_test = train_test_split(
        df[['x_max', 'x_min', 'x_mean', 'x_median', 'x_range', 'x_std', 'x_var', 'x_kurt', 'x_skew',
            'y_max', 'y_min', 'y_mean', 'y_median', 'y_range', 'y_std', 'y_var', 'y_kurt', 'y_skew',
            'z_max', 'z_min', 'z_mean', 'z_median', 'z_range', 'z_std', 'z_var', 'z_kurt', 'z_skew']], 
        df['state'], test_size=0.1, shuffle=True)
    # create the classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=5)
    l_reg = LogisticRegression(max_iter = 10000)
    scaler = StandardScaler()

    clf1 = make_pipeline(scaler,l_reg)
    clf2 = make_pipeline(scaler,rf)

    clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2)], voting='soft')
    
    print("fitting to pipeline")
    clf.fit(X_train, Y_train)
    print("\n### FITTING COMPLETE ###\n")
    # Define the training curve parameters
    #train_sizes = np.linspace(0.1, 1.0, 10)  # Training set sizes as a fraction of total dataset size
    #train_sizes_abs, train_scores, test_scores = learning_curve(
    #    clf, X_train, Y_train, train_sizes=train_sizes, cv=5, scoring='neg_mean_squared_error')
#
    ## Compute the mean and standard deviation of training and test scores
    #train_scores_mean = -np.mean(train_scores, axis=1)
    #train_scores_std = np.std(train_scores, axis=1)
    #test_scores_mean = -np.mean(test_scores, axis=1)
    #test_scores_std = np.std(test_scores, axis=1)
#
    ## Plot the training curve
    #plt.figure(figsize=(10, 6))
    #plt.title("Training Curve")
    #plt.xlabel("Training Set Size")
    #plt.ylabel("Mean Squared Error")
    #plt.grid()
    #plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
    #                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
    #plt.fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
    #                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
    #plt.plot(train_sizes_abs, train_scores_mean, 'o-', color="r", label="Training score")
    #plt.plot(train_sizes_abs, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    #plt.legend(loc="best")
    #plt.show()
    
    print("\n### TRAINING COMPLETE ###\n")


    # --------- Classifier Metrics ------- #
    y_pred_clf = clf.predict(X_test)
    acc = accuracy_score(Y_test, y_pred_clf)
    recall = recall_score(Y_test, y_pred_clf)
    f1 = f1_score(Y_test, y_pred_clf)
    precision = precision_score(Y_test, y_pred_clf)
    roc_auc = roc_auc_score(Y_test, y_pred_clf)
    confusion = confusion_matrix(Y_test, y_pred_clf)
    classification = classification_report(Y_test, y_pred_clf)
    #accuracy = clf.score(X_test,Y_test)
    print('#### Classifier Metrics ####')
    print('Accuracy:', acc)
    print('Recall:', recall)
    print('F1 Score:', f1)
    print("Precision:", precision)
    print("ROC AUC:", roc_auc)
    print("Confusion Matrix:\n", confusion)
    print("Classification Report:\n", classification)
    # -------------------------------------#

    #evaluate_csv('C:\\Users\\joshu\Project-Ajax\\Evaluation Dataset\\sample_jumping.csv')

    return clf

def evaluate_csv(file_path):
    current_dir = os.getcwd()
    filename = os.path.join(current_dir, 'finalized_model.sav')
    clf = pickle.load(open(filename, 'rb'))
    testing_dataframe = pd.read_csv(file_path)

    testing_dataframe_features = get_features_from_csv(testing_dataframe)
    testing_dataframe_features = testing_dataframe_features.fillna(0)
    print(testing_dataframe_features)
    y_pred = clf.predict(testing_dataframe_features)
    print(y_pred)
    y_pred_df = pd.DataFrame({'state': y_pred})

    result_dataframe = pd.concat([testing_dataframe_features, y_pred_df], axis=1)

    num_walk_flag = np.count_nonzero(y_pred == 1)
    num_run_flag = np.count_nonzero(y_pred == 2)
    ratio_walk = num_walk_flag / len(y_pred)
    print("Walk confidence = ", ratio_walk)

    fig, ax = plt.subplots(figsize = (8.5, 4),
             dpi = 100)
    x = testing_dataframe['Time (s)']
    y = testing_dataframe['Absolute acceleration (m/s^2)']
    
    ax.plot(x,y)

    # Find the indices where y_pred is 2
    y_pred_2_indices = np.where(y_pred == 2)[0]

    # Find the indices where y_pred is 1
    y_pred_1_indices = np.where(y_pred == 1)[0]

    # Create a mask for y_pred = 2
    mask_2 = np.zeros_like(y, dtype=bool)
    mask_2[y_pred_2_indices] = True

    # Create a mask for y_pred = 1
    mask_1 = np.zeros_like(y, dtype=bool)
    mask_1[y_pred_1_indices] = True

    # Create a mask to exclude the first 5sec points (due to rolling window)
    mask_exclude = np.zeros_like(y, dtype=bool)
    mask_exclude[500:] = True
    print(mask_exclude)

    mask_1 = np.logical_and(mask_exclude, mask_1)
    mask_2 = np.logical_and(mask_exclude, mask_2)

    # Shade the areas where y_pred is 2 as green
    jumping_section = plt.fill_between(x, y, where=mask_2, color='green', alpha=0.5)

    # Shade the areas where y_pred is 1 as grey
    walking_section = plt.fill_between(x, y, where=mask_1, color='grey', alpha=0.5)

    ax.legend([walking_section, jumping_section], ['Walking', 'Jumping'])
    ax.set_xlabel('Time Elapsed(s)')
    ax.set_ylabel('Absolute Acceleration(m/s^2)')

    #plt.show()
    return fig, result_dataframe

# Main Function
def init_pipeline(): 
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

    # dictionary containing data frames of raw data
    # original
    dataMem1 = read_csv_files(folder_paths[0])
    dataMem2 = read_csv_files(folder_paths[1])
    dataMem3 = read_csv_files(folder_paths[2])

    dataMem1WindowFeatures = extract_rolling_window(dataMem1)
    dataMem2WindowFeatures = extract_rolling_window(dataMem2)
    dataMem3WindowFeatures = extract_rolling_window(dataMem3)
    
    print("Merging features from all members...")
    feature_list = []

    for d in [dataMem1WindowFeatures, dataMem2WindowFeatures, dataMem3WindowFeatures]:
        for df in d.values():
            feature_list.append(df)

    feature_dataframe = pd.concat(feature_list,ignore_index=True)
    print(feature_dataframe)
    # TO DO: Add dictionary for windows of data frames
    print("Initiating classification")
    clf = classifier(feature_dataframe)

    filename = 'finalized_model.sav'
    file = open(filename, 'wb')
    pickle.dump(clf, file)
    file.close()

    output_file = 'data.h5'
    output = h5py.File(output_file, 'w')
    mem1_group = output.create_group("Harrison Dataset")
    mem2_group = output.create_group("Josh Dataset")
    mem3_group = output.create_group("Jacintha Dataset")
    classifier_data_group = output.create_group("Classifier Data")

    write_dataframes_to_hdf5_group(dataMem1, output_file, mem1_group)
    write_dataframes_to_hdf5_group(dataMem2, output_file, mem2_group)
    write_dataframes_to_hdf5_group(dataMem3, output_file, mem3_group)
    #classifier_data_group.create_dataset('classifier', data=clf)
    # save the model to disk
    
    evaluate_csv('C:\\Users\\joshu\Project-Ajax\\Evaluation Dataset\\sample_jumping.csv')

    # Prints HDF5 structure
    #with h5py.File(output_file, 'r') as hdf:
    #    items = list(hdf.items())
    #    print(items)
    #    mem1_items=list(mem1_group.items())
    #    mem2_items=list(mem2_group.items())
    #    mem3_items=list(mem3_group.items())
    #    window_items = list(window_group.items())
    #    print("\n#######     Harrison    #######")
    #    for item in mem1_items:
    #        print(item)
    #    print("\n#######       Josh      #######")
    #    for item in mem2_items:
    #        print(item)
    #    print("\n#######     Jacintha    #######")
    #    for item in mem3_items:
    #        print(item)
#
    #    print("\n#######     Windows    #######")
    #    for item in window_items:
    #        print(item)

if __name__ == "__main__":
    init_pipeline()
    #evaluate_csv('C:\\Users\\joshu\Project-Ajax\\Evaluation Dataset\\sample_jumping.csv')
    #Harrison