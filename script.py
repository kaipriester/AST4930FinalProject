#############
#  README   #
#############
'''
Original Authors: Siddha Ganju, Peter Jenniskins

Date: 2018.10.08

Organizations: NASA Frontier Development Lab

Extensive edits and updates: Surya Ambardar

Additional supervised learning methods: Kai Priester

Date: 2020.09.05(original), 2021.11.16(updates)

Summary: data extracted from summary and rejected meteor instances processed with supervised learning methods
'''


#############
#  IMPORTS  #
#############
from collections import defaultdict
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']= '2'
import warnings
with warnings.catch_warnings():
        warnings.filterwarnings("ignore")#, category=FutureWarning)
        from os import listdir
        from datetime import date
        from datetime import datetime
        from pathlib2 import Path
        import itertools
        import pandas as pd
        import numpy as np
        import random
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        from sklearn.utils import shuffle
        
        #added
        import time 
        from sklearn import neighbors
        from sklearn.model_selection import GridSearchCV
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.svm import SVC
        import matplotlib.pyplot as plt
        from sklearn.inspection import permutation_importance
        from collections import Counter
        import tensorflow.keras as keras

#############
# FUNCTIONS #
#############


#add avg to it
len_train_instance = 5
# directory where gefdat dir is present
dir = "gefdat"
random.seed(1234)


def is_file_standard(file_path):
        if len(file_path.split("\\")[-1].split("_")) == 9:
                return True
        else:
                return False
            
# confirm summary and rejected files are present
#USED
def confirm_files(dir, files):
    result = True
    for file in files:
        path = Path(dir + "/" + file)
        result = result and path.is_file()
    return result

    
# read summary and rejected files
def read_summary_files(dir, summary_files):
    summary_ids = []
    for file in summary_files:
        path = dir + "/" + file
        f = open(path, 'r')
        # Read and ignore header lines MeteorLog
        for _ in itertools.repeat(None,3):
            _ = f.readline()
        # do nothing to headers
        # read rest of the file
        data = f.readlines()
        for each_line in data:
            # split by space and take first meteor id only
            temp_id = each_line.split()[0]
            date_id = each_line.split()[1]
            if temp_id not in summary_ids:
                summary_ids.append(temp_id)
        return summary_ids

def ensure_no_overlap(summary_ids, rejected_ids):
    return len(list(set(summary_ids).intersection(rejected_ids)))

# function to transfer MEAS filenames so that we get "meteorid" = [filename_meteorid**, filename_meteorid**]
def make_dict(all_meas_files):
    combined_meas = defaultdict(list)
    for each in all_meas_files:
        meteor_id = each.split(".")[0].split("_")[-1]
        combined_meas[meteor_id].append(each)
    return combined_meas

# function to combine text of all MEAS files
def combine_files(dir, meteor_id, file_list, category):
    for each_file in file_list:
        path = dir + "/" + each_file
        f_in = open(path, 'r')
        # Read and ignore header lines - 8 lines
        for _ in itertools.repeat(None,8):
            _ = f_in.readline()
        # do nothing to headers
        # read rest of the file
        data = f_in.readlines()
        f_in.close()
        # write these lines to output file and dir
        output_path = dir + "/" + category
        output_date = dir.split(".")[0].split("\\")[-2]

#function to get list of summary and rejected filenames
def parse_gefdat(dir):
    # list of all CAMS_MEAS_*** files
    all_meas_files = []
    # list of all "Summary" Meteor/Orbit files
    summary_files = []
    # list of all "Rejected" Meteor/Orbit files
    rejected_files = []

    # get list of files from gefdat folders
    for file in os.listdir(dir):
        if file.endswith(".txt") and "CAMS_MEAS" in file:
            all_meas_files.append(file)
        elif file.endswith(".txt") and "Summary" in file:
            summary_files.append(file)
        elif file.endswith(".txt") and "Rejected" in file:
            rejected_files.append(file)

    # confirm summary and rejected files are present
    if not confirm_files(dir, summary_files):
        print("Summary files do not exist")
        exit()

    if not confirm_files(dir, rejected_files):
        print("Rejected files do not exist")
        exit()

    # make lists of meteor ids from summary and rejected
    summary_ids = read_summary_files(dir, summary_files)

    rejected_ids = read_summary_files(dir, rejected_files)

    # ensure that all ids in summary_ids and rejected_ids are unique and non-overlapping
    if ensure_no_overlap(summary_ids, rejected_ids) != 0:
        print(summary_ids, rejected_ids, "Overlapping ids in summary and rejected")
        exit(0)

    # combine MEAS filenames so that we get "meteorid" = [filename_meteorid**, filename_meteorid**]
    combined_meas = make_dict(all_meas_files)

    # combine and write summary meteor ids
    for each_id in summary_ids:
        combine_files(dir, each_id, combined_meas[each_id], "Summary")

    # combine and write rejected meteor ids
    for each_id in rejected_ids:
        combine_files(dir, each_id, combined_meas[each_id], "Rejected")


# function to parse directory structures
def parse_dirs(parent_dir):
    gefdat_folders = [x[0] for x in os.walk(parent_dir) if "gefdat" in x[0] and "Summary" not in x[0] and "Rejected" not in x[0] and "displayJPEGs" not in x[0] and "Spectral Files" not in x[0] and "not used" not in x[0]]
    for each_gefdat in gefdat_folders:
        parse_gefdat(each_gefdat)

#fxn get_filenames returns lists of all  filenames in CAMS-Summary and CAMS-Rejected
def get_filenames():
    summary_list = []
    #CHANGE DIRECTORIES IF NEEDED
    summary_listdir = os.listdir("../CAMS-Summary")
    for filename in summary_listdir:
        summary_list.append(os.path.join("../CAMS-Summary", filename))
    rejected_list = []
    rejected_listdir = os.listdir("../CAMS-Rejected")
    for filename in rejected_listdir:
        rejected_list.append(os.path.join("../CAMS-Rejected", filename))
    return summary_list, rejected_list

#fxn read_single_file takes a file name and extract information from relevant columns
def parse_meas_file(filename):
        data = pd.read_csv(filename, sep="  " ,header = None, engine = 'python')
        data.columns = ["frame", "time", "range", "height", "vel", "intenr", "mv_vmag", "mv_flux", "lat", "long", "azims", "zenang"]
        columns_to_use = ["height", "vel", "mv_flux", "lat", "long"]
        select_dataframe = data[columns_to_use].copy()
        select_dataframe = select_dataframe.replace('[^\d.]','0', regex=True)
        del data
        scaler = MinMaxScaler(feature_range = (0,1))
        for each_col in columns_to_use:
                select_dataframe[each_col] = scaler.fit_transform(np.array(select_dataframe[each_col]).reshape(-1,1))
        return select_dataframe.to_numpy()

#USED
def read_single_file(filename):
        return parse_meas_file(filename)

# fxn shape_meteor_data extracts and shapes information from the file
def shape_meteor_data(contents, len_instance):
        if contents.shape[0] < len_instance:
                return 0
        multiples = contents.shape[0]//len_instance
        less_rows_contents = contents[ 0:len_instance * multiples, : ]
        inference_data = np.array_split( less_rows_contents, multiples)
        final_data = np.array(inference_data)
        return final_data

# fxn scale_meteor_data extracts information from files longer than len_instance(60 for now)
#timesteps, breaking files longer than 120 timesteps into multiple instances
def scale_meteor_data(len_instance = len_train_instance):
    scaler = MinMaxScaler(feature_range = (0,1))
    summary_list, rejected_list = get_filenames()
#     random.shuffle(summary_list)
#     random.shuffle(rejected_list)
    summary_list = summary_list
    rejected_list = rejected_list
    for j in range(len(summary_list)):
        contents = read_single_file(summary_list[j])
        to_add = shape_meteor_data(contents, len_instance)
        if isinstance(to_add, int):
                continue
        if j == 0:
                summary_instances =  to_add
        else:
                try:
                        summary_instances = np.append(summary_instances, to_add, axis=0)
                except:
                        summary_instances =  to_add
                        print("summary except")
    for j in range(len(rejected_list)):
        contents = read_single_file(rejected_list[j])
        to_add = shape_meteor_data(contents, len_instance)
        if isinstance(to_add, int):
                continue
        if j == 0:
                rejected_instances = to_add
        else:
                try:
                        rejected_instances = np.append(rejected_instances, to_add, axis = 0)
                except:
                        rejected_instances = to_add
                        print("rejected except")
    print("\n", len(summary_instances), " summary instances and ", len(rejected_instances), "rejected instances created \n")
    return np.array(summary_instances), np.array(rejected_instances)
  
#fxn create_dataset labels data with 0 for rejected or 1 for summary,
# and splits labelled data into test and train
def create_dataset(len_instance = len_train_instance):
    summ, rej = scale_meteor_data(len_instance)
    x, y = [], []
    for i in range(summ.shape[0]):
        x.append(summ[i])
        y.append(1)
    for i in range(rej.shape[0]):
        x.append(rej[i])
        y.append(0)
    x, y = shuffle(x, y)
    end = int(len(y) * .90)
    train_x = np.array(x[:end])
    train_y = np.array(y[:end])
    test_x = np.array(x[end:])
    test_y = np.array(y[end:])
    print("Full dataset created: ", "# Summary - ", len(summ), "# Rejected - ", len(rej))
    return x, y, train_x, train_y, test_x, test_y

#data extracted
# DATA EXTRACTION END ##########################################################################################################################

def knn_classifier(train_x, train_y, test_x, test_y):
    print("KNN------------------------------------------------------------------------------------")

    tstart = time.time()

    pipe = Pipeline([
        ('knn', KNeighborsClassifier())
    ])

    param_grid = {'knn__n_neighbors': np.arange(10)+1,
                 'knn__weights': ['uniform','distance']}
    grid_search = GridSearchCV(pipe, param_grid, cv=5, return_train_score=True, verbose=1)
    grid_search.fit(train_x, train_y)
    
    print("Elapsed time: {:.2f}".format(time.time()-tstart)+" seconds")
    print("Test score: {:.2f}".format(grid_search.score(test_x, test_y)))
    print("Best parameters: {}".format(grid_search.best_params_))
    
    model = grid_search.best_estimator_
    model.fit(train_x, train_y)
    
    r = permutation_importance(model, test_x, test_y, n_repeats=10, random_state=0)
    
    return r
    
def dt_classifier(train_x, train_y, test_x, test_y):
    print("DT------------------------------------------------------------------------------------")
    
    tstart = time.time()
    
    param_grid = {'max_depth': np.arange(10)+1,
              'criterion': ['gini','entropy']}

    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=0), param_grid, cv=5, 
                           return_train_score=True, verbose=1)
    grid_search.fit(train_x, train_y)

    print("Elapsed time: {:.2f}".format(time.time()-tstart)+" seconds")
    print("Test score: {:.2f}".format(grid_search.score(test_x, test_y)))
    print("Best parameters: {}".format(grid_search.best_params_))
    
    model = grid_search.best_estimator_
    model.fit(train_x, train_y)
    
    r = permutation_importance(model, test_x, test_y, n_repeats=10, random_state=0)
    
    return r
    

def svm_classifier(train_x, train_y, test_x, test_y):
    print("SVM------------------------------------------------------------------------------------")
    
    tstart = time.time()
    
    pipe = Pipeline([
        ('SVM', SVC(kernel='rbf'))
    ])

    param_grid = {'SVM__C': [0.01, 0.1, 1., 10., 100.],
                  'SVM__gamma': [0.01, 0.1, 1., 10., 100.]}

    grid_search = GridSearchCV(pipe, param_grid, cv=5, return_train_score=True, verbose=1)
    grid_search.fit(train_x, train_y)
    
    print("Elapsed time: {:.2f}".format(time.time()-tstart)+" seconds")
    print("Test score: {:.2f}".format(grid_search.score(test_x, test_y)))
    print("Best parameters: {}".format(grid_search.best_params_))
    
    model = grid_search.best_estimator_
    model.fit(train_x, train_y)
    
    r = permutation_importance(model, test_x, test_y, n_repeats=10, random_state=0)
    
    return r

def nn_classifier(train_x, train_y, test_x, test_y):
    print("NN------------------------------------------------------------------------------------")
    
    tstart = time.time()
    
    model = keras.models.Sequential([
        keras.layers.Input(shape=train_x.shape[1]),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dense(1000, activation='relu',),
        keras.layers.Dense(np.unique(train_y).shape[0], activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')
    
    history = model.fit(train_x, train_y, epochs=150, validation_split=0.2)
    
    loss, accuracy = model.evaluate(test_x, test_y, verbose=0)
    
    print("Elapsed time: {:.2f}".format(time.time()-tstart)+" seconds")
    print("Test score: {:.2f}".format(accuracy))
    
def plot_feature_importances(model, name):
    n_features = train_x.shape[1]
    plt.barh(np.arange(n_features), r.importances_mean, align='center')
    if(name == "inter2"):
        plt.yticks(np.arange(n_features),["height", "vel", "mv_flux"])
    else:
        plt.yticks(np.arange(n_features), ["height", "vel", "mv_flux", "lat", "long"])
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.show()
    plt.savefig('results/' + name + '.png')    

    
#CONTROL#################################################################################################    
x, y, train_x, train_y, test_x, test_y = create_dataset(len_instance=60)
train_x = train_x[:,0,:]
test_x = test_x[:,0,:]

print("Datasets------------------------------------------------------------------------------------")
print("train_x:")
print(train_x)
print("train_y:")
print(train_y)
print("test_x:")
print(test_x)
print("test_y:")
print(test_y)

#INSPECT FEATURE IMPORTANCE###########################################################################
#"height", "vel", "mv_flux", "lat", "long"
# plt.scatter(train_x[:,2], train_x[:,0], alpha=0.5, c=train_y)
# plt.xlabel("mv_flux")
# plt.ylabel("height")

# plt.savefig('results/classify.png')

r = knn_classifier(train_x, train_y, test_x, test_y)
plot_feature_importances(r, "knn")

r = dt_classifier(train_x, train_y, test_x, test_y)
plot_feature_importances(r, "dt")

r = svm_classifier(train_x, train_y, test_x, test_y)
plot_feature_importances(r, "svm")

nn_classifier(train_x, train_y, test_x, test_y)


#TRY MODELS WITH ONE FEATURE#############################################################################
train_x = np.delete(train_x, [0,1,3,4], 1)
test_x = np.delete(test_x, [0,1,3,4], 1)
print(test_x)

knn_classifier(train_x, train_y, test_x, test_y)
dt_classifier(train_x, train_y, test_x, test_y)
svm_classifier(train_x, train_y, test_x, test_y)


