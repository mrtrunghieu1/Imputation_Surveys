# Necessary packages
import sys
sys.path.append("..")
import datetime
import csv
import numpy as np
import os
# from sklearn import model_selection, preprocessing
# My Library
from data_loader import data_loader
from gain import gain

# from util import csv_reader
# from data_helper import file_list, data_folder, data_K_Fold, imputed_dataset
file_list = ['abalone', 'heart', 'tic-tac-toe']
data_K_Fold = "C:\\Users\\Administrator\\Desktop\\Code_Test\\Research\\Data-Imputation-master\\data_K_Fold"

def check_exist_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def csv_reader(save_folder, file_name, i, method, missingness):
    file_name_folder = os.path.join(save_folder, file_name)
    if method == 'original_data' and missingness == None:
    
        train_folder = os.path.join(file_name_folder, 'train/original_data')
        check_exist_folder(train_folder)
        test_folder = os.path.join(file_name_folder, 'test/original_data')
        check_exist_folder(test_folder)
        train_path = os.path.join(train_folder, 'train_{}.csv'.format(i))
        test_path = os.path.join(test_folder, 'test_{}.csv'.format(i))
    elif method == 'data_missing':
        # train_folder = os.path.join(file_name_folder, 'train/data_missing_{}'.format(missingness))
        # test_folder = os.path.join(file_name_folder, 'test/data_missing_{}'.format(missingness))
        train_folder = os.path.join(file_name_folder, 'train/train_{}'.format(i))
        test_folder = os.path.join(file_name_folder, 'test/test_{}'.format(i))
        train_path = os.path.join(train_folder, 'train_{}_missing_{}.csv'.format(i, missingness))
        test_path = os.path.join(test_folder, 'test_{}_missing_{}.csv'.format(i, missingness))
    # Loading train and test csv
    X_train = np.genfromtxt(train_path, delimiter=',')
    X_test = np.genfromtxt(test_path, delimiter=',')

    return X_train, X_test
'''Begin start code Python'''
try:
    from_id = int(sys.args[0])
    to_id = int(sys.args[1])
except:
    from_id = 0
    to_id = len(file_list)

# Initial parameter
fold_size = 11
missingness_flag = [10, 20, 30, 40, 50]  # t% missing data  
seed = 42
cat_cols = [1,2,5,6,8,10,11,12,13]
num_cols = [0,3,4,7,9]

gain_parameters = {'batch_size': 128,
                     'hint_rate': 0.9,
                     'alpha': 100,
                     'iterations': 10000}

# Main program
for i_file in range(from_id, to_id):
    file_name = file_list[i_file]
    print(datetime.datetime.now(), "File {}: {}".format(i_file, file_name))

    for i in range(1, fold_size):
        (D_train, D_test) = csv_reader(data_K_Fold, file_name, i, method='original_data', missingness=None)
        x_train = D_train[:, :(D_train.shape[1] - 1)]
        y_train = D_train[:,-1]
        x_test = D_test[:, :(D_test.shape[1] - 1)]
        y_test = D_test[:,-1]
        D = np.concatenate((D_train, D_test), axis = 0)
        for missingness in missingness_flag:
             # Load data and introduce missingness
            missingness /= 100
            ori_data_x, miss_data_x, data_m = data_loader(D, missingness)
            # Impute missing data
            imputed_data_x = gain(miss_data_x, gain_parameters)
            print(imputed_data_x)




'''Code Finished'''