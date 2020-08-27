# Library
import os
from numpy import savetxt
import csv
import numpy as np
import random

#My Library
from util import check_exist_folder

'''Begin start code Python'''
def K_Fold_cross_validation(kf, X_full, save_folder, file_name, count_fold):
    file_name_folder = os.path.join(save_folder, file_name)

    train_folder = os.path.join(file_name_folder, 'train/original_data')
    check_exist_folder(train_folder)
    test_folder = os.path.join(file_name_folder, 'test/original_data')
    check_exist_folder(test_folder)
    # Split K_fold    
    for train_index, test_index in kf.split(X_full):
        # Setup path enviroments
        count_fold += 1
        train_path = os.path.join(train_folder, 'train_{}.csv'.format(count_fold))
        test_path = os.path.join(test_folder, 'test_{}.csv'.format(count_fold))

        X_train, X_test = X_full[train_index], X_full[test_index]
        savetxt(train_path, X_train, delimiter=',')
        savetxt(test_path, X_test, delimiter=',')

def missing_data_generation(data, missingness):
    # Generate data from original data with missingness
    X_data = data[:, : (data.shape[1] - 1)]
    y_data = data[:, -1]
    y_data = y_data.reshape(1, data.shape[0])
    X_matrix = np.zeros((data.shape[0], data.shape[1]))

    missingness /= 100
    missing_size = int(X_data.shape[0] * missingness)
    idx_sample = random.sample(range(X_data.shape[0]), missing_size)
    for i in idx_sample:
        number_missing_feature = random.randint(1, X_data.shape[1])
        # Random position without duplicate
        index_missing_feature = random.sample(range(X_data.shape[1]), number_missing_feature)
        for j in index_missing_feature:
            X_data[i][j] = np.NaN
    X_matrix[:, : (data.shape[1]-1)] = X_data
    X_matrix[:, -1] = y_data
    return X_matrix
'''Code Finished'''