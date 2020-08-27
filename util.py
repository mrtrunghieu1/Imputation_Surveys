# Library
import os
from numpy import savetxt
import csv
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import json
# My Library

'''Begin start code Python'''
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
    
def write_file(data_train, data_test, save_folder, file_name, missingness, i):
    file_name_folder = os.path.join(save_folder, file_name)
    sub_train_path = os.path.join(file_name_folder, 'train/train_{}'.format(i))
    check_exist_folder(sub_train_path)
    sub_test_path = os.path.join(file_name_folder, 'test/test_{}'.format(i))
    check_exist_folder(sub_test_path)
    data_missing_train_path = os.path.join(sub_train_path, 'train_{}_missing_{}.csv'.format(i, missingness))
    data_missing_test_path = os.path.join(sub_test_path, 'test_{}_missing_{}.csv'.format(i, missingness))
    savetxt(data_missing_train_path, data_train, delimiter=',')
    savetxt(data_missing_test_path, data_test, delimiter=',')

def evaluation_report(predict, grouth_truth):
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(grouth_truth, predict, average='macro')
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(grouth_truth, predict, average='micro')
    accuracy = accuracy_score(grouth_truth, predict)
    return accuracy, p_macro, r_macro, f1_macro, p_micro, r_micro, f1_micro

def write_report(dict, save_folder, imp_name, missing, classification_name):
    file_name_folder = os.path.join(save_folder, imp_name)
    missing_folder = os.path.join(file_name_folder, 'data_missing_{}'.format(missing))
    classification_folder = os.path.join(missing_folder, classification_name)
    check_exist_folder(classification_folder)

    name_metrics = ['accuracy', 'p_macro', 'r_macro', 'f1_macro', 'p_micro', 'r_micro', 'f1_micro']
    for name_metric in name_metrics:
        data = {'{}'.format(name_metric) : dict[name_metric], 'mean': np.mean(dict[name_metric]),
        'std': np.std(dict[name_metric])}
        path = os.path.join(classification_folder, '{}.txt'.format(name_metric))
        with open(path, 'w') as outfile:
            json.dump(data, outfile)    
        




'''Code Finished'''
