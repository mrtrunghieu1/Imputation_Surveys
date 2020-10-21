'''Main function for SOTA-classification with imputed dataset'''

# Necessary packages
import sys
import datetime
import os
import numpy as np
import argparse

# My packages
from data_helper import file_list, imputed_dataset, result_path
from utils import csv_reader, evaluation_report, write_report
from train import model_prediction

'''Start code Python'''
def main(args):
    """ Main function for classification with imputed dataset
    
    Args:
        - from_id: start index to file list
        - to_id: end index to file list
        - fold_size: fold_size start from index 1 

    Returns:
        -     
    """
    # Input parameters
    from_id = args.from_id
    to_id = args.to_id
    fold_size = args.fold_size


    # Initial parameters
    binary_classifiers = [1, 1, 1, 1]  # 1: Activate or 0: Deactivate
    classfication_flag = [i for i, clsf in enumerate(binary_classifiers) if clsf == 1]
    missingness_flag = [0, 10, 20, 30, 40, 50]  # t% missing data 

    # Loading data
    for i_file in range(from_id, to_id):
        file_name = file_list[i_file]
        print(datetime.datetime.now(), "File {}: {}".format(i_file, file_name))
        file_data_path = os.path.join(imputed_dataset, file_name)
        result_data_path = os.path.join(result_path, file_name)
        for name_imputation in os.listdir(file_data_path):
            for missing in missingness_flag:
                for clf_flag in classfication_flag:
                    dict_eval = {'accuracy': [], 'p_macro': [], 'r_macro': [], 'f1_macro': [],
                                'p_micro': [], 'r_micro': [], 'f1_micro': []}
                    for i in range(1, fold_size):
                        D_train, D_test = csv_reader(file_data_path, name_imputation, i, method='data_missing',
                                                    missingness=missing)

                        features_D_train = D_train[:, :-1]
                        labels_D_train = D_train[:, -1].astype(np.int32)
                        features_D_test = D_test[:, :-1]
                        labels_D_test = D_test[:, -1].astype(np.int32)

                        classes = np.unique(labels_D_test)
                        n_classes = len(classes)

                        labels_predicted, name_classification_algo = model_prediction(features_D_train, features_D_test,
                                                                                    labels_D_train, clf_flag, n_classes)
                        accuracy, p_macro, r_macro, f1_macro, p_micro, r_micro, f1_micro = evaluation_report(
                            labels_predicted, labels_D_test)
                        dict_eval['accuracy'].append(accuracy)
                        dict_eval['p_macro'].append(p_macro)
                        dict_eval['r_macro'].append(r_macro)
                        dict_eval['f1_macro'].append(f1_macro)
                        dict_eval['p_micro'].append(p_micro)
                        dict_eval['r_micro'].append(r_micro)
                        dict_eval['f1_micro'].append(f1_micro)

                    write_report(dict_eval, result_data_path, name_imputation, missing, name_classification_algo)


if __name__ == "__main__":
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--from_id',
        help='start index to file list',
        default=0,
        type=int
    )
    parser.add_argument(
        '--to_id',
        help='end index to file list',
        default=len(file_list),
        type=int
    )
    parser.add_argument(
        '--fold_size',
        help='fold_size start from index 1',
        default=11,
        type=int
    )
    args = parser.parse_args()

    # Calls main function
    main(args) 
'''Code Finished'''
