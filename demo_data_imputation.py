'''Main function to prepare missing dataset'''



# Necessary packages
import os
import sys
import datetime
import numpy as np
from sklearn.model_selection import KFold
from numpy import savetxt
import csv
import random
import argparse
# My packages
from data_helper import file_list, data_folder, data_K_Fold, imputed_dataset
from util import csv_reader, write_file
from data_processing import K_Fold_cross_validation, missing_data_generation
from data_imputation import imputation_method
from train import model_prediction
'''Begin start code Python'''
# try:
#     from_id = int(sys.args[0])
#     to_id = int(sys.args[1])
# except:
#     from_id = 0
#     to_id = len(file_list)
def main(args):
    '''Main function for prepare processing data
    
    Args:
        - from_id: start index to file list
        - to_id: end index to file list
        - review_missing_flag: Set flag is True create missing data
        - review_imputed_flag: Set flag is True imputed missing value

    Returns:
        - Write file missing data
        - Write file imputed values 
    '''
    # Flag
    review_missing_flag = args.review_missing_flag
    review_imputed_flag = args.review_imputed_flag

    # Parameters
    from_id = args.from_id 
    to_id = args.to_id

    fold_size = 11
    random.seed(0)
    missingness_flag = [0, 10, 20, 30, 40, 50]  # t% missing data  
    binary_flag = [0,0,0,0,1,0]          # 1 activate imputation algorithm
    imputation_flag = [i for i, impf in enumerate(binary_flag) if impf == 1]

    # Load data and introduce missingness
    for i_file in range(from_id, to_id):
        file_name = file_list[i_file]
        print(datetime.datetime.now(), "File {}: {}".format(i_file, file_name))
        # Data Processing
        if review_missing_flag:
            # Data loader
            D_train = np.loadtxt(data_folder + '/train1/' + file_name + '_train1.dat', delimiter=',')
            D_val = np.loadtxt(data_folder + '/val/' + file_name + '_val.dat', delimiter=',')
            D_test = np.loadtxt(data_folder + '/test/' + file_name + '_test.dat', delimiter=',')

            X_full = np.concatenate((D_train, D_val, D_test), axis = 0)

            # K-Fold Cross Validation approach first time
            kf_1 = KFold(n_splits = 5, shuffle = True)
            kf_1.split(X_full)
            # K-Fold Cross Validation approach second time
            kf_2 = KFold(n_splits = 5, shuffle = True)
            kf_2.split(X_full)
            # Save file csv train(i)-test(i) i=<1,5>
            K_Fold_cross_validation(kf_1, X_full, data_K_Fold, file_name, 0)
            # Save file csv train(i)-test(i) i=<5,10>
            K_Fold_cross_validation(kf_2, X_full, data_K_Fold, file_name, 5)

            # Loading data K-Fold 
            for i in range(1, fold_size):
                (D_train, D_test) = csv_reader(data_K_Fold, file_name, i, method='original_data', missingness=None)
                for missingness in missingness_flag:
                    D_train_missing = missing_data_generation(D_train, missingness)
                    D_test_missing = missing_data_generation(D_test, missingness)
                    write_file(D_train_missing, D_test_missing, data_K_Fold, file_name, missingness, i)
            

        # Loading data processed and imputed dataset
        if review_imputed_flag:
            for i in range(1, fold_size):
                for missingness in missingness_flag:
                    (D_missing_train, D_missing_test) = csv_reader(data_K_Fold, file_name, i, method='data_missing', missingness=missingness)
                    for imp_flag in imputation_flag:
                        imputed_train, imputed_test, imp_name = imputation_method(D_missing_train, D_missing_test, imp_flag, missingness)
                        imputation_path = os.path.join(file_name, imp_name)
                        write_file(imputed_train, imputed_test, imputed_dataset, imputation_path, missingness, i)

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
        '--review_missing_flag',
        help='Set flag is True create missing data',
        default=False,
        type=bool
    )
    parser.add_argument(
        '--review_imputed_flag',
        help='Set flag is True imputed missing value',
        default=False,
        type=bool
    )

    args = parser.parse_args()
    
    # Call main function
    main(args)
# '''Code Finished'''

