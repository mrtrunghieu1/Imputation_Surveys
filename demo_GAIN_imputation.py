'''Main function impute with Generative Adversarial Imputation Networks (GAIN)'''

# Necessary packages
import sys

sys.path.append("..")
import datetime
import csv
import numpy as np
import os
import argparse

# My packages
from data_helper import file_list, data_K_Fold, imputed_dataset
from gain import gain
from utils import csv_reader, write_file

'''Begin start code Python'''


def main(args):
    '''Main function for imputed with GAIN

    Args:
        - from_id: start index to file list
        - to_id: end index to file list
        - fold_size: fold_size start from index 1 
        - miss_rate: probability of missing components
        - batch_size: batch size
        - hint_rate: hint rate
        - alpha: hyperparamenter
        - iterations: iterations

    Returns:
        - write imputed_data: imputed data
    '''

    # Input parameters
    from_id = args.from_id
    to_id = args.to_id
    fold_size = args.fold_size

    gain_parameters = {'batch_size': args.batch_size,
                       'hint_rate': args.hint_rate,
                       'alpha': args.alpha,
                       'iterations': args.iterations}

    # Initial parameter 
    missingness_flag = [0, 10, 20, 30, 40, 50]  # t% missing data  

    # Data missing loader
    for i_file in range(from_id, to_id):
        file_name = file_list[i_file]
        print(datetime.datetime.now(), "File {}: {}".format(i_file, file_name))
        for i in range(1, fold_size):
            for missingness in missingness_flag:
                (D_miss_train, D_miss_test) = csv_reader(data_K_Fold, file_name, i, method='data_missing',
                                                         missingness=missingness)
                # Impute missing data
                imputed_train_D = gain(D_miss_train, gain_parameters)
                imputed_test_D = gain(D_miss_test, gain_parameters)
                imputed_path = os.path.join(imputed_dataset, file_name)
                write_file(imputed_train_D, imputed_test_D, imputed_path, 'GAIN', missingness, i)


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
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch',
        default=128,
        type=int)
    parser.add_argument(
        '--hint_rate',
        help='hint probability',
        default=0.9,
        type=float)
    parser.add_argument(
        '--alpha',
        help='hyperparameter',
        default=100,
        type=float)
    parser.add_argument(
        '--iterations',
        help='number of training interations',
        default=10000,
        type=int)

    args = parser.parse_args()

    # Calls main function
    main(args)
'''Code Finished'''
