# Library
import sys

sys.path.append("..")
import datetime
import csv
import numpy as np
from sklearn import model_selection, preprocessing
# My Library
from GINN.ginn import GINN
from ginn.utils import degrade_dataset, data2onehot
from util import csv_reader
from data_helper import file_list, data_folder, data_K_Fold, imputed_dataset

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
cat_cols = [1, 2, 5, 6, 8, 10, 11, 12, 13]
num_cols = [0, 3, 4, 7, 9]
# Main program
for i_file in range(from_id, to_id):
    file_name = file_list[i_file]
    print(datetime.datetime.now(), "File {}: {}".format(i_file, file_name))

    for i in range(1, fold_size):
        (D_train, D_test) = csv_reader(data_K_Fold, file_name, i, method='original_data', missingness=None)
        x_train = D_train[:, :(D_train.shape[1] - 1)]
        y_train = D_train[:, -1]
        x_test = D_test[:, :(D_test.shape[1] - 1)]
        y_test = D_test[:, -1]
        for missingness in missingness_flag:
            missingness /= 100
            cx_train, cx_train_mask = degrade_dataset(x_train, missingness, seed, np.nan)
            cx_test, cx_test_mask = degrade_dataset(x_test, missingness, seed, np.nan)

            cx_tr = np.c_[cx_train, y_train]
            cx_te = np.c_[cx_test, y_test]

            mask_tr = np.c_[cx_train_mask, np.ones(y_train.shape)]
            mask_te = np.c_[cx_test_mask, np.ones(y_test.shape)]
            # Here we proprecess the data applying a one-hot encoding for the categorical variables. We get the encoded dataset three different
            # masks that indicates the missing features and if these features are categorical or numerical, plus the new columns for the categorical variables with their one-hot range.
            [oh_x, oh_mask, oh_num_mask, oh_cat_mask, oh_cat_cols] = data2onehot(np.r_[cx_tr, cx_te],
                                                                                 np.r_[mask_tr, mask_te], num_cols,
                                                                                 cat_cols)

            # We scale the features with a min max scaler that will preserve the one-hot encoding
            oh_x_tr = oh_x[:x_train.shape[0], :]
            oh_x_te = oh_x[x_train.shape[0]:, :]

            oh_mask_tr = oh_mask[:x_train.shape[0], :]
            oh_num_mask_tr = oh_mask[:x_train.shape[0], :]
            oh_cat_mask_tr = oh_mask[:x_train.shape[0], :]

            oh_mask_te = oh_mask[x_train.shape[0]:, :]
            oh_num_mask_te = oh_mask[x_train.shape[0]:, :]
            oh_cat_mask_te = oh_mask[x_train.shape[0]:, :]

            scaler_tr = preprocessing.MinMaxScaler()
            oh_x_tr = scaler_tr.fit_transform(oh_x_tr)

            scaler_te = preprocessing.MinMaxScaler()
            oh_x_te = scaler_te.fit_transform(oh_x_te)

            # Now we are ready to impute the missing values on the training set!
            imputer = GINN(oh_x_tr,
                           oh_mask_tr,
                           oh_num_mask_tr,
                           oh_cat_mask_tr,
                           oh_cat_cols,
                           num_cols,
                           cat_cols
                           )

            imputer.fit()
            imputed_tr = scaler_tr.inverse_transform(imputer.transform())

            break

'''Code Finished'''
