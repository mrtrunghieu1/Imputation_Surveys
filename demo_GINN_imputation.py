'''Main function impute with Graph Imputation Neural Networks (GINN)'''

#Necessary packages
import datetime
import csv
import numpy as np
from sklearn import model_selection, preprocessing
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
#My packages
from data_helper import file_list, data_K_Fold, dictionary_datasets, imputed_dataset
from utils import csv_reader, mask_generation, data2onehot, write_file      
from GINN.ginn.core import GINN

'''Begin start code Python'''
def main(args):
    '''Main function for imputed with GINN

    Args:
        - from_id: start index to file list
        - to_id: end index to file list
        - fold_size: fold_size start from index 1 
    Returns:
        - write imputed_data: imputed data
    '''

    #Input parameters
    from_id = args.from_id 
    to_id = args.to_id
    fold_size = args.fold_size

    #Initial parameters
    missingness_flag = [0, 10, 20, 30, 40, 50]  # t% missing data  
    seed = 42

    #Main program 
    for i_file in range(from_id, to_id):
        file_name = file_list[i_file]
        print(datetime.datetime.now(), "File {}: {}".format(i_file, file_name))
        for i in tqdm(range(1, fold_size)):
            for missingness in missingness_flag:
                (D_miss_train, D_miss_test) = csv_reader(data_K_Fold, file_name, i, method='data_missing', missingness=missingness)
                x_train = D_miss_train[:, :(D_miss_train.shape[1] - 1)]
                y_train = D_miss_train[:,-1]
                x_test = D_miss_test[:, :(D_miss_test.shape[1] - 1)]
                y_test = D_miss_test[:,-1]

                missing_train, missing_train_mask = mask_generation(x_train)
                missing_test, missing_test_mask = mask_generation(x_test)

                cx_train = np.c_[missing_train, y_train]
                cx_test = np.c_[missing_test, y_test]
                
                mask_train = np.c_[missing_train_mask, np.ones(y_train.shape)]
                mask_test = np.c_[missing_test_mask, np.ones(y_test.shape)]


# Here we proprecess the data applying a one-hot encoding for the categorical variables. We get the encoded dataset 
# three different masks that indicates the missing features and if these features are categorical or numerical,
#  plus the new columns for the categorical variables with their one-hot range.
                numerical_columns = dictionary_datasets['{}'.format(file_name)]['numerical']  
                categorical_columns = dictionary_datasets['{}'.format(file_name)]['categorical']  
                [oh_data, oh_mask, oh_numerical_mask, oh_categorical_mask, oh_categorical_columns] = data2onehot(np.r_[cx_train, cx_test], np.r_[mask_train, mask_test], numerical_columns, categorical_columns)

#We scale the features with a min max scaler that will preserve the one-hot encoding
                oh_data_train = oh_data[ :x_train.shape[0], :]
                oh_data_test = oh_data[x_train.shape[0]: , :]

                oh_mask_train = oh_mask[ :x_train.shape[0], :]
                oh_num_mask_train = oh_mask[ :x_train.shape[0], :]
                oh_cat_mask_train = oh_mask[ :x_train.shape[0], :]

                oh_mask_test = oh_mask[x_train.shape[0]: , :]
                oh_num_mask_test = oh_mask[x_train.shape[0]: , :]
                oh_cat_mask_test = oh_mask[x_train.shape[0]: , :]
                
                # Scaler
                scaler_train = preprocessing.MinMaxScaler()
                oh_data_train = scaler_train.fit_transform(oh_data_train)

                scaler_test = preprocessing.MinMaxScaler()
                oh_data_test = scaler_test.fit_transform(oh_data_test)

#Now we are ready to impute the missing values on the training set!
                imputer_train = GINN(
                    oh_data_train,
                    oh_mask_train,
                    oh_num_mask_train,
                    oh_cat_mask_train,
                    oh_categorical_columns,
                    numerical_columns,
                    categorical_columns
                )
#Now we are ready to impute the missing values on the testing set!
                imputer_test = GINN(
                    oh_data_test,
                    oh_mask_test,
                    oh_num_mask_test,
                    oh_cat_mask_test,
                    oh_categorical_columns,
                    numerical_columns,
                    categorical_columns
                )
                # Transform
                imputer_train.fit()
                imputed_train = scaler_train.inverse_transform(imputer_train.transform())
                imputer_test.fit()
                imputed_test = scaler_test.inverse_transform(imputer_test.transform())
                # Write result
                imputed_path = os.path.join(imputed_dataset, file_name)
                write_file(imputed_train, imputed_test, imputed_path, 'GINN', missingness, i)

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