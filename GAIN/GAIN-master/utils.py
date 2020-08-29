'''Utility functions for GAIN.

(1) normalization: MinMax Normalizer
(2) renormalization: Recover the data from normalzied data
(3) rounding: Handlecategorical variables after imputation
(4) rmse_loss: Evaluate imputed data in terms of RMSE
(5) xavier_init: Xavier initialization
(6) binary_sampler: sample binary random variables
(7) uniform_sampler: sample uniform random variables
(8) sample_batch_index: sample random batch index
'''
 
# Necessary packages
import numpy as np
import tensorflow as tf
import os
from numpy import savetxt

def normalization (data, parameters=None):
  '''Normalize data in [0, 1] range.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  '''

  # Parameters
  _, dim = data.shape
  norm_data = data.copy()
  
  if parameters is None:
  
    # MixMax normalization
    min_val = np.zeros(dim)
    max_val = np.zeros(dim)
    
    # For each dimension
    for i in range(dim):
      min_val[i] = np.nanmin(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
      max_val[i] = np.nanmax(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   
      
    # Return norm_parameters for renormalization
    norm_parameters = {'min_val': min_val,
                       'max_val': max_val}

  else:
    min_val = parameters['min_val']
    max_val = parameters['max_val']
    
    # For each dimension
    for i in range(dim):
      norm_data[:,i] = norm_data[:,i] - min_val[i]
      norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)  
      
    norm_parameters = parameters    
      
  return norm_data, norm_parameters


def renormalization (norm_data, norm_parameters):
  '''Renormalize data from [0, 1] range to the original range.
  
  Args:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  
  Returns:
    - renorm_data: renormalized original data
  '''
  
  min_val = norm_parameters['min_val']
  max_val = norm_parameters['max_val']

  _, dim = norm_data.shape
  renorm_data = norm_data.copy()
    
  for i in range(dim):
    renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
    renorm_data[:,i] = renorm_data[:,i] + min_val[i]
    
  return renorm_data


def rounding (imputed_data, data_x):
  '''Round imputed data for categorical variables.
  
  Args:
    - imputed_data: imputed data
    - data_x: original data with missing values
    
  Returns:
    - rounded_data: rounded imputed data
  '''
  
  _, dim = data_x.shape
  rounded_data = imputed_data.copy()
  
  for i in range(dim):
    temp = data_x[~np.isnan(data_x[:, i]), i]
    # Only for the categorical variable
    if len(np.unique(temp)) < 20:
      rounded_data[:, i] = np.round(rounded_data[:, i])
      
  return rounded_data


def rmse_loss (ori_data, imputed_data, data_m):
  '''Compute RMSE loss between ori_data and imputed_data
  
  Args:
    - ori_data: original data without missing values
    - imputed_data: imputed data
    - data_m: indicator matrix for missingness
    
  Returns:
    - rmse: Root Mean Squared Error
  '''
  
  ori_data, norm_parameters = normalization(ori_data)
  imputed_data, _ = normalization(imputed_data, norm_parameters)
    
  # Only for missing values
  nominator = np.sum(((1-data_m) * ori_data - (1-data_m) * imputed_data)**2)
  denominator = np.sum(1-data_m)
  
  rmse = np.sqrt(nominator/float(denominator))
  
  return rmse


def xavier_init(size):
  '''Xavier initialization.
  
  Args:
    - size: vector size
    
  Returns:
    - initialized random vector.
  '''
  in_dim = size[0]
  xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
  return tf.random_normal(shape = size, stddev = xavier_stddev)
      

def binary_sampler(p, rows, cols):
  '''Sample binary random variables.
  
  Args:
    - p: probability of 1
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - binary_random_matrix: generated binary random matrix.
  '''
  unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
  binary_random_matrix = 1*(unif_random_matrix < p)
  return binary_random_matrix


def uniform_sampler(low, high, rows, cols):
  '''Sample uniform random variables.
  
  Args:
    - low: low limit
    - high: high limit
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - uniform_random_matrix: generated uniform random matrix.
  '''
  return np.random.uniform(low, high, size = [rows, cols])       


def sample_batch_index(total, batch_size):
  '''Sample index of the mini-batch.
  
  Args:
    - total: total number of samples
    - batch_size: batch size
    
  Returns:
    - batch_idx: batch index
  '''
  total_idx = np.random.permutation(total)
  batch_idx = total_idx[:batch_size]
  return batch_idx
  

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