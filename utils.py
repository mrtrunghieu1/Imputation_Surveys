'''Utility functions for GAIN.

(1)  normalization: MinMax Normalizer
(2)  renormalization: Recover the data from normalzied data
(3)  rounding: Handlecategorical variables after imputation
(4)  rmse_loss: Evaluate imputed data in terms of RMSE
(5)  xavier_init: Xavier initialization
(6)  binary_sampler: sample binary random variables
(7)  uniform_sampler: sample uniform random variables
(8)  sample_batch_index: sample random batch index
(9)  check_exist_folder: Check the directory exists or not
(10) csv_reader: Loader csv files 
(11) write_file: Save result to csv files 
(12) evaluation_report: Evaluate model via parameters: accuracy, p_macro, r_macro, f1_macro, p_micro, r_micro, f1_micro 
(13) write_report: Save classification results to json file
(14) mask_generation: create new mask
'''
 
# Necessary packages
import numpy as np
import tensorflow as tf
import os
from numpy import savetxt
import csv
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import json
import math
# My packages


'''Begin start code Python'''

#<1> 
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

#<2>
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

#<3>
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

#<4>
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

#<5>
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
      
#<6>
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

#<7>
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

#<8>
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
  
#<9>
def check_exist_folder(path):
  '''Check if the directory exists or not? 
  Args:
    - path: path to directory
  
  Returns:
    - if directory is not exists, the new folder was created 
  '''
  if not os.path.exists(path):
    os.makedirs(path)

#<10>
def csv_reader(save_folder, file_name, i, method, missingness):
  '''Loader csv files 
  Args:
    - save_folder: path to save directory
    - file_name: name of UCI datasets (ex: abalone, heart, tic-tac-toe)
    - i: index of fold_size
    - method: original data or missing data 
    - missingness: missingness constant
  
  Returns:
    - X_train: matrix train data
    - X_test:  matrix test data
  '''
  file_name_folder = os.path.join(save_folder, file_name)
  if method == 'original_data' and missingness == None:
  
      train_folder = os.path.join(file_name_folder, 'train/original_data')
      check_exist_folder(train_folder)
      test_folder = os.path.join(file_name_folder, 'test/original_data')
      check_exist_folder(test_folder)
      train_path = os.path.join(train_folder, 'train_{}.csv'.format(i))
      test_path = os.path.join(test_folder, 'test_{}.csv'.format(i))
  elif method == 'data_missing':
      train_folder = os.path.join(file_name_folder, 'train/train_{}'.format(i))
      test_folder = os.path.join(file_name_folder, 'test/test_{}'.format(i))
      train_path = os.path.join(train_folder, 'train_{}_missing_{}.csv'.format(i, missingness))
      test_path = os.path.join(test_folder, 'test_{}_missing_{}.csv'.format(i, missingness))
  # Loading train and test csv
  X_train = np.genfromtxt(train_path, delimiter=',')
  X_test = np.genfromtxt(test_path, delimiter=',')

  return X_train, X_test

#<11>
def write_file(data_train, data_test, save_folder, file_name, missingness, i):
  '''Save result to csv files 
  Args:
    - data_train: data train
    - data_test: data test
    - save_folder: path to save directory
    - file_name: name of UCI datasets (ex: abalone, heart, tic-tac-toe)
    - missingness: missingness constant
    - i: index of fold_size
    
  Returns:
    - Results were saved to save_folder
  '''  
  file_name_folder = os.path.join(save_folder, file_name)
  sub_train_path = os.path.join(file_name_folder, 'train/train_{}'.format(i))
  check_exist_folder(sub_train_path)
  sub_test_path = os.path.join(file_name_folder, 'test/test_{}'.format(i))
  check_exist_folder(sub_test_path)
  data_missing_train_path = os.path.join(sub_train_path, 'train_{}_missing_{}.csv'.format(i, missingness))
  data_missing_test_path = os.path.join(sub_test_path, 'test_{}_missing_{}.csv'.format(i, missingness))
  savetxt(data_missing_train_path, data_train, delimiter=',')
  savetxt(data_missing_test_path, data_test, delimiter=',')

#<12>
def evaluation_report(predict, grouth_truth):
  '''Evaluate model via parameters: accuracy, p_macro, r_macro, f1_macro, p_micro, r_micro, f1_micro 
  Args:
    - predict: predicted labels 
    - grouth_truth: grouth truth labels

  Returns:
    - accuracy, p_macro, r_macro, f1_macro, p_micro, r_micro, f1_micro
  '''  
  p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(grouth_truth, predict, average='macro')
  p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(grouth_truth, predict, average='micro')
  accuracy = accuracy_score(grouth_truth, predict)
  return accuracy, p_macro, r_macro, f1_macro, p_micro, r_micro, f1_micro

#<13>
def write_report(dict, save_folder, imp_name, missingness, classification_name):
  '''Save classification results to json file
  Args:
    - dict: dictionary contains evaluation parameters just caculated
    - save_folder: path to save directory
    - imp_name: name of imputation methods
    - missingness: missingness constant
    - classification_name: name of classification methods (SOTA)

  Returns:
    - Results were saved to save_folder
  ''' 
  file_name_folder = os.path.join(save_folder, imp_name)
  missing_folder = os.path.join(file_name_folder, 'data_missing_{}'.format(missingness))
  classification_folder = os.path.join(missing_folder, classification_name)
  check_exist_folder(classification_folder)

  name_metrics = ['accuracy', 'p_macro', 'r_macro', 'f1_macro', 'p_micro', 'r_micro', 'f1_micro']
  for name_metric in name_metrics:
      data = {'{}'.format(name_metric) : dict[name_metric], 'mean': np.mean(dict[name_metric]),
      'std': np.std(dict[name_metric])}
      path = os.path.join(classification_folder, '{}.txt'.format(name_metric))
      with open(path, 'w') as outfile:
          json.dump(data, outfile)    
        
#<14>
def mask_generation(missing_matrix):
  '''Create mask from missing data
  Args:
    - missing_matrix: data with value missing

  Returns:
    - (missing matrix ,mask)
  ''' 
  data_1d = missing_matrix.flatten()
  n_data = len(data_1d)
  mask_1d = np.ones(n_data)
  
  nan_id = [i for i, element in enumerate(data_1d) if math.isnan(element)]
  for i in nan_id:
      mask_1d[i] = 0
  
  mask = mask_1d.reshape(missing_matrix.shape)
  
  return missing_matrix, mask

#<15>
def encode_classes(col):
  """
  Args:  
    - col: categorical vector of any type

  Returns: 
    - labels: categorical vector of int in range 0-num_classes
  """
  classes = set(col)
  classes_dict = {c: i for i, c in enumerate(classes)}
  labels = np.array(list(map(classes_dict.get, col)), dtype=np.int32)
  return labels
  
#<16>
def data2onehot(data, mask, num_cols, cat_cols):
  """
  Args:
      - data: corrupted dataset
      - mask: mask of the corruption
      - num_cols: vector contaning indexes of columns having numerical values
      - cat_cols: vector contaning indexes of columns having categorical values

  Returns:
      - one-hot encoding of the dataset
      - one-hot encoding of the corruption mask
      - mask of the numerical entries of the one-hot dataset
      - mask of the categorical entries of the one-hot dataset
      - vector containing start-end idx for each categorical variable
  """
  # find most frequent class
  fill_with = []
  for col in cat_cols:
      l = list(data[:, col])
      fill_with.append(max(set(l), key=l.count))

  # meadian imputation
  filled_data = data.copy()
  for i, col in enumerate(cat_cols):
      filled_data[:, col] = np.where(mask[:, col], filled_data[:, col], fill_with[i])

  for i, col in enumerate(num_cols):
      filled_data[:, col] = np.where(
          mask[:, col], filled_data[:, col], np.nanmedian(data[:, col])
      )

  # encode into 0-N lables
  for col in cat_cols:
      filled_data[:, col] = encode_classes(filled_data[:, col])

  num_data = filled_data[:, num_cols]
  num_mask = mask[:, num_cols]
  cat_data = filled_data[:, cat_cols]
  cat_mask = mask[:, cat_cols]

  # onehot encoding for masks and categorical variables
  onehot_cat = []
  cat_masks = []
  for j in range(cat_data.shape[1]):
      col = cat_data[:, j].astype(int)
      col2onehot = np.zeros((col.size, col.max() + 1), dtype=float)
      col2onehot[np.arange(col.size), col] = 1
      mask2onehot = np.zeros((col.size, col.max() + 1), dtype=float)
      for i in range(cat_data.shape[0]):
          if cat_mask[i, j] > 0:
              mask2onehot[i, :] = 1
          else:
              mask2onehot[i, :] = 0
      onehot_cat.append(col2onehot)
      cat_masks.append(mask2onehot)

  cat_starting_col = []
  oh_data = num_data
  oh_mask = num_mask

  # build the big mask
  for i in range(len(onehot_cat)):
      cat_starting_col.append(oh_mask.shape[1])

      oh_data = np.c_[oh_data, onehot_cat[i]]
      oh_mask = np.c_[oh_mask, cat_masks[i]]

  oh_num_mask = np.zeros(oh_data.shape)
  oh_cat_mask = np.zeros(oh_data.shape)

  # build numerical mask
  oh_num_mask[:, range(num_data.shape[1])] = num_mask

  # build categorical mask
  oh_cat_cols = []
  for i in range(len(cat_masks)):
      start = cat_starting_col[i]
      finish = start + cat_masks[i].shape[1]
      oh_cat_mask[:, start:finish] = cat_masks[i]
      oh_cat_cols.append((start, finish))

  return oh_data, oh_mask, oh_num_mask, oh_cat_mask, oh_cat_cols

'''Code Finished'''
