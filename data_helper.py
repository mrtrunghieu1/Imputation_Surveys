file_list = ['heart', 'tic-tac-toe', 'abalone']

data_folder = "/Users/AnhVu/Study/Machine_learning/Data/Tran_Val_Test"

data_K_Fold = "data_K_Fold"

imputed_dataset = "imputed_dataset"

result_path = "result"

# data_folder = "C:\\Users\\DELL\\Desktop\\Research\\Imputation_Surveys\\data"
#
# data_K_Fold = "C:\\Users\\DELL\\Desktop\\Research\\Imputation_Surveys\\data_K_Fold"
#
# imputed_dataset = "C:\\Users\\DELL\\Desktop\\Research\\Imputation_Surveys\\imputed_dataset"
#
# result_path = "C:\\Users\\DELL\\Desktop\\Research\\Imputation_Surveys\\result"

dictionary_datasets = {
    'abalone': {'name': 'abalone', 'numerical': [0, 1, 2, 3, 4, 5, 6, 7], 'categorical': [8]},
    'heart': {'name': 'heart', 'numerical': [0, 3, 4, 7, 9], 'categorical': [1, 2, 5, 6, 8, 10, 11, 12, 13]},
    'tic-tac-toe': {'name': 'tic-tac-toe', 'numerical': [], 'categorical': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
}
