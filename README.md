# Data-Imputation
>Imputation Methods for Missing Data and evaluate imputation methods via sate-of-the-art algorithm classification (SOTA).

## Table of contents
  - [Enviroments](#Enviroments)
  - [Example command](#example-command)
  - [Imputation Methods](#imputation-methods)
  - [Classification Algorithm](#classification-algorithm)
  - [Contact](#contact)

## Enviroments
* Python 3.6
* tensorflow==1.14
* pytorch
* pip install dgl==0.4.3

## Example command
* Create new datasets with different missing rates or imputed value
```bash
$ python demo_data_imputation.py --from_id 0 --to_id 2 --review_missing_flag True 
--review_imputed_flag True
```
* Using Generative Adversarial Imputation Networks method for imputation
```bash
$ python demo_GAIN_imputation.py --from_id 0 --to_id 2 --fold_size 11 --batch_size 32 --hint_rate 0.9 --alpha 100 --iterations 10000
```
* Using Graph Imputation Neural Networks method for imputation
```bash
$ python demo_GINN_imputation.py
```
## Imputation Methods
Add many imputation methods about project. List of some imputation methods:
* SimpleImputer(strategy = "mean")
* SimpleImputer(strategy = "median")
* SimpleImputer(strategy = "most_frequent")
* SimpleImputer(strategy = "constant")
* K-Nearest Neighbor Imputation
* Multiple Imputation by Chained Equations(MICE)
* Generative Adversarial Imputation Networks (GAIN)
* Graph Imputation Neural Networks (GINN)
## Classification Algorithm
Some algorithms have been used to evaluate the accuracy of the methods imputed.
* KNeighborsClassifier(n_neighbors=10)
* RandomForestClassifier(n_estimators=200)
* XGBClassifier(n_estimators=200)
* DecisionTreeClassifier()

## Contact
Created by [@HieuVu](https://github.com/mrtrunghieu1) - feel free to contact me!

