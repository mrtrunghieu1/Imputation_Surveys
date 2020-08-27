# Data-Imputation
>Imputation Methods for Missing Data and evaluate imputation methods via Sate of The Art algorithm classification (SOTA).

## Table of contents
* [Setup](#setup)
* [Imputation Methods](#imputation-methods)
* [Classification Algorithm](#classification-algorithm)

## Setup
* ### Note: when running project configure setup 2 review flag is False
>review_flag = False 

>review_imputaion_flag = False


## Imputation Methods
Add many imputation methods about project. List of some imputation methods:
* SimpleImputer(strategy = "mean")
* SimpleImputer(strategy = "median")
* SimpleImputer(strategy = "most_frequent")
* SimpleImputer(strategy = "constant")
* K-Nearest Neighbor Imputation
* Multiple Imputation by Chained Equations(MICE)
## Classification Algorithm
Some algorithms have been used to evaluate the accuracy of the methods imputed.
* KNeighborsClassifier(n_neighbors=10)
* RandomForestClassifier(n_estimators=200)
* XGBClassifier(n_estimators=200)
* DecisionTreeClassifier()


## Contact
Created by [@HieuVu](https://github.com/mrtrunghieu1) - feel free to contact me!

