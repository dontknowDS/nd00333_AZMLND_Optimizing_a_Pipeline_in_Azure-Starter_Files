# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
This dataset contains data about people and features about them. We seek to predict column "y" which is binary. 

The best performing model was a xgboostclassifier with an accuracy of 0.918.

## Scikit-learn Pipeline
The data contained data about people. The target variable y was fitted via logistic regression in training.

The used hyperparameters were "C" and "max_iter", the first being a regularization parameter and the former being used for convergence.
I used hyperparameter tuning, which is also called hyperparameter optimization. It is the process of finding the configuration of hyperparameters that result in the best performance given a metric. There are several strategies for hyperparameter tuning. 
For this project we used Random Search as Parameter sampling strategy with a given search space for the parameters 'C' and 'max_iter'. 
Random Search is a technique where random combinations of the given hyperparameter search spaces are tested in order to find the best solution for the built model.
I decided to use Random Search, as it requires less experiments and thus less computation time than a grid search across the same search space while yielding good results.

I also implemented an early stopping policy, so the runtime doesn't get too long and also to avoid overfitting. 
I chose Bandit policy for this, which stops the algorithm when the difference between the current run's metric and the best run's metric is less than the 'slack_factor', which is a given argument which I chose to be 0.1. The frequency in which this policy is apllied was chosen as 1, meaning every run. This way my goals to avoid overfitting and avoid long training time with no improvements have been fulfilled.


The resulting Accuracy was 0.911


## AutoML
For the AutoML configuration I chose the following parameters:
![image](https://user-images.githubusercontent.com/77688775/211331170-2169b6a4-4a1f-4fc7-967d-ac41c0fcbdfd.png)
The timeout defines how long the experiment continues to run. I chose 30 Minutes, as I expected good results for this task in this time frame.
I chose 3 as the number of folds for the Cross-validation, that is 3 different pairs of subsets for training and validation data, so we get meaningful accuracy results about our algorithm performance. 


The best performing model was a xgboostclassifier with an accuracy of 0.918, see the metrics from the best run:
![image](https://user-images.githubusercontent.com/77688775/211310022-a2c021b6-4bee-49f8-8689-fa7191657102.png)
For the classification task I consiered accuracy as the most important metric, which is why I chose it as my primary metric.


## Pipeline comparison
The xgboostclassifier outperformed the LogisticRegression slightly. Reasons might be the model has both linear model solver and tree learning algorithms
by training multiple decision trees and then combines the results.
Advantage of the Scikit-learn pipeline was a considerable shorter training time, which was especially short due to my hyperparameter serach and early termination strategy.

## Future work
To further improve the results the autoML training limit of 30minutes could be increased. As for the Scikit-learn pipeline, 
a bigger search space and bayesian search strategy for the hyperparemters could be used to improve performance.

## Proof of cluster clean up
![image](https://user-images.githubusercontent.com/77688775/211310965-a2c67a6e-3ecf-4353-ae66-b08e06262968.png)
**Image of cluster marked for deletion**
