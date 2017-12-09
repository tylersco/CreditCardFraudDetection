# Supervised Credit Card Fraud Detection
Data mining project exploring supervised techniques for detecting fraudulent credit card transactions.

# Project Structure
* `src/data/` contains `load_data.py` to load the dataset of credit card transactions
* `src/figures/` contains all plots and images generated from the code
* `src/results/` contains text files with metrics for all supervised classification models
  - Each text file contains metrics from 20 replications of running the models with the mean and standard deviation values
* `src/exec_traces/` contains text files with execution traces and output from running all of the Python files in the project
* `src/initial_analysis` contains all code associated with the intial analysis
* `src/supervised` contains all code associated with the supervised classification models
  - `src/supervised/` contains the logistic regression, naive Bayes, SVM, and decision tree models
  - `src/supervised/ensemble_methods` contains implementations of the random forest, bagging, and boosting models
  - `src/supervised/neural_nets` contains implementations for 2 versions of the neural network (v2 is the main neural net model)
* `src/classifier_results.pdf` contains the aggregated results from all of the classifiers
  - These are the results used in the paper
  
# Usage
To run any of the Python files:

`python3 [file.py] [path to creditcard.csv file]`

Example:

`python3 logistic_regression.py ~/Downloads/creditcard.csv`
