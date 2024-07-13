import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

sys.stdout = open('/home/vicente/BDLab/NetCo/BioInformedSubjectRepresentation/model_output.txt', 'w')

CurrentSmokers_Networks = [
    "CurrentSmokersNetwork1-98",
    "CurrentSmokersNetwork2-35",
    "CurrentSmokersNetwork3-16",
    "CurrentSmokersNetwork4-12",
    "CurrentSmokersNetwork5-12",
]

# Creating a dictionary with the network name as key and the columns as the value stored in a list
network_columns = {}

for network in CurrentSmokers_Networks:
    network_columns[network] = []

path = "/home/vicente/BDLab/NetCo/BioInformedSubjectRepresentation/Current/"

for network in CurrentSmokers_Networks:
    df = pd.read_csv(f"{path}{network}.csv")
    network_columns[network].extend(df.columns.tolist())

current_smokers_omics = pd.read_csv(path + "current_smokers_omics_data.csv")
current_network_dfs = {}

for network, cols in network_columns.items():
    cols.remove('Unnamed: 0')
    current_network_dfs[network] = current_smokers_omics[cols]

# Now we have a dictionary(current_network_dfs) with all the networks as keys and a dataframe with the columns from the omics data set

# Target class stored in the gold csv file which is a single column
target_class = pd.read_csv(f"{path}current_smokers_gold.csv")

for index, row in target_class.iterrows():
    if row["finalgold_visit"] == 1 or row["finalgold_visit"] == 2:
        target_class.loc[index, "finalgold_visit"] = 1
    elif row["finalgold_visit"] == 3 or row["finalgold_visit"] == 4:
        target_class.loc[index, "finalgold_visit"] = 2


y = target_class['finalgold_visit'].values

# Grid Search: parameter tunning for logistic regression
param_grid_reg = {'C': [0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear', 'newton-cg', 'saga']}

# Grid Search: parameter tunning for random forest
param_grid_rf = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2],'bootstrap': [True, False]}
all_results_current_rf = {}
all_results_current_reg = {}

for network, data in current_network_dfs.items():
    X = data.values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

    # LOGISTIC REGRESSION
    grid_search_reg = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_reg, cv=5, scoring='accuracy')
    grid_search_reg.fit(X_train, y_train)

    y_pred_reg = grid_search_reg.best_estimator_.predict(X_test)
    #report1 = classification_report(y_test, y_pred_reg)

    #Run 500 times
    all_results_current_reg[network] = [[],[]]
    for i in range(500):
        #report2 = classification_report(y_test, y_pred_rf)
        accuracy2 = accuracy_score(y_test, y_pred_reg)
        f1_score2 = f1_score(y_test, y_pred_reg,average='macro')
        all_results_current_reg[network][0].append(accuracy2)
        all_results_current_reg[network][1].append(f1_score2) 

    #print(f"Logistic Regression Model for: {network}")
    #print(report1)
    #print("Best Parameters:", grid_search_reg.best_params_)

    # RANDOM FOREST
    grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, scoring='accuracy')
    
    grid_search_rf.fit(X_train, y_train)

    #Run 500 times
    y_pred_rf = grid_search_rf.best_estimator_.predict(X_test)
    #report2 = classification_report(y_test, y_pred_rf)

    #Run 500 times
    all_results_current_rf[network] = [[],[]]
    for i in range(500):
        #report2 = classification_report(y_test, y_pred_rf)
        accuracy2 = accuracy_score(y_test, y_pred_rf)
        f1_score2 = f1_score(y_test, y_pred_rf,average='macro')
        all_results_current_rf[network][0].append(accuracy2)
        all_results_current_rf[network][1].append(f1_score2)

    #print(f"Random Forest Model for: {network}")
    #print(report2)
    #print("Best Parameters:", grid_search_rf.best_params_)
    #print(network)


### FORMER SMOKER BASELINES ###

FormerSmokers_Networks = [
    "FormerSmokersNetwork1-88",
    "FormerSmokersNetwork2-16",
    "FormerSmokersNetwork3-24",
    "FormerSmokersNetwork4-24"
]


former_path = "/home/vicente/BDLab/NetCo/BioInformedSubjectRepresentation/Former/"
former_smoker_network = {}

for name in FormerSmokers_Networks:
    former_smoker_network[name] = []

for network in FormerSmokers_Networks:
    df = pd.read_csv(f"{former_path}{network}.csv")
    former_smoker_network[network].extend(df.columns.tolist())

former_smokers_omics = pd.read_csv(f"{former_path}former_smokers_omics_data.csv")
former_networks_dfs= {}

#Constructing a dictionary with the foermer smokers networks and the columns names taken from omics large dataset
for network, cols in former_smoker_network.items():
    cols.remove('Unnamed: 0')
    former_networks_dfs[network] = former_smokers_omics[cols]

# Now we have a dictionary(network_dfs) with all the networks as keys and a dataframe with the columns from the omics data set

# Target class stored in the gold csv file which is a single column
former_smokers_target = pd.read_csv(f"{former_path}former_smokers_gold.csv")

for index, row in former_smokers_target.iterrows():
    if row["finalgold_visit"] == 1 or row["finalgold_visit"] == 2:
        target_class.loc[index, "finalgold_visit"] = 1
    elif row["finalgold_visit"] == 3 or row["finalgold_visit"] == 4:
        target_class.loc[index, "finalgold_visit"] = 2

y = former_smokers_target['finalgold_visit'].values


# Grid Search: parameter tunning for logistic regression
param_grid_reg = {'C': [0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear', 'newton-cg', 'saga']}

# Grid Search: parameter tunning for random forest
param_grid_rf = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2],'bootstrap': [True, False]}

all_results_former_rf = {}
all_results_former_reg = {}

for network, data in former_networks_dfs.items():
    X = data.values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

    # LOGISTIC REGRESSION
    grid_search_reg = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_reg, cv=5, scoring='accuracy')
    grid_search_reg.fit(X_train, y_train)

    #Run 1000 times
    y_pred_reg = grid_search_reg.best_estimator_.predict(X_test)
    #report1 = classification_report(y_test, y_pred_reg)
 
    #Run 1000 times
    all_results_former_reg[network] = [[],[]]
    for i in range(1000):
        #report2 = classification_report(y_test, y_pred_rf)
        accuracy2 = accuracy_score(y_test, y_pred_reg)
        f1_score2 = f1_score(y_test, y_pred_reg,average='macro')
        all_results_former_reg[network][0].append(accuracy2)
        all_results_former_reg[network][1].append(f1_score2) 


    #print(f"Logistic Regression Model for: {network}")
    #print(report1)
    #print("Best Parameters:", grid_search_reg.best_params_)

    # RANDOM FOREST
    grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, scoring='accuracy')
    grid_search_rf.fit(X_train, y_train)
    y_pred_rf = grid_search_rf.best_estimator_.predict(X_test)

    #Run 1000 times
    all_results_former_rf[network] = [[],[]]
    for i in range(1000):
        #report2 = classification_report(y_test, y_pred_rf)
        accuracy2 = accuracy_score(y_test, y_pred_rf)
        f1_score2 = f1_score(y_test, y_pred_rf,average='macro')
        all_results_former_rf[network][0].append(accuracy2)
        all_results_former_rf[network][1].append(f1_score2)

    #print(f"Random Forest Model for: {network}")
    #print(report2)
    #print("Best Parameters:", grid_search_rf.best_params_)
    #print(network)


# Calculate avg accuracy, avg f1-Score and std

#500
# all_results_current_rf 
# all_results_current_reg

for key, value in all_results_current_rf.items():
    accuracy = np.array(value[0])
    f1score = np.array(value[1])
    total_accuracy = np.average(accuracy)
    total_f1score = np.average(f1score)
    std_acc = np.std(accuracy)
    std_f1 = np.std(f1score)

    print("Random Forest Results for Current Smokers")
    print(f"For Netkwork: {key}\nAccuracy avg: {total_accuracy} - Std: {std_acc}\nF1-Score: {total_f1score} - Std:{std_f1}\n")

for key, value in all_results_current_reg.items():
    accuracy = np.array(value[0])
    f1score = np.array(value[1])
    total_accuracy = np.average(accuracy)
    total_f1score = np.average(f1score)
    std_acc = np.std(accuracy)
    std_f1 = np.std(f1score)

    print("Logistic Regression for Current Smokers")
    print(f"For Netkwork: {key}\nAccuracy avg: {total_accuracy} - Std: {std_acc}\nF1-Score: {total_f1score} - Std:{std_f1}\n")
import sklearn.datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import numpy as np
import pandas as pd
import os
import re


def load_data(args):
    dataset_dir = args.dataset_dir
    # Define maximum number of features (this could also be determined dynamically)

    if dataset_dir is None:
        raise ValueError('No Dataset Directory Specified!')

    omics_dataset_name = ''
    phenotype_file_name = ''
    for _, _, files in os.walk(dataset_dir):
        for file in files:
            if re.compile('.*omics_data\.csv').match(file):
                omics_dataset_name = file
            elif re.compile('.*gold\.csv').match(file):
                phenotype_file_name = file
    if omics_dataset_name == '' or phenotype_file_name == '':
        raise FileNotFoundError

    omics_dataset = pd.read_csv(os.path.join(dataset_dir, omics_dataset_name))
    phenotype = pd.read_csv(os.path.join(dataset_dir,
                                         phenotype_file_name))
    omics_dataset_complete = omics_dataset.merge(phenotype, how='left', left_index=True, right_index=True)
    # Combining Classes for Less Severity in Balancing Classes
    omics_dataset_complete['finalgold_visit'] = np.where(omics_dataset_complete['finalgold_visit'] == 2, 1,
                                                         omics_dataset_complete['finalgold_visit'])
    omics_dataset_complete['finalgold_visit'] = np.where(omics_dataset_complete['finalgold_visit'] == 3, 2,
                                                         omics_dataset_complete['finalgold_visit'])
    omics_dataset_complete['finalgold_visit'] = np.where(omics_dataset_complete['finalgold_visit'] == 4, 2,
                                                         omics_dataset_complete['finalgold_visit'])
    # Replacing PRISM (-1) with 5 for Prediction to Work!
    omics_dataset_complete['finalgold_visit'] = np.where(omics_dataset_complete['finalgold_visit'] == -1, 3,
                                                         omics_dataset_complete['finalgold_visit'])

    dataset_dir = args.dataset_dir
    omics_networks_l = []
    omics_datasets = []
    for dir_name, _, files in os.walk(dataset_dir):
        for file in files:
            if re.compile('.*Network\d+-\d+\.csv').match(file):
                omics_networks_l.append(os.path.join(dir_name, file))

    if not omics_networks_l:
        raise FileNotFoundError

    # After loading each dataset:
    for omics_network_f in omics_networks_l:
        omics_network_adj = pd.read_csv(omics_network_f, index_col=0)
        omics_network_nodes_names = omics_network_adj.index.tolist()
        dataset = omics_dataset_complete[omics_network_nodes_names + ['finalgold_visit']]

        # Padding features if fewer than max_features
        num_features = dataset.shape[1] - 1  # exclude label column
        if num_features < max_features:
            # Create a DataFrame of zeros
            padding = pd.DataFrame(0, index=np.arange(len(dataset)), columns=np.arange(max_features - num_features))
            dataset = pd.concat([dataset.drop(['finalgold_visit'], axis=1), padding, dataset['finalgold_visit']], axis=1)

        omics_datasets.append(dataset)

    return omics_datasets



#1000
# all_results_former_rf 
# all_results_former_reg 


for key, value in all_results_former_rf.items():
    accuracy = np.array(value[0])
    f1score = np.array(value[1])
    total_accuracy = np.average(accuracy)
    total_f1score = np.average(f1score)
    std_acc = np.std(accuracy)
    std_f1 = np.std(f1score)

    print("Random Forest Results for Former Smokers")
    print(f"For Netkwork: {key}\nAccuracy avg: {total_accuracy} - Std: {std_acc}\nF1-Score: {total_f1score} - Std:{std_f1}\n")

for key, value in all_results_former_reg.items():
    accuracy = np.array(value[0])
    f1score = np.array(value[1])
    total_accuracy = np.average(accuracy)
    total_f1score = np.average(f1score)
    std_acc = np.std(accuracy)
    std_f1 = np.std(f1score)

    print("Logistic Regression Results for Former Smokers")
    print(f"For Netkwork: {key}\nAccuracy avg: {total_accuracy} - Std: {std_acc}\nF1-Score: {total_f1score} - Std:{std_f1}\n")


sys.stdout.close()