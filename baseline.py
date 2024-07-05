import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

#sys.stdout = open('C:/Users/ramosv/Desktop/NetCo/BioInformedSubjectRepresentation/model_output.txt', 'w')

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

path = "C:/Users/ramosv/Desktop/NetCo/BioInformedSubjectRepresentation/Current/"

for network in CurrentSmokers_Networks:
    df = pd.read_csv(f"{path}{network}{".csv"}")
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
param_grid_reg = {'C': [0.1, 1, 10, 100], 'solver': ['liblinear', 'saga']}

# Grid Search: parameter tunning for random forest
param_grid_rf = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2],'bootstrap': [True, False]}


for network, data in current_network_dfs.items():
    X = data.values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

    # LOGISTIC REGRESSION
    grid_search_reg = GridSearchCV(LogisticRegression(max_iter=1000, class_weight='balanced'), param_grid_reg, cv=5, scoring='accuracy')
    grid_search_reg.fit(X_train, y_train)

    y_pred_reg = grid_search_reg.best_estimator_.predict(X_test)
    report1 = classification_report(y_test, y_pred_reg)

    print(f"Logistic Regression Model for: {network}")
    print(report1)
    #print("Best Parameters:", grid_search_reg.best_params_)

    # RANDOM FOREST
    grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42,class_weight='balanced'), param_grid_rf, cv=5, scoring='accuracy')
    grid_search_rf.fit(X_train, y_train)

    y_pred_rf = grid_search_rf.best_estimator_.predict(X_test)
    report2 = classification_report(y_test, y_pred_rf)

    print(f"Random Forest Model for: {network}")
    print(report2)
    #print("Best Parameters:", grid_search_rf.best_params_)


### FORMER SMOKER BASELINES ###

FormerSmokers_Networks = [
    "FormerSmokersNetwork1-88",
    "FormerSmokersNetwork2-16",
    "FormerSmokersNetwork3-24",
    "FormerSmokersNetwork4-24"
]


former_path = "C:/Users/ramosv/Desktop/NetCo/BioInformedSubjectRepresentation/Former/"
former_smoker_network = {}

for name in FormerSmokers_Networks:
    former_smoker_network[name] = []

for network in FormerSmokers_Networks:
    df = pd.read_csv(f"{former_path}{network}{".csv"}")
    former_smoker_network[network].extend(df.columns.tolist())

former_smokers_omics = pd.read_csv(f"{former_path}former_smokers_omics_data.csv")
former_networks_dfs= {}

#Constructing a dictionary with the foermer smokers networks and the columns names taken from omics large dataset
for network, cols in former_smoker_network.items():
    cols.remove('Unnamed: 0')
    former_networks_dfs[network] = former_smokers_omics[cols]

# Now we have a dictionary(network_dfs) with all the networks as keys and a dataframe with the columns from the omics data set

# Grid Search: parameter tunning for logistic regression
param_grid_reg = {'C': [0.1, 1, 10, 100], 'solver': ['liblinear', 'saga']}

# Grid Search: parameter tunning for random forest
param_grid_rf = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}

# Target class stored in the gold csv file which is a single column
former_smokers_target = pd.read_csv(f"{former_path}{"former_smokers_gold.csv"}")

for index, row in former_smokers_target.iterrows():
    if row["finalgold_visit"] == 1 or row["finalgold_visit"] == 2:
        target_class.loc[index, "finalgold_visit"] = 1
    elif row["finalgold_visit"] == 3 or row["finalgold_visit"] == 4:
        target_class.loc[index, "finalgold_visit"] = 2

y = former_smokers_target['finalgold_visit'].values

for network, data in former_networks_dfs.items():
    X = data.values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

    # LOGISTIC REGRESSION
    grid_search_reg = GridSearchCV(LogisticRegression(max_iter=1000, class_weight='balanced'), param_grid_reg, cv=5, scoring='accuracy')
    grid_search_reg.fit(X_train, y_train)

    y_pred_reg = grid_search_reg.best_estimator_.predict(X_test)
    report1 = classification_report(y_test, y_pred_reg)

    print(f"Logistic Regression Model for: {network}")
    print(report1)
    #print("Best Parameters:", grid_search_reg.best_params_)

    # RANDOM FOREST
    grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42,class_weight='balanced'), param_grid_rf, cv=5, scoring='accuracy')
    grid_search_rf.fit(X_train, y_train)

    y_pred_rf = grid_search_rf.best_estimator_.predict(X_test)
    report2 = classification_report(y_test, y_pred_rf)

    print(f"Random Forest Model for: {network}")
    print(report2)
    #print("Best Parameters:", grid_search_rf.best_params_)

#sys.stdout.close()
