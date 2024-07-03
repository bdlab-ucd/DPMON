import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


CurrentSmokers_Networks = [
    "CurrentSmokersNetwork1-98",
    "CurrentSmokersNetwork2-35",
    "CurrentSmokersNetwork3-16",
    "CurrentSmokersNetwork4-12",
    "CurrentSmokersNetwork5-12",
]

# Creating a dictionary with a network name as key and the columns as the value stored in a list
network_columns = {}

for network in CurrentSmokers_Networks:
    network_columns[network] = []

path = "C:/Users/ramosv/Desktop/NetCo/BioInformedSubjectRepresentation/Current/"

for network in CurrentSmokers_Networks:
    df = pd.read_csv(f"{path}{network}{".csv"}")
    network_columns[network].extend(df.columns.tolist())

current_smokers_omics = pd.read_csv(path + "current_smokers_omics_data.csv")
network_dfs = {}

for network, cols in network_columns.items():
    cols.remove('Unnamed: 0')
    network_dfs[network] = current_smokers_omics[cols]

# Now we have a dictionary(network_dfs) with all the networks as keys and a dataframe with the columns from the omics data set
#print(type(network_dfs["CurrentSmokersNetwork5-12"]))

target_class = pd.read_csv(f"{path}{"current_smokers_gold.csv"}")

y = target_class['finalgold_visit'].values
for network, data in network_dfs.items():
    X = data.values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(f"Logistic Regression Model for: {network} \n {report}")