import pandas as pd
import numpy as np


path = "/home/vicente/BDLab/NetCo/BioInformedSubjectRepresentation/Current/"


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
print(f"y values DF: {y}")

for network, data in current_network_dfs.items():
    X = data.values
    print(f"x values as DF: {X}")


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
print(f"y values DF: {y}")


for network, data in former_networks_dfs.items():
    X = data.values
    print(f"x values as DF: {X}")
    # X = data.to_numpy()
    # print(f"x values as numpy: {X}")

    # data.to_numpy()

    # return X, y


