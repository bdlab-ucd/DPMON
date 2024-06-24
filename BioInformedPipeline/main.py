import os
import statistics
from functools import partial

import matplotlib.pyplot as plt
import torch
import pyreadstat
import numpy as np
import pandas as pd
import networkx as nx
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from torch_geometric.transforms import RandomNodeSplit
from torchmetrics.classification import MulticlassAUROC



from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import tempfile
from ray import train, tune
from ray.train import Checkpoint

CONFIG = {
    'gcn_layer_num': tune.choice([2, 4, 8, 16, 32, 64, 128]),
    'gcn_hidden_dim': tune.choice([4, 8, 16, 32, 64, 128]),
    'lr': tune.loguniform(1e-4, 1e-1),
    'weight_decay': tune.loguniform(1e-4, 1e-1),
    'nn_hidden_dim1': tune.choice([4, 8, 16, 32, 64, 128]),
    'nn_hidden_dim2': tune.choice([4, 8, 16, 32, 64, 128]),
    'num_epochs': tune.choice([2, 16, 64, 512, 1024, 4096, 8192]),
}

def get_new_datasets():
    # Using a Graph that is Independent of the Subjects Dataset
    PPI_graph_adj = pd.read_csv('/home/shussein/NetCO/data/PPI_Yong/ppi_graph_1183_mRNA_updated_root_2656sub.csv', delimiter='\t')
    nodes_names = PPI_graph_adj.columns.tolist()
    PPI_graph = nx.from_numpy_array(PPI_graph_adj.to_numpy())

    # COPDGene_SOMASCAN_1.3K Subjects Dataset
    COPDGene_SOMASCAN13_dataset = pd.read_csv('/home/shussein/NetCO/data/SOMASCAN13/COPDGene_SOMASCAN13_subjects.csv', index_col=None)
    # COPDGene_SOMASCAN13_dataset = COPDGene_SOMASCAN13_dataset.set_index('sid')


    print("Number of Columns %s" % len(COPDGene_SOMASCAN13_dataset.columns.tolist()))
    print("Columns Names %s" % COPDGene_SOMASCAN13_dataset.columns.tolist())
    print("Nodes Names %s" % nodes_names)
    # TODO: Refactor this Code
    nodes_features = []
    for node_name in nodes_names:
        node_features = [1]
        nodes_features.append(node_features)

    features = np.array(nodes_features)
    PPI_graph.remove_edges_from(nx.selfloop_edges(PPI_graph))

    x = np.zeros(features.shape)
    graph_nodes = list(PPI_graph.nodes)
    for m in range(features.shape[0]):
        x[graph_nodes[m]] = features[m]
    x = torch.from_numpy(x).float()

    # Edges Indexes
    edge_index = np.array(list(PPI_graph.edges))
    edge_index = np.concatenate((edge_index, edge_index[:, ::-1]), axis=0)
    edge_index = torch.from_numpy(edge_index).long().permute(1, 0)

    # Edges Weights
    edge_weight = np.array(list(nx.get_edge_attributes(PPI_graph, 'weight').values()))
    edge_weight = np.concatenate((edge_weight, edge_weight), axis=0)
    edge_weight = torch.from_numpy(edge_weight).float()

    dataset = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=torch.zeros(len(nodes_names)))

    transform = RandomNodeSplit(num_val=100, num_test=200)
    data = transform(dataset)
    return data, COPDGene_SOMASCAN13_dataset

def get_datasets():
    # Preparing the Graph Dataset
    DATASET_DIR = '/home/shussein/NetCO/GNNs/data/COPD/SparsifiedNetworks'
    dataset_name = 'trimmed_fev1_0.515_0.111_adj.csv'

    graph_adj_file = os.path.join(DATASET_DIR, dataset_name)
    graph_adj = pd.read_csv(graph_adj_file, index_col=0).to_numpy()
    nodes_names = pd.read_csv(graph_adj_file, index_col=0).index.tolist()

    original_dataset_sid = pd.read_csv('/home/shussein/NetCO/GNNs/data/COPD/SparsifiedNetworks/fev1_X.csv', index_col=0)
    original_dataset_sid.index.name = 'sid'

    clinical_variables, meta = pyreadstat.read_sas7bdat("/home/shussein/NetCO/Data/COPDGene_P1P2P3_SM_NS_Long_Oct22.sas7bdat")
    clinical_variables = clinical_variables.set_index('sid')
    # Filtering on Visit Number
    clinical_variables = clinical_variables[clinical_variables['visitnum'] == 2.0]

    clinical_variables_comorbidities = ['Angina', 'CongestHeartFail', 'CoronaryArtery', 'HeartAttack', 'PeriphVascular',
                                        'Stroke', 'TIA', 'Diabetes',
                                        'Osteoporosis', 'HighBloodPres', 'HighCholest', 'CognitiveDisorder',
                                        'MacularDegen', 'KidneyDisease', 'LiverDisease']
    clinical_variables = clinical_variables.assign(comorbidities=clinical_variables[clinical_variables_comorbidities].sum(axis=1))

    complete_original_dataset = pd.merge(clinical_variables, original_dataset_sid, left_index=True, right_index=True)
    clinical_variables = complete_original_dataset[clinical_variables.columns]

    complete_original_dataset['finalgold_visit'].fillna(0, inplace=True)

    complete_original_dataset.drop(complete_original_dataset[complete_original_dataset['finalgold_visit'] == -1].index, inplace=True)
    original_dataset_sid = original_dataset_sid[original_dataset_sid.index.isin(complete_original_dataset.index.tolist())]
    clinical_variables_cols = ['gender', 'age_visit', 'Chronic_Bronchitis', 'PRM_pct_emphysema_Thirona',
                               'PRM_pct_normal_Thirona', 'Pi10_Thirona', 'comorbidities']

    # Unifying Classes 0, -1, {1, 2} -> 1, {3, 4} -> 2
    complete_original_dataset['finalgold_visit'] = np.where(complete_original_dataset['finalgold_visit'] == 2, 1,
                                                            complete_original_dataset['finalgold_visit'])
    complete_original_dataset['finalgold_visit'] = np.where((complete_original_dataset['finalgold_visit'] == 3) |
                                                            (complete_original_dataset['finalgold_visit'] == 4), 2,
                                                            complete_original_dataset['finalgold_visit'])

    original_dataset_sid['finalgold_visit'] = complete_original_dataset['finalgold_visit']

    graph = nx.from_numpy_array(graph_adj)

    nodes_features = []
    for node_name in nodes_names:
        node_features = []
        for clinical_variable in clinical_variables_cols:
            node_features.append(
                abs(original_dataset_sid[node_name].corr(
                    complete_original_dataset[clinical_variable].astype('float64'))))
        nodes_features.append(node_features)

    features = np.array(nodes_features)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    x = np.zeros(features.shape)
    graph_nodes = list(graph.nodes)
    for m in range(features.shape[0]):
        x[graph_nodes[m]] = features[m]
    x = torch.from_numpy(x).float()

    # Edges Indexes
    edge_index = np.array(list(graph.edges))
    edge_index = np.concatenate((edge_index, edge_index[:, ::-1]), axis=0)
    edge_index = torch.from_numpy(edge_index).long().permute(1, 0)

    # Edges Weights
    edge_weight = np.array(list(nx.get_edge_attributes(graph, 'weight').values()))
    edge_weight = np.concatenate((edge_weight, edge_weight), axis=0)
    edge_weight = torch.from_numpy(edge_weight).float()

    dataset = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=torch.zeros(len(nodes_names)))

    transform = RandomNodeSplit(num_val=5, num_test=5)
    data = transform(dataset)
    # synthetic_dataset = pd.read_csv('../SubjectsSyntheticData/synthetic_data_300_epochs.csv')
    return data, original_dataset_sid


class ResidualBlockN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualBlockN, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += identity
        out = self.relu(out)
        return out
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index, edge_weight):
        identity = x
        out = self.conv1(x, edge_index, edge_weight)
        out = F.relu(out)
        out = self.conv2(out, edge_index, edge_weight)
        out = self.bn(out)
        out += identity
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(ResNet, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.residual_layers = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.bn1(x)

        for layer in self.residual_layers:
            x = layer(x, edge_index, edge_weight)

        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return x

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num=2, dropout=True, **kwargs):
        super(GCN, self).__init__()
        self.layer_num = layer_num
        self.dropout = dropout
        self.conv_first = GCNConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv_first(x, edge_index, edge_weight)
        x = F.relu(x)

        if self.dropout:
            x = F.dropout(x, training=self.training)

        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x.clone(), edge_index, edge_weight)
            x = F.relu(x)

            if self.dropout:
                x = F.dropout(x, training=self.training)
        return x

class Autoencoder(nn.Module):  # TODO: Need to revise this architecture
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
class DownstreamTaskNN(nn.Module):
    def __init__(self, input_size, hidden_dim1, hidden_dim2, output_dim):
        super(DownstreamTaskNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

class DimensionAveraging(nn.Module):
    def __init__(self):
        super(DimensionAveraging, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=1, keepdim=True)

class NeuralNetwork(nn.Module):
    def __init__(self,
                 gcn_input_dim,
                 gcn_hidden_dim,
                 gcn_layer_num,
                 ae_encoding_dim,
                 nn_input_dim,
                 nn_hidden_dim1,
                 nn_hidden_dim2,
                 nn_output_dim):
        super(NeuralNetwork, self).__init__()
        self.gcn = GCN(input_dim=gcn_input_dim, hidden_dim=gcn_hidden_dim, layer_num=gcn_layer_num)
        # self.residual_layer_gcn = ResidualBlockN(gcn_hidden_dim, gcn_hidden_dim)  # Residual block after GCN

        # self.resnet = ResNet(input_dim=gcn_input_dim, hidden_dim=gcn_hidden_dim, num_layers=gcn_layer_num)
        # TODO: Revisit the Autoencoder - Since we Only Need to Encode so the Model doesn't learn here!
        self.autoencoder = Autoencoder(input_dim=gcn_hidden_dim, encoding_dim=ae_encoding_dim)
        # self.residual_layer_ae = ResidualBlockN(ae_encoding_dim, ae_encoding_dim)  # Residual block after Autoencoder

        self.dim_averaging = DimensionAveraging()
        # self.predictor = DownstreamTaskNN(nn_input_dim, nn_hidden_dim1, nn_hidden_dim2, nn_output_dim)
        self.predictor = LogisticRegression(nn_input_dim, nn_output_dim)
        self.epoch = 0
        # self.residual_layer_predictor = ResidualBlockN(nn_output_dim, nn_output_dim)  # Residual block after Predictor


    def forward(self, graph_data, original_dataset):

        nodes_embeddings = self.gcn(graph_data)
        # Plot the Embedding Space - before Averaging-

        # # Define the number of rows and columns for the grid
        # num_rows = int(np.ceil(np.sqrt(nodes_embeddings.shape[1])))
        # num_cols = int(np.ceil(nodes_embeddings.shape[1] / num_rows))
        #
        # # Create the grid of subplots
        # fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
        #
        # # Flatten the axes array to make it easier to iterate
        # axes = axes.flatten()
        #
        # # Plot data in each subplot
        # for i in range(nodes_embeddings.shape[1]):
        #     axes[i].scatter(range(0, nodes_embeddings.shape[0]), nodes_embeddings[:, i].detach().numpy())
        #     axes[i].set_title(f'Plot {i + 1}')
        #
        # # Hide any empty subplots
        # for i in range(nodes_embeddings.shape[1], num_rows * num_cols):
        #     fig.delaxes(axes[i])
        #
        # # Adjust layout
        # plt.tight_layout()
        # plt.savefig(f'{self.epoch}.png')
        # # Show the plot
        # plt.show()


        # print("Nodes Embeddings %s" % nodes_embeddings)
        # nodes_embeddings = self.residual_layer_gcn(nodes_embeddings)  # Apply residual block after GCN

        # nodes_embeddings = self.resnet(graph_data)
        # print("Dimenions of Embeddings after GCN")
        # print(nodes_embeddings.shape)
        # print(nodes_embeddings)
        nodes_embeddings_ae = self.autoencoder(nodes_embeddings)
        # print("Scaling Factors %s" % nodes_embeddings_ae)
        # print("Embeddings %s" % nodes_embeddings)
        # nodes_embeddings_ae = self.residual_layer_ae(nodes_embeddings_ae)  # Apply residual block after Autoencoder

        # print("Dimensions of Embeddings after Autoencoder")
        # print(nodes_embeddings_ae.shape)
        # print(nodes_embeddings_ae)

        nodes_embeddings_avg = self.dim_averaging(nodes_embeddings)
        # Plot the Embedding Space - after Averaging-
        # print(range(0, len(nodes_embeddings_avg.detach().numpy())))
        # print(nodes_embeddings_avg.detach().numpy())
        # plt.scatter(range(0, len(nodes_embeddings_avg.detach().numpy())), nodes_embeddings_avg.detach().numpy())
        # plt.savefig("Nodes Embeddings - Epoch %d" % self.epoch)
        # print("Dimensions of Embeddings after Averaging")
        # print(nodes_embeddings_avg.shape)
        # print("Averaging Embeddings %s" % nodes_embeddings_avg)

        # Multiplying the original_dataset with the generated embeddings
        # original_dataset_with_embeddings = torch.mul(original_dataset, nodes_embeddings_ae.expand(original_dataset.shape[1], original_dataset.shape[0]).t())
        original_dataset_with_embeddings = torch.mul(original_dataset,
                                                     nodes_embeddings_avg.expand(original_dataset.shape[1],
                                                                                original_dataset.shape[0]).t())
        # # TODO: Revert Changes
        # # Using Random Scaling Factors
        # nodes_embeddings_rand = torch.randn_like(nodes_embeddings_avg)
        # # print("Random Scaling Factors %s" % nodes_embeddings_rand)
        # original_dataset_with_embeddings = torch.mul(original_dataset,
        #                                              nodes_embeddings_rand.expand(original_dataset.shape[1],
        #                                                                          original_dataset.shape[0]).t())
        # print("New Dataset %s" % original_dataset_with_embeddings)
        # print("Dimensions of the Dataset after Integration")
        # print(original_dataset_with_embeddings.shape)
        # print(original_dataset_with_embeddings)
        predictions = self.predictor(original_dataset_with_embeddings)
        # predictions = self.residual_layer_predictor(predictions)  # Apply residual block after Predictor

        # print("Dimensions of the Dataset after Prediction")
        # print(predictions.shape)
        # print(predictions)
        self.epoch += 1
        return predictions, original_dataset_with_embeddings

def train_n(config):
    # graph_data, dataset = get_datasets()
    graph_data, dataset = get_new_datasets()
    dataset_after_training_l = []

    stratified_data = []
    stratified_labels = []
    no_copd = dataset.copy()
    no_copd.loc[no_copd['finalgold_visit'] == 2, 'finalgold_visit'] = 1
    print("No COPD Indx %s" % no_copd.index.tolist())
    moderate_copd = dataset.copy()
    moderate_copd = moderate_copd.loc[(moderate_copd['finalgold_visit'] == 0) | (moderate_copd['finalgold_visit'] == 1)]
    print("Moderate COPD Indx %s" % moderate_copd.index.tolist())
    severe_copd = dataset.copy()
    severe_copd = severe_copd.loc[(severe_copd['finalgold_visit'] == 0) | (severe_copd['finalgold_visit'] == 2)]
    severe_copd.loc[severe_copd['finalgold_visit'] == 2, 'finalgold_visit'] = 1
    print("Severe COPD Indx %s" % severe_copd.index.tolist())

    # y_no_copd = dataset[dataset['finalgold_visit'] == 0]
    stratified_data.append(no_copd)
    stratified_data.append(moderate_copd)
    stratified_data.append(severe_copd)
    # stratified_labels.append(y_no_copd)

    for idx, data in enumerate(stratified_data):
        print("Training for Dataset %d" % idx)
        # X = dataset.drop(['finalgold_visit'], axis=1)
        # Y = dataset['finalgold_visit']
        X = data.drop(['finalgold_visit'], axis=1)
        Y = data['finalgold_visit']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.to_numpy())
        X_test_scaled = scaler.transform(X_test.to_numpy())

        # Convert to PyTorch Tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.LongTensor(y_train.to_numpy())

        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.LongTensor(y_test.to_numpy())

        gcn_input_dim = graph_data.x.shape[1]
        nn_input_dim = X_train_tensor.shape[1]
        nn_output_dim = 2  # Number of Classes

        model = NeuralNetwork(gcn_input_dim=gcn_input_dim, gcn_hidden_dim=config['gcn_hidden_dim'],
                              gcn_layer_num=config['gcn_layer_num'],
                              ae_encoding_dim=1,
                              nn_input_dim=nn_input_dim, nn_hidden_dim1=config['nn_hidden_dim1'],
                              nn_hidden_dim2=config['nn_hidden_dim2'],
                              nn_output_dim=nn_output_dim)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

        # Training loop
        for epoch in range(config['num_epochs']):
            optimizer.zero_grad()
            # Forward pass
            output, dataset_with_embeddings = model(graph_data, X_train_tensor)
            # Compute loss
            loss = criterion(output, y_train_tensor)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Print loss for monitoring
            print(f"Epoch [{epoch + 1}/{epoch}], Loss: {loss.item()}")

            # print("*****************GRADIENTS********************")
            # print("Gradients for the Model!!!! ")
            # # Track gradients (gnn_model) and model parameters in TensorBoard
            # for name, param in model.named_parameters():
            #     print(f'{name}.grad', param.grad, epoch)

            metrics = {"loss": loss.item()}
            with tempfile.TemporaryDirectory() as tempdir:
                torch.save(
                    {"epoch": epoch, "model_state": model.state_dict()},
                    os.path.join(tempdir, "checkpoint.pt"),
                )
                train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))

            with torch.no_grad():
                model.eval()
                # print("Before Testing")
                outputs, test_with_embeddings = model(graph_data, X_test_tensor)  # .flatten()
                # print("After Testing")
                l, predicted = torch.max(outputs, 1)
                accuracy = torch.sum(predicted == y_test_tensor).item() / len(y_test)

                confusion_mtrx = confusion_matrix(y_test_tensor, predicted)
                mc_auroc = MulticlassAUROC(num_classes=3, average='macro', thresholds=None)
                print(f'Test Accuracy: {accuracy}')
                # print(f'Classification Report: {classification_report(y_test_tensor, predicted)}')
                # print(f'ROC AUC Score: {mc_auroc(outputs, y_test_tensor)}')
                # print(f'Confusion Matrix: {confusion_mtrx}')
                metrics["accuracy"] = accuracy
                with tempfile.TemporaryDirectory() as tempdir:
                    train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))
        updated_dataset_df = pd.DataFrame(torch.cat((dataset_with_embeddings, test_with_embeddings), dim=0).detach().numpy(), index=data.index)
        # print("Updated DF Size")
        # print(updated_dataset_df.shape)
        # print("Updated DF Index %s" % updated_dataset_df.index.tolist())
        # print("Len of Index %s" % len(updated_dataset_df.index.tolist()))
        if idx == 0:
            filter_y = dataset[dataset['finalgold_visit'] == 0]
            # print("Filterred Y Index %s" % filter_y.index.tolist())
            # print("Full dataset index %s" % updated_dataset_df.index.tolist())
            updated_dataset = updated_dataset_df.loc[filter_y.index.tolist()]
            updated_dataset['finalgold_visit'] = filter_y['finalgold_visit']
            # print(updated_dataset)
            # print(updated_dataset.shape)
            dataset_after_training_l.append(updated_dataset)
        elif idx == 1:
            filter_y = moderate_copd[moderate_copd['finalgold_visit'] == 1]

            # print("Filterred Y Index %s" % filter_y.index.values)
            # print("Fixed Indicies %s" % mapped_indexes)
            # print("Full dataset index %s" % updated_dataset_df.index.values)
            # for k in filter_y.index.tolist():
            #     if k not in updated_dataset_df.index.tolist():
            #         print("WEEEE!!")
            #         print(k)
            # print("After Wee")
            # print(updated_dataset_df.iloc[filter_y.index.tolist()])
            # for index in filter_y.index.tolist():
            #     print("Index %s" % index)
            #     print(updated_dataset_df[index])
            #     print("Passed!")
            updated_dataset = updated_dataset_df.loc[filter_y.index.values]
            updated_dataset['finalgold_visit'] = filter_y['finalgold_visit']
            # print(updated_dataset_1)
            # print(updated_dataset_1.shape)
            dataset_after_training_l.append(updated_dataset)
        elif idx == 2:
            filter_y = severe_copd[severe_copd['finalgold_visit'] == 1]
            # print("Filterred Y Index %s" % filter_y.index.tolist())
            # print("Full dataset index %s" % updated_dataset_df.index.tolist())
            updated_dataset = updated_dataset_df.loc[filter_y.index.tolist()]
            updated_dataset['finalgold_visit'] = filter_y['finalgold_visit'] * 2
            # print(updated_dataset)
            # print(updated_dataset.shape)
            dataset_after_training_l.append(updated_dataset)

    # We are Done Training! Time to Use the New Dataset to Predict!
    print("**** DONE TRAINING ON DATASETS! TIME TO PREDICT ****")
    dataset_after_training_df = pd.concat(dataset_after_training_l)
    # Shuffle the Dataframe
    dataset_after_training_df = dataset_after_training_df.sample(frac=1)
    dataset_after_training_df.to_csv("after_training.csv", index=None)
    X = dataset_after_training_df.drop(['finalgold_visit'], axis=1)
    Y = dataset_after_training_df['finalgold_visit']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.to_numpy())
    X_test_scaled = scaler.transform(X_test.to_numpy())

    # Convert to PyTorch Tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train.to_numpy())

    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test.to_numpy())

    nn_input_dim = X_train_tensor.shape[1]
    nn_output_dim = 3  # Number of Classes
    model = LogisticRegression(input_dim=nn_input_dim, output_dim=nn_output_dim)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # Training loop
    for epoch in range(config['num_epochs']):
        optimizer.zero_grad()
        # Forward pass
        output = model(X_train_tensor)
        # Compute loss
        loss = criterion(output, y_train_tensor)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print loss for monitoring
        print(f"Epoch [{epoch + 1}/{epoch}], Loss: {loss.item()}")

        # print("*****************GRADIENTS********************")
        # print("Gradients for the Model!!!! ")
        # # Track gradients (gnn_model) and model parameters in TensorBoard
        # for name, param in model.named_parameters():
        #     print(f'{name}.grad', param.grad, epoch)

        metrics = {"loss": loss.item()}
        with tempfile.TemporaryDirectory() as tempdir:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict()},
                os.path.join(tempdir, "checkpoint.pt"),
            )
            train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))

        with torch.no_grad():
            model.eval()
            # print("Before Testing")
            outputs = model(X_test_tensor)  # .flatten()
            # print("After Testing")
            l, predicted = torch.max(outputs, 1)
            accuracy = torch.sum(predicted == y_test_tensor).item() / len(y_test)

            confusion_mtrx = confusion_matrix(y_test_tensor, predicted)
            mc_auroc = MulticlassAUROC(num_classes=3, average='macro', thresholds=None)
            print(f'Test Accuracy: {accuracy}')
            # print(f'Classification Report: {classification_report(y_test_tensor, predicted)}')
            # print(f'ROC AUC Score: {mc_auroc(outputs, y_test_tensor)}')
            # print(f'Confusion Matrix: {confusion_mtrx}')
            metrics["accuracy"] = accuracy
            with tempfile.TemporaryDirectory() as tempdir:
                train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))
    return accuracy

def train_nn(config):
    # graph_data, dataset = get_datasets()
    graph_data, dataset = get_new_datasets()

    X = dataset.drop(['finalgold_visit'], axis=1)
    Y = dataset['finalgold_visit']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.to_numpy())
    X_test_scaled = scaler.transform(X_test.to_numpy())

    # Convert to PyTorch Tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train.to_numpy())

    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test.to_numpy())

    gcn_input_dim = graph_data.x.shape[1]
    nn_input_dim = X_train_tensor.shape[1]
    nn_output_dim = 3  # Number of Classes

    model = NeuralNetwork(gcn_input_dim=gcn_input_dim, gcn_hidden_dim=config['gcn_hidden_dim'],
                          gcn_layer_num=config['gcn_layer_num'],
                          ae_encoding_dim=1,
                          nn_input_dim=nn_input_dim, nn_hidden_dim1=config['nn_hidden_dim1'],
                          nn_hidden_dim2=config['nn_hidden_dim2'],
                          nn_output_dim=nn_output_dim)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # Training loop
    for epoch in range(config['num_epochs']):
        optimizer.zero_grad()
        # Forward pass
        output, dataset_with_embeddings = model(graph_data, X_train_tensor)
        # Compute loss
        loss = criterion(output, y_train_tensor)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print loss for monitoring
        print(f"Epoch [{epoch + 1}/{epoch}], Loss: {loss.item()}")

        # Running KNN with Each Epoch!!!
        grid_params = {'n_neighbors': [5, 7, 9, 11, 13, 15, 19, 21],
                       'weights': ['uniform', 'distance'],
                       'metric': ['minkowski', 'euclidean', 'manhattan'],
                       'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                       }
        gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose=1, cv=3, n_jobs=-1)
        scaler = MinMaxScaler()
        scaler.fit(dataset_with_embeddings.detach().numpy())
        synthetic_dataset_scaled = scaler.transform(dataset_with_embeddings.detach().numpy())
        X_train1, X_test1, y_train1, y_test1 = train_test_split(synthetic_dataset_scaled,
                                                            y_train, test_size=0.3,
                                                            random_state=0)
        print("Training Dataset %s" % X_train1)
        g_res = gs.fit(X_train1, y_train1)
        print("Training Dataset")
        print("Best Score %s" % g_res.best_score_)
        print("Best Params %s" % g_res.best_params_)

        # print("*****************GRADIENTS********************")
        # print("Gradients for the Model!!!! ")
        # # Track gradients (gnn_model) and model parameters in TensorBoard
        # for name, param in model.named_parameters():
        #     print(f'{name}.grad', param.grad, epoch)

        metrics = {"loss": loss.item()}
        with tempfile.TemporaryDirectory() as tempdir:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict()},
                os.path.join(tempdir, "checkpoint.pt"),
            )
            train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))

        with torch.no_grad():
            model.eval()
            # print("Before Testing")
            outputs, test_with_embeddings = model(graph_data, X_test_tensor)  # .flatten()
            # print("After Testing")
            l, predicted = torch.max(outputs, 1)
            accuracy = torch.sum(predicted == y_test_tensor).item() / len(y_test)

            confusion_mtrx = confusion_matrix(y_test_tensor, predicted)
            mc_auroc = MulticlassAUROC(num_classes=3, average='macro', thresholds=None)
            print(f'Test Accuracy: {accuracy}')
            # print(f'Classification Report: {classification_report(y_test_tensor, predicted)}')
            # print(f'ROC AUC Score: {mc_auroc(outputs, y_test_tensor)}')
            # print(f'Confusion Matrix: {confusion_mtrx}')
            metrics["accuracy"] = accuracy
            with tempfile.TemporaryDirectory() as tempdir:
                train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))

            # Running KNN with Each Epoch!!!
            grid_params = {'n_neighbors': [5, 7, 9, 11, 13, 15, 19, 21],
                           'weights': ['uniform', 'distance'],
                           'metric': ['minkowski', 'euclidean', 'manhattan'],
                           'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                           }
            gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose=1, cv=3, n_jobs=-1)
            scaler = MinMaxScaler()
            scaler.fit(test_with_embeddings.detach().numpy())
            synthetic_dataset_scaled = scaler.transform(test_with_embeddings.detach().numpy())
            X_train2, X_test2, y_train2, y_test2 = train_test_split(synthetic_dataset_scaled,
                                                                y_test, test_size=0.3,
                                                                random_state=0)
            print("Testing Dataset %s" % X_test2)
            g_res = gs.fit(X_test2, y_test2)
            print("Testing Dataset")
            print("Best Score %s" % g_res.best_score_)
            print("Best Params %s" % g_res.best_params_)

    return accuracy

def main():
    accuracies = []
    for i in range(1):
        hyperparams_tuning = False
        if hyperparams_tuning:
            reporter = CLIReporter(metric_columns=["loss"])
            GRACE_PERIOD = 10
            # Scheduler to stop bad performing trials
            scheduler = ASHAScheduler(
                metric="loss",
                mode="min",
                grace_period=GRACE_PERIOD,
                reduction_factor=2)

            # TODO: Remove this higher limit on the number of trials
            NUM_SAMPLES = 5000
            result = tune.run(partial(train_n),
                              resources_per_trial={"cpu": 8, "gpu": 0},
                              config=CONFIG, num_samples=NUM_SAMPLES, scheduler=scheduler,
                              # local_dir='outputs/raytune_results_%s' % experiment_name,
                              name='PredictionOneAvgingSOMASCAN',
                              keep_checkpoints_num=1,
                              checkpoint_score_attr='min-loss',
                              progress_reporter=reporter
                              )
            best_trial = result.get_best_trial("loss", "min", "last")
            print("Best trial config: {}".format(best_trial.config))
            print("Best trial final test loss: {}".format(best_trial.last_result["loss"]))
        else:
            accuracy = train_nn(config={
                'gcn_layer_num': 4,
                'gcn_hidden_dim': 8,
                'lr': 0.029352058542109778,
                'weight_decay': 0.0010178240820048331,
                'nn_hidden_dim1': 8,
                'nn_hidden_dim2': 32,
                'num_epochs': 10,
            })
            print("Accuracy: {}".format(accuracy))
            accuracies.append(accuracy)
    print("Best Accuracy: {}".format(max(accuracies)))
    print("Average Accuracy is {}".format(sum(accuracies) / len(accuracies)))
    # print("Standard Deviation is {}".format(statistics.stdev(accuracies)))

if __name__ == "__main__":
    main()