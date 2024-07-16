import pandas as pd
import numpy as np
import json
import sys

    # {
    #     "results": {
    #         "Log Loss - mean": 2.2717870253111125,
    #         "Log Loss - std": 0.40811022391677804,
    #         "AUC - mean": 0.5081119853494429,
    #         "AUC - std": 0.014521277753021006,
    #         "Accuracy - mean": 0.4817812269031781,
    #         "Accuracy - std": 0.002474783957263427,
    #         "F1 score - mean": 0.31329313066861436,
    #         "F1 score - std": 0.0026954937530048084
    #     },
    #     "train_time": 0.2971571883334339,
    #     "inference_time": 0.013977207999990545,
    #     "model_params": {
    #         "dim": 128,
    #         "depth": 3,
    #         "heads": 8,
    #         "weight_decay": -4,
    #         "learning_rate": -3,
    #         "dropout": 0
    #     }
    # }

if __name__ == "__main__":


    sys.stdout = open('/home/vicente/BDLab/NetCo/BioInformedSubjectRepresentation/tabsurvey/Results/TabSurvey_output.txt', 'w')

    results_Tab = [
        "results_TabTransformer_CurrentSmokersNetwork1-98.json",
        "results_TabTransformer_CurrentSmokersNetwork2-35.json",
        "results_TabTransformer_CurrentSmokersNetwork3-16.json",
        "results_TabTransformer_CurrentSmokersNetwork4-12.json",
        "results_TabTransformer_CurrentSmokersNetwork5-12.json",
        "results_TabTransformer_FormerSmokersNetwork1-88.json",
        "results_TabTransformer_FormerSmokersNetwork2-16.json",
        "results_TabTransformer_FormerSmokersNetwork3-24.json",
        "results_TabTransformer_FormerSmokersNetwork4-24.json",
    ]

    results_Saint = [
        "results_SAINT_CurrentSmokersNetwork1-98.json",
        "results_SAINT_CurrentSmokersNetwork4-12.json",
        "results_SAINT_CurrentSmokersNetwork5-12.json",
        "results_SAINT_FormerSmokersNetwork1-88.json",
    ]
    
    path = '/home/vicente/BDLab/NetCo/BioInformedSubjectRepresentation/tabsurvey/Results/'


    
    # Read the results from each iteration: Then get average of "Accuracy - mean", "Accuracy - std", "F1 score - mean", "F1 score - std".
    for results in results_Saint:
        with open(f"{path}{results}", "r") as file:
            # Loading json file into dictionary for preocessing
            data = json.load(file)

        accuracy = []
        std_acc = []
        f1score = []
        std_f1 = []

        for items in data:
            accuracy.append(float(items["results"]["Accuracy - mean"]))
            std_acc.append(float(items["results"]["Accuracy - std"]))
            f1score.append(float(items["results"]["F1 score - mean"]))
            std_f1.append(float(items["results"]["F1 score - std"]))

        total_accuracy = np.average(np.array(accuracy))
        total_std_acc = np.average(np.array(std_acc))
        total_f1score = np.average(np.array(f1score))
        total_std_f1 = np.average(np.array(std_f1))

        print(f"Results for: {results}\n")
        print(f"Accuracy avg: {total_accuracy} - Std: {total_std_acc}\nF1-Score: {total_f1score} - Std:{total_std_f1}\n")   


    sys.stdout.close()