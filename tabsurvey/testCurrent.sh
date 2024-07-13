#!/bin/bash

N_TRIALS=2
EPOCHS=10

# Models and configurations
declare -A MODELS
MODELS=( 
         ["SAINT"]="torch"
         ["TabTransformer"]="torch"
          )

CONFIGS=( "config/current.yml" )

# Network configurations
CurrentSmokers_Networks=(
    "CurrentSmokersNetwork1-98"
    "CurrentSmokersNetwork2-35"
    "CurrentSmokersNetwork3-16"
    "CurrentSmokersNetwork4-12"
    "CurrentSmokersNetwork5-12"
)

# Loop through each configuration, model, and network
for config in "${CONFIGS[@]}"; do
    for model in "${!MODELS[@]}"; do
        for network in "${CurrentSmokers_Networks[@]}"; do
            # Extract the number of features from the network name
            num_features="${network##*-}"

            printf "\n\n----------------------------------------------------------------------------\n"
            printf 'Training %s with %s on dataset %s (features: %s) in env %s\n\n' "$model" "$config" "$network" "$num_features" "${MODELS[$model]}"

            # Loop only executes this line 8 times!
            # 1 for the CONFIG file, 2 for the Models (SAINT and TabTransformer) and 4 times for each network
            python train.py --config "$config" --model_name "$model" --dataset "$network" --num_features "$num_features" --n_trials $N_TRIALS --epochs $EPOCHS --optimize_hyperparameters
        done
    done
done
