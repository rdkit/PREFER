# Working in AzureML
PREFER can also run in AzureML. Main scripts regarding AzureML are collected in PREFER/prefer/azure_ml/
The following steps should be implemented:


## STEP 1: install the full cddd and moler environments
As follows:

```
conda env create -f moler-environment-light.yml
OR
conda env create -f cddd-environment-light.yml
```

## STEP 2: import datasets and models in your Azure Storage
Import the dataset you want to use as well as the cddd and moler models in your Azure storage

## STEP 3: prepare the config file
Prepare a yaml file with the following information:

```
path_to_df: 'path_in_Azure_to_the_saved_df'
experiment_name: 'name of the experiment'
id_column_name:  ''
smiles_column_name:  ''
properties_column_name_list: 
      - 'property1_columns_name' # NB if more than one then it is a multitasking
problem_type: 'regression' # Can be regression or classification
splitting_strategy: 'random'
representations:
    'DESCRIPTORS2D' : '' 
    'FINGERPRINTS': ''
    'CDDD': 'path_in_azure_to_stored_CDDD_model'
```

An example is provided in PREFER/prefer/azure_ml/config_logD_azure.yaml



## STEP 3: import datasets and models in your Azure Storage
Go to PREFER/prefer/azure_ml and run the following command:

```
python schedule_global_model_pipeline.py --prefer_args config_logD_azure.yaml --prefer_env cddd-environment.yml
```

One can also use the moler-environment.yml for running experiments with moler environment. 




