#  Copyright (c) 2023, Novartis Institutes for BioMedical Research Inc. and Microsoft Corporation
#  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: 
#
#     * Redistributions of source code must retain the above copyright 
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following 
#       disclaimer in the documentation and/or other materials provided 
#       with the distribution.
#     * Neither the name of Novartis Institutes for BioMedical Research Inc. nor Microsoft Corporation 
#       nor the names of its contributors may be used to endorse or promote 
#       products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Created by Jessica Lanini, January 2023


import sys


import numpy as np
import os
from os import listdir
import datetime
import pandas as pd
from os.path import isfile, join

from prefer.src.molecule_representations import MoleculeRepresentations
from prefer.src.vector_molecule_representations import VectorMoleculeRepresentations


# Auto-SKLearn
from ConfigSpace.configuration_space import ConfigurationSpace
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, UNSIGNED_DATA, INPUT
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.metrics import balanced_accuracy, r2


def create_model_based_molecule_representation(run_commands, model_representations_path, experiment_name,split_type):
    
    '''
    This function is used to create model based representations and store them. This is the function version of the python script compute_model_based_representations.py
    '''
    print(f'Computing molecular representation for {experiment_name} using the following commands: {run_commands}')
    os.system(run_commands)
    print(f'The representations  model_representations_path: {model_representations_path}')
    files = [f for f in listdir(model_representations_path) if isfile(join(model_representations_path, f))]
    collect_dates = []
    mapping = {}
    for file in files:
        date = file.split('_')[-1]
        date = date.replace('.pkl','')
        date = datetime.datetime.strptime(date, '%Y%m%d-%H%M%S')
        collect_dates.append(date)
        mapping[date] = file

    collect_dates.sort()
    representation_path = f'{model_representations_path}/{mapping[collect_dates[-1]]}'
    model_name = representation_path.split("_")[0]

    model_name = model_name.replace(".", "")
    model_name = model_name.replace("/", "")
    vector_repr = VectorMoleculeRepresentations(
        df=pd.DataFrame(), representation_name="", split_type=" "
    )
    model_based_representation = vector_repr.load(representation_path)
    model_based_representation.experiment_name = experiment_name
    model_based_representation.representation_name = model_name
    model_based_representation.split_type = split_type
    if "molecule_representation" in model_based_representation.df.columns.values:
        return model_based_representation.df.molecule_representation
    elif model_name in model_based_representation.df.columns.values:
        return model_based_representation.df.model_name
    else:
        raise ValueError(f'Dataframe related to the representation computed does not contain neither molecule_representation nor {model_name}. Cannot pass representations')
        

def ensemble_predictions_autosklearn(autosklearnmodel, X: np.ndarray):
    '''
    Given the AutoSklearnRegressor or AutoSklearnClassifier based on ensembles (the best one found by AutoSKlearn) of ensemble models (one for each fold of the CV of each model) 
    and a numpy array with the set of features to predict
    return the weighted predictions, the predictions and the ensemble weights.
    This may be useful if one needs to use each results of the ensembles pther resons (e.g. uncertainty estimation)
    
    The main structure, if ensemble and CV are used to find the best autosklearn model is the following:
    
    Ensemble member 1 (weght_1):   - model folder 1
                                   - model folder 2
                                   - ...
                                   - model folder k
    Ensemble member 2 (weght_2):   - model folder 1
                                   - model folder 2
                                   - ...
                                   - model folder k
    ....
    Ensemble member M (weght_M):   - model folder 1
                                   - model folder 2
                                   - ...
                                   - model folder k
                                   
    In the code for each ensemble member the mean prediction (for each new sample) over the models in each folder is computed
    
    Ensemble member 1 (weght_1):   - mean(predictions1, predictions 2, ..., predictions k) = mean_prediction1
    Ensemble member 2 (weght_2):   - mean(predictions1, predictions 2, ..., predictions k) = mean_prediction2
    ....
    Ensemble member M (weght_M):   - mean(predictions1, predictions 2, ..., predictions k) = mean_predictionM
    
    And then the weighted mean is computed 
    final prediction = 1/M*(weght_1*mean_prediction1+weght_2*mean_prediction2+...+weght_M*mean_predictionM)
    
    '''
    n = X.shape[0]
    predictions_weighted = np.zeros(n)
    predictions = []
    weights =  []
    for pipeline in autosklearnmodel.get_models_with_weights():
        predictions_weighted = predictions_weighted+(pipeline[1].predict(X)*pipeline[0])
        predictions.append(pipeline[1].predict(X))
        weights.append(pipeline[0])
    return predictions, weights



def output_dataframe_preparation(
    df, index_train: list, index_test: list, predictions_train, predictions_test
):
    """
    Function to prepare the final dataframe to be stored.
    Given the initial dataframe, the indices of the train and the test set and the corresponding predictions generate a final dataframe of the shape:
    
    ¦id¦smiles¦property_1¦model_predictions_property_1¦is_train¦
    """

    # Convert predictions train/test as numpy array!
    predictions_train = np.array(predictions_train)
    predictions_test = np.array(predictions_test)

    if (len(index_train) != predictions_train.shape[0]) or (
        len(index_test) != predictions_test.shape[0]
    ):

        raise ValueError(
            "Predictions in the train or in the test sets have a different lenght with respect to the corresponding indices"
        )
    else:

        if (predictions_train.ndim>1):
            num_of_tasks = predictions_train.shape[1]
        else:
            num_of_tasks = 1

        if num_of_tasks > 1:  # multitasking
            for property_index in range(0, num_of_tasks):
                df.at[
                    index_train, f"model_predictions_property_{property_index+1}"
                ] = predictions_train[:, property_index]
                df.at[
                    index_test, f"model_predictions_property_{property_index+1}"
                ] = predictions_test[:, property_index]
        else:  # single task
            df.at[index_train, "model_predictions_property_1"] = predictions_train
            df.at[index_test, "model_predictions_property_1"] = predictions_test

        df.at[index_train, "is_train"] = True
        df.at[index_test, "is_train"] = False

    return df


def check_if_small_data(representation: MoleculeRepresentations,):
    """
    Function to discriminate whether the input dataset can be considered small according to the following assumption: we consider datasets to be small if the number of datapoints is less than 5000 - see Stanley, Megan, et al. "Fs-mol: A few-shot learning dataset of molecules." Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2). 2021.
    """
    N = representation.df.shape[0]
    if N < 5000:
        print('>>> Number of samples less than 5000 --> setting small data hyperparams')
        return True
    else:
        return False


def convert_list_into_dict(list_: list):
    dict_ = {}
    for elem in list_:
        if "=" in elem:
            elem_tmp = elem.split("=")
        elif ":" in elem:
            elem_tmp = elem.split(":")
        else:
            raise ValueError(
                'The assumption is that each elemet of the list is a string containing a key-value pairs separated by ":" or "=". Neither ":" or "=" was found in the string, so it cannot be split into key-value pairs.'
            )
        elem_tmp[0] = elem_tmp[0].rstrip()
        elem_tmp[0] = elem_tmp[0].lstrip()

        elem_tmp[1] = elem_tmp[1].rstrip()
        elem_tmp[1] = elem_tmp[1].lstrip()
        dict_[elem_tmp[0]] = elem_tmp[1]

    return dict_


def convert_atype_to_btype(a, b):
    if type(b) == int:
        return int(a)
    elif type(b) == float:
        return float(a)
    elif type(b) == str:
        a = str(a)
        a = a.replace('"', "")
        return a
    elif type(b) == bool:
        return bool(a)
    else:
        raise ValueError(f"{type(b)} not known")


def get_autosklearn_customized_model(model_instance: list, model_type: str, working_directory: str):
    """
    Function to convert the list of customized parameters given as input by the user into a autosklearn model object
    """
    dict_ = get_autosklearn_params()
    model_instance_dict = convert_list_into_dict(model_instance)
    for param in model_instance_dict.keys():
        if param in dict_:
            tmp_val = convert_atype_to_btype(model_instance_dict[param], dict_[param])
            dict_[param] = tmp_val
    if model_type == "regression":
        automl = AutoSklearnRegressor(
            time_left_for_this_task=dict_["time_left_for_this_task"],
            per_run_time_limit=dict_["per_run_time_limit"],
            initial_configurations_via_metalearning=dict_[
                "initial_configurations_via_metalearning"
            ],
            ensemble_size=dict_["ensemble_size"],
            ensemble_nbest=dict_["ensemble_nbest"],
            max_models_on_disc=dict_["max_models_on_disc"],
            seed=dict_["seed"],
            memory_limit=dict_["memory_limit"],
            resampling_strategy=dict_["resampling_strategy"],
            delete_tmp_folder_after_terminate=dict_["delete_tmp_folder_after_terminate"],
            metric=r2,
            n_jobs=dict_["n_jobs"],
            tmp_folder=working_directory + "/_autosklearn",
        )

    elif model_type == "classification":
        automl = AutoSklearnClassifier(
            time_left_for_this_task=dict_["time_left_for_this_task"],
            per_run_time_limit=dict_["per_run_time_limit"],
            initial_configurations_via_metalearning=dict_[
                "initial_configurations_via_metalearning"
            ],
            ensemble_size=dict_["ensemble_size"],
            ensemble_nbest=dict_["ensemble_nbest"],
            max_models_on_disc=dict_["max_models_on_disc"],
            seed=dict_["seed"],
            memory_limit=dict_["memory_limit"],
            resampling_strategy=dict_["resampling_strategy"],
            delete_tmp_folder_after_terminate=dict_["delete_tmp_folder_after_terminate"],
            metric=balanced_accuracy,
            n_jobs=dict_["n_jobs"],
            tmp_folder=working_directory + "/_autosklearn",
        )
    else:
        raise ValueError(
            f"{model_type} not known as model_type. Only classification and regression are possible"
        )

    return automl


def get_autosklearn_params():
    print(
        "WARNING: not all the autosklearn params can be changed. The full list of exposed autosklearn params is the following: "
    )
    dict_ = {
        "time_left_for_this_task": 3600,
        "per_run_time_limit": 360,
        "initial_configurations_via_metalearning": 25,
        "ensemble_size": 50,
        "ensemble_nbest": 50,
        "max_models_on_disc": 50,
        "seed": 1,
        "memory_limit": 3072,
        "resampling_strategy": "holdout",
        "delete_tmp_folder_after_terminate": True,
        "n_jobs": -1,
    }
    for key in dict_.keys():
        print(key)
    return dict_


class NoPreprocessing(AutoSklearnPreprocessingAlgorithm):
    """
    From https://automl.github.io/auto-sklearn/master/examples/80_extending/example_extending_data_preprocessor.html
    """

    def __init__(self, **kwargs):
        """This preprocessors does not change the data"""
        # Some internal checks makes sure parameters are set
        for key, val in kwargs.items():
            setattr(self, key, val)

    def fit(self, X, Y=None):
        return self

    def transform(self, X):
        return X

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "NoPreprocessing",
            "name": "NoPreprocessing",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (SPARSE, DENSE, UNSIGNED_DATA),
            "output": (INPUT,),
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        return ConfigurationSpace()  # Return an empty configuration as there is None


def retrieve_models_names(bench):
    """
    Functionality useful for AutoSKlearn compatibility
    Given the bench object this function returns a string containing the list (lenght == ensemble lenght) of models lists (for each ensemble member one model for each fold of the CV is kept by autosklearn - as soft ensemble)
    Return a list of these ensemble models names for each of the representation under analysis
    """
    ensemble_models = []
    ensemble_models_list = []
    representation_names = bench.representations
    if bench.problem_type == "regression":
        _check_my_dummy = "MyDummyRegressor"
    else:
        _check_my_dummy = "MyDummyClassifier"
    print(
        "WARNING: retrieve_models_names works only for one representation per benchmarking object"
    )
    for representation_name in representation_names:
        if _check_my_dummy in str(
            bench.best_estimator[representation_name].get_models_with_weights()
        ):  # no models was found better than dummy
            print(
                ">>>>>>>>>>>>>>>>>>> [WARNING]: No model better than Dummy has been found by AutoSklearn - The registered model will compute only the mean of the training set"
            )
            ensemble_models.append(f"[{_check_my_dummy}]")
        else:  # optimized models were found
            for key in (
                bench.best_estimator[representation_name].show_models().keys()
            ):  # looping over the ensemble members
                # just take the first model of the k-models (/k-fold) since it is the same in each
                k_folds_models = []
                model_type = None
                if "estimators" in bench.best_estimator[representation_name].show_models()[key]:
                    for k_folds_model in bench.best_estimator[representation_name].show_models()[
                        key
                    ]["estimators"]:
                        # defining model type
                        if "sklearn_regressor" in k_folds_model:
                            model_type = "sklearn_regressor"
                        else:
                            model_type = "sklearn_classifier"

                        # appending model names
                        if (
                            str(k_folds_model[model_type]).split("(")[0] not in k_folds_models
                        ):  # this is not needed given that all the submodels for each CV fold related to the same ensemble member should be the same type (e.g. SVR), but with different hyperparam - but in this case we just need the info on the name
                            k_folds_models.append(str(k_folds_model[model_type]).split("(")[0])
                        else:
                            continue
                else:
                    # defining model type
                    if (
                        "sklearn_regressor"
                        in bench.best_estimator[representation_name].show_models()[key]
                    ):
                        model_type = "sklearn_regressor"
                    else:
                        model_type = "sklearn_classifier"
                    k_folds_models.append(
                        str(
                            bench.best_estimator[representation_name].show_models()[key][model_type]
                        ).split("(")[0]
                    )

                ensemble_models.append(k_folds_models)
        ensemble_models_list.append(str(ensemble_models))
    return ensemble_models_list


def convert_prediction_types(predictions):
    """
    Function to convert ndarrays (possible way the customize MLP models arrange the output) into a 1darray
    """
    if ((isinstance(predictions[0], np.ndarray)) or isinstance(predictions[0], list)) and (
        len(predictions[0]) == 1
    ):  # case when predictions = [[a],[b],[c],..]
        return np.array([x[0] for x in predictions])
    else:
        return predictions


def retrieve_metrics_names(problem_type):
    """
    methods used to provide the metrics names for regression and classification problems
    """
    if problem_type == "regression":
        return [
            "RMSE_test",
            "RMSE_train",
            "R2",
            "RMSE_test_Norm",
            "Mean test error value",
            "Max test error value",
            "Min error value",
            "25th percentile (error)",
            "50th percentile (error)",
            "75th percentile (error)",
        ]
    elif problem_type == "classification":
        return [
            "Balanced_Accuracy",
            "F1_score",
            "Precision",
            "Recall",
            "ROC_AUC",
            "kappa_score",
            "deltaAUPRC",
            "prob_threshold",
            "Balanced_Accuracy_default",
            "F1_score_default",
            "Precision_default",
            "Recall_default",
            "kappa_score_default",
        ]
    else:
        raise ValueError("ERROR: problem_type not known")
