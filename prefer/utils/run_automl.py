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


#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys


# Add PREFER dependency
from prefer.utils.models_utils import get_autosklearn_customized_model
from prefer.utils.models_evaluation import compute_model_performances
from prefer.utils.save_load import save_models_autosklearn

# Add AutoSklearn dependencies
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.metrics import balanced_accuracy, r2


def compute_default_automl(time_left_for_this_task, per_run_time_limit, num_folds, memory_limit, working_directory, multitasking, model_type):
    if model_type == "regression":
        if(multitasking):
            default_automl = AutoSklearnRegressor(
                time_left_for_this_task=time_left_for_this_task,
                per_run_time_limit=per_run_time_limit,
                memory_limit = memory_limit,
                # One can use models trained during cross-validation directly to predict
                # for unseen data. For this, all k models trained during k-fold
                # cross-validation are considered as a single soft-voting ensemble inside
                resampling_strategy="cv",
                resampling_strategy_arguments={"folds": num_folds},
                metric=r2,
                n_jobs=-1,
                tmp_folder=working_directory + "/_autosklearn",
            )

        else:
            default_automl = AutoSklearnRegressor(
                time_left_for_this_task=time_left_for_this_task,
                per_run_time_limit=per_run_time_limit,
                memory_limit = memory_limit,
                # One can use models trained during cross-validation directly to predict
                # for unseen data. For this, all k models trained during k-fold
                # cross-validation are considered as a single soft-voting ensemble inside
                resampling_strategy="cv",
                exclude = {
                    'regressor': ["liblinear_svr", "libsvm_svr"]
                },
                resampling_strategy_arguments={"folds": num_folds},
                metric=r2,
                n_jobs=-1,
                tmp_folder=working_directory + "/_autosklearn",
            )
    elif model_type == "classification":
        if(multitasking):
            default_automl = AutoSklearnClassifier(
                time_left_for_this_task=time_left_for_this_task,
                per_run_time_limit=per_run_time_limit,
                memory_limit = memory_limit,
                resampling_strategy="cv",
                resampling_strategy_arguments={"folds": num_folds},
                n_jobs=-1,
                tmp_folder=working_directory + "/_autosklearn",
            )

        else:
            default_automl = AutoSklearnClassifier(
                time_left_for_this_task=time_left_for_this_task,
                per_run_time_limit=per_run_time_limit,
                memory_limit = memory_limit,
                resampling_strategy="cv",
                resampling_strategy_arguments={"folds": num_folds},
                metric=balanced_accuracy,
                exclude = {
                'classifier': ["libsvm_svc", "liblinear_svc"]
                },
                n_jobs=-1,
                tmp_folder=working_directory + "/_autosklearn",
            )
    else:
        raise ValueError(f'{model_type} not known. Only classification or regression can be selected')


    return default_automl


def run_automl(
    model_selection_params: dict,
    representation,
    working_directory: str = None,
    model_instance: list = None,
    small_data: bool = False,
):
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(f"Current location is: {dir_path}")
    print(f"tmp_folder: {working_directory}/_autosklearn")

    """
    This function is used to launch an AutoSklearn job given a specific molecular representation

    Inputs:
    - model_selection_params, a dictionary with the following shape:
        model_selection_params = {
                "X_test": Xtest,
                "y_test": ytest,
                "X_tr": Xtrain,
                "y_tr": ytrain,
                "model_type": self.problem_type,
                "metrics_names": self.metrics_names,
            }
    - representation, a string with the name of the molecular representation under analysis
    - working_directory, a string indicating the directory where to store the results

    Outputs:
    - outputs, a dictionary with the shape:
        outputs = {
            "metrics": metrics, # dictionary with the regression/classification metrics used for the model evaluation on the test set
            "predictions_test": predict_test,
            "predictions_train": predict_train,
            "test_labels": model_selection_params["y_test"],
            "y_score": y_score,
            "best_estimator": refit_estimator,
        }

    """
    multitasking = False
    representation_name = representation.representation_name
    if not working_directory:
        working_directory = "."
    elif working_directory.endswith("/"):
        working_directory = working_directory[:-1]

    model_type = model_selection_params["model_type"]

    if small_data:
        # set param for small data
        time_left_for_this_task = 300  # seconds (5 minutes)
        per_run_time_limit = 30  # seconds
        num_folds = 5  # to avoid overfitting
        memory_limit = 6072 #MB
    else:
        # set param for big data
        time_left_for_this_task = 7200  # default
        per_run_time_limit = 720  # default
        num_folds = 5
        memory_limit = 6072 #MB
    if model_selection_params["y_tr"].ndim>1:
        multitasking = True
        
    # compute dafault setting automl model
    default_automl = compute_default_automl(time_left_for_this_task, per_run_time_limit, num_folds, memory_limit, working_directory, multitasking, model_type)
    
    if model_instance:
        try:
            print(f"Model instance was provided. Try to get AutoSklearn custom model")
            automl = get_autosklearn_customized_model(
                model_instance, model_type, working_directory
            )
        except Exception as e:
            print(
                f" The following error occurred while trying to set the customized autosklearnregressor: {e}"
            )
            print("Default regressor will be set instead")
            automl = default_automl
    else:
        automl = default_automl


    automl.fit(
        model_selection_params["X_tr"],
        model_selection_params["y_tr"],
        dataset_name=representation_name,
    )
    print("Autosklearn model fitted!")

    # refit with the entire training set
    # refit best estimator
    # From https://automl.github.io/auto-sklearn/master/examples/40_advanced/example_resampling.html
    # During fit(), models are fit on individual cross-validation folds. To use all available data, we call refit() which trains all models in the final ensemble on the whole dataset.
    print(
        "Refitting models with the entire training set. This procedure is needed when cv or holdout have been selected for the resampling startegy, as described in https://automl.github.io/auto-sklearn/master/api.html. If this is not the case refir may not be needed, but will be applied anyways."
    )
    refit_automl = automl.refit(model_selection_params["X_tr"], model_selection_params["y_tr"])
    # save model refitted
    model_output_name = working_directory + f"/best_model_refit_{representation_name}"
    print(f"Saving refitted model as: {model_output_name}.pkl")
    save_models_autosklearn(refit_automl, model_output_name)

    (metrics, predict_train, predict_test, y_score,) = compute_model_performances(
        model_type,
        model_selection_params["X_tr"],
        model_selection_params["y_tr"],
        model_selection_params["X_test"],
        model_selection_params["y_test"],
        model_selection_params["metrics_names"],
        refit_automl,
    )
    print("compute_model_performances completed!")
    outputs = {
        "metrics": metrics,
        "predictions_test": predict_test,
        "predictions_train": predict_train,
        "test_labels": model_selection_params["y_test"],
        "y_score": y_score,
        "best_estimator": refit_automl,  # fitted ONLY with the entire training set!
    }

    return outputs
