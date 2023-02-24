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
import os
import pickle
import sys
from datetime import date
import time
import pandas as pd

from prefer.utils.run_automl import run_automl
from prefer.utils.models_evaluation import show_results
from prefer.utils.models_evaluation import plot_results
from prefer.utils.save_load import save_models_autosklearn, load_models_autosklearn
from prefer.utils.models_utils import (
    retrieve_metrics_names,
    retrieve_models_names,
    check_if_small_data,
    output_dataframe_preparation,
)
from prefer.src.molecule_representations import MoleculeRepresentations


cwd = os.path.dirname(__file__)


class Benchmarking:

    """
    Benchmarking class to benchmark different models for different features (e.g. molecular representations) and
    different end-points (e.g. molecular properties)
    This is a child class of the PropertyPredictionModels class that can be used also in a more general scenario.

    inputs:
    - df: initial dataframe
    - problem_type: string defining the type of the problem; e.g. 'regression' or 'classification'
    - experiment_name: string defining the name of the experiment

    output: collections of final results of the best estimators found for each combination of model type,
            representation and property.
    """

    def __init__(
        self, problem_type: str = "", working_directory=None, model_instance: list = None,
    ):

        self.experiment_name = None
        self.df = dict()
        self.problem_type = problem_type
        self.metrics_names = retrieve_metrics_names(problem_type)
        self.model_instance = model_instance
        self.metrics = dict()
        self.best_grid = dict()
        self.predictions_test = dict()
        self.predictions_train = dict()
        self.test_labels = dict()
        self.y_score = dict()
        self.best_estimator = dict()
        self.best_estimator_scores = dict()
        self.test_labels_outer_cv = dict()
        self.test_predictions_outer_cv = dict()
        self.best_params_outer_cv = dict()
        self.table_metrics = pd.DataFrame()
        self.representations = []
        self.final_model = dict()
        self.today = []
        self.molecule_representations_obj_list = []
        self.models_ids = (
            {}
        )  # this is a dictionary to store, for each representation, info related to the generator models ids
        self.working_directory = working_directory
        self.features_scaling_type = dict()
        self.features_means_vect = dict()
        self.features_stds_vect = dict()
        self.path_to_results = []

    def create_model_selection_params(self, Xtrain, ytrain, Xtest, ytest) -> dict:
        """
        method to set the model_selection_params needed for the benchmarking loop.
        Information provided by the user at the class instance level will be used as well as test and train sets
        obtained by splitting the initial dataset according to a specific splitting strategy.
        """
        if not self.metrics_names:
            self.metrics_names = retrieve_metrics_names(self.problem_type)

        model_selection_params = {
            "X_test": Xtest,
            "y_test": ytest,
            "X_tr": Xtrain,
            "y_tr": ytrain,
            "model_type": self.problem_type,
            "metrics_names": self.metrics_names,
        }
        return model_selection_params

    def store_outputs(self, outputs, representation: MoleculeRepresentations):
        """
        method to store results from the benchmark job.
        """
        representation_name = representation.representation_name
        df = representation.df
        self.metrics[representation_name] = outputs["metrics"]
        self.predictions_test[representation_name] = outputs["predictions_test"]
        self.predictions_train[representation_name] = outputs["predictions_train"]
        self.test_labels[representation_name] = outputs["test_labels"]
        self.y_score[representation_name] = outputs["y_score"]
        self.best_estimator[representation_name] = outputs["best_estimator"]
        self.df[representation_name] = df

    def benchmark(
        self, representations: list, experiment_name: str = "new_benchmarking",
    ):

        """
        method to perform the benchmarking.
        input:
        - list of MoleculeRepresentations objects to test
        """
        if not self.working_directory:
            local_path = os.path.join(cwd, "..")
            timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
            path = local_path + "/working_directory_" + str(os.getpid()) + "_" + timestamp
            os.mkdir(path)
            self.working_directory = path
        else:
            if self.working_directory.endswith("/"):  # Normalise away trailing slashes
                self.working_directory = self.working_directory[:-1]

        self.molecule_representations_obj_list = representations
        self.experiment_name = experiment_name

        for representation in representations:
            small_data = check_if_small_data(
                representation
            )  # evaluate whether your dataset can be considered to be 'small' to set the corresponding hyperparams
            Xtrain, ytrain, Xtest, ytest, index_train, index_test = representation.split(
                return_indices=True
            )
            # Storing info related to the representation
            self.features_scaling_type[
                representation.representation_name
            ] = (
                representation.scale_type
            )  # it is None in case no scaling has been applied to the features
            self.features_means_vect[
                representation.representation_name
            ] = representation.features_means
            self.features_stds_vect[
                representation.representation_name
            ] = representation.features_stds

            # do some conversion to enable valid pickling in case scalling is applied
            if representation.scale_type is not None:
                self.features_means_vect[
                    representation.representation_name
                ] = self.features_means_vect[representation.representation_name].to_numpy()
                self.features_stds_vect[
                    representation.representation_name
                ] = self.features_stds_vect[representation.representation_name].to_numpy()

                # if features have been scaled, check if the scalining has been done only wrt mean and std of training set
                conditions = []
                std_train = pd.DataFrame(Xtrain).std()
                mean_train = pd.DataFrame(Xtrain).mean()
                std_test = pd.DataFrame(Xtest).std()
                mean_test = pd.DataFrame(Xtest).mean()
                conditions.append(all(round(mean_train, 3) == 0))
                conditions.append(not all(round(mean_test, 3) == 0))
                if representation.scale_type == "standardization":
                    conditions.append(all(round(std_train, 3) == 1.0))
                    conditions.append(not all(round(std_test, 3) == 1.0))
                if not all(conditions):
                    raise RuntimeError("Something is wrong with the feature scaling")

            self.models_ids[
                representation.representation_name
            ] = (
                representation.model_id
            )  # Associate to each molecular representation the coresponding id important in case of data-driven molecular representations to correctly identify the generator model.

            model_selection_params = self.create_model_selection_params(
                Xtrain, ytrain, Xtest, ytest
            )

            outputs = run_automl(
                model_selection_params,
                representation=representation,
                working_directory=self.working_directory,
                model_instance=self.model_instance,
                small_data=small_data,
            )

            representation.df = output_dataframe_preparation(
                representation.df,
                index_train,
                index_test,
                outputs["predictions_train"],
                outputs["predictions_test"],
            )

            representation.df.rename(
                columns={"molecule_representation": representation.representation_name,},
                inplace=True,
            )

            property_cols = [col for col in representation.df.columns if "Property" in col]
            for index, property_col in enumerate(property_cols):
                representation.df.rename(
                    columns={f"Property_{index+1}": f"true_label_{index+1}",}, inplace=True,
                )
            self.store_outputs(outputs, representation)
            self.representations.append(representation.representation_name)

    def create_summary_table(self, plot_res=False):
        """
        method to generate a table with all the results obtained from the benchmarking
        to visualize the results one can call self.table_metrics
        """
        model_names = retrieve_models_names(self)
        if self.table_metrics.empty:
            self.table_metrics = show_results(
                self.experiment_name,
                self.metrics,
                self.metrics_names,
                self.representations,
                model_names,
                plot_res=plot_res,
            )

    def plot_res(self, path=None):
        """
        method to plot the main results of the benchmarking and save them in the dir_to_save
        """
        plot_results(
            self.experiment_name,
            self.problem_type,
            self.metrics,
            self.predictions_test,
            self.test_labels,
            self.representations,
            path,
        )

    def save(self, path: str):
        """
        method to save the benchmarking object at the position indicated by path as .pkl.
        The best autosklearn model will be saved separately as .pkl

        Usage:
        bench.save('./data/bench')
        """
        print(f"Saving bench and each model (one for each molecular representation, in {path})")
        today = date.today()
        self.today = today
        for representation in self.representations:
            autosklearn_model = self.best_estimator[representation]
            model_name = path.replace("bench", "best_model_refit")
            print(f"Saving autosklearn model in {model_name}_{representation}")
            save_models_autosklearn(
                autosklearn_model, model_name + "_" + representation,
            )

        with open(path + ".pkl", "wb") as output:
            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)
        print(f"bench object saved as {path}.pkl")

    def load(self, path_to_class: str, path_to_model: str = None, compileKeras: bool = False):
        """
        Method to load the benchmarking object as well as the corresponding best models (separately saved).
        input:
        - path_to_class is the path where the benchmarking object as been stored as .pkl
        - path_to_model is the path where the best models (autosklearn) have been saved
        Usage:
        bench = Benchmarking(problem_type = 'regression')
        bench.load('./data/')
        """
        if not path_to_class.endswith("/"):
            path_to_class = path_to_class + "/"
        if not path_to_model:
            path_to_model = path_to_class
        elif not path_to_model.endswith("/"):
            path_to_model = path_to_model + "/"

        with open(path_to_class + "bench.pkl", "rb") as input:
            tmp = pickle.load(input)

        self.df = tmp["df"]
        self.metrics = tmp["metrics"]
        self.best_grid = tmp["best_grid"]
        self.predictions_test = tmp["predictions_test"]
        self.predictions_train = tmp["predictions_train"]
        self.test_labels = tmp["test_labels"]
        self.y_score = tmp["y_score"]
        self.best_estimator_scores = tmp["best_estimator_scores"]
        self.table_metrics = tmp["table_metrics"]
        self.representations = tmp["representations"]
        self.problem_type = tmp["problem_type"]
        self.metrics_names = tmp["metrics_names"]
        self.test_labels_outer_cv = tmp["test_labels_outer_cv"]
        self.test_predictions_outer_cv = tmp["test_predictions_outer_cv"]
        self.best_params_outer_cv = tmp["best_params_outer_cv"]
        self.molecule_representations_obj_list = tmp["molecule_representations_obj_list"]
        self.experiment_name = tmp["experiment_name"]

        if "models_ids" in tmp:
            self.models_ids = tmp["models_ids"]

        if "today" in tmp:
            self.today = tmp["today"]
        if "working_directory" in tmp:
            self.working_directory = tmp["working_directory"]

        if "features_scaling_type" in tmp:
            self.features_scaling_type = tmp["features_scaling_type"]
        if "features_means_vect" in tmp:
            self.features_means_vect = tmp["features_means_vect"]
        if "features_stds_vect" in tmp:
            self.features_stds_vect = tmp["features_stds_vect"]

        models = dict()
        # load models
        for representation in self.representations:
            models[representation] = load_models_autosklearn(
                path_to_model + "best_model_refit_" + representation
            )
        self.best_estimator = models
