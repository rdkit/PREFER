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

import logging
import os
import pickle
import sys
import tempfile
from distutils.dir_util import copy_tree

from shutil import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras.models


from prefer.utils.data_utils import save_png_as_html

from sklearn.metrics import (
    confusion_matrix,
    auc,
    roc_curve,
)


def create_directory(dir_name):
    """
    function to create another directory
    """

    try:
        os.makedirs(dir_name)
        logging.info("Directory %s created", dir_name)
    except FileExistsError:
        logging.info("Directory %s already exists", dir_name)


def saving_procedure_autosklearn(bench, dir_destination, drop_df_in_bench: bool = False):
    """
    function to store all the results of the benchmarking job; in particular the benchmarking object and the best models.
    """

    try:
        os.makedirs(dir_destination)
        logging.info("Directory %s created", dir_destination)
    except FileExistsError:
        logging.info("Directory %s already exists", dir_destination)
    dir_ = dir_destination + "/"
    # Save dataframe in the dir_destination as a .csv
    merged = pd.DataFrame()
    first_time = True
    for representation in bench.df.keys():
        if first_time:
            merged = bench.df[representation]
            first_time = False
        else:
            merged = merged.merge(bench.df[representation], on="ID")

    merged.to_csv(f"{dir_}df_complete.csv", index=False)
    merged.to_json(f"{dir_}df_complete.json")
    merged.to_pickle(f"{dir_}df_complete.pkl")
    if drop_df_in_bench:
        bench.df = dict()
    # Save bench
    bench.save(dir_ + "bench")


def save_models_autosklearn(model, dir_: str):
    with open(f"{dir_}.pkl", "wb") as f:
        pickle.dump(model, f)
    logging.info(f"Saved model to disk ({dir_}.pkl)")


def load_models_autosklearn(path_to_model):
    with open(f"{path_to_model}.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model


def save_combined_results(
    benchs, dir_: str, model_type: str, name: str = "PREFER_comparison_figure"
):
    """
    Function to combine the results of a list of benchs in the form of a table and different plots

    inputs:
    - benchs: list of benchmarking objects to combine
    - dir_: string reporting the directory where to save the combined results
    - model_type: string reporting the type of problem - classification or regression

    """
    if dir_.endswith("/"):  # Normalise away trailing slashes
        dir_ = dir_[:-1]
    # find maximum number of columns
    max_cols = 0
    for bench in benchs:
        best_grid = bench.best_grid
        test_labels = bench.test_labels
        representation_names = bench.representations
        repres_name = representation_names[0]
        model_names = bench.models_params["model_names"]
        if best_grid[repres_name]:
            model_name = model_names[0]  # we should have only one model at a time

            if not isinstance(test_labels[repres_name][model_name], pd.DataFrame):
                test_labels_df = pd.DataFrame(test_labels[repres_name][model_name].tolist())
            else:
                test_labels_df = test_labels[repres_name][model_name]
            if len(test_labels_df.columns) > max_cols:
                max_cols = len(test_labels_df.columns)

    if model_type == "regression":
        fig, axes = plt.subplots(len(benchs), max_cols, figsize=(15, 22))
        for index_bench, bench in enumerate(benchs):
            df_name = bench.experiment_name
            best_grid = bench.best_grid
            predictions_test = bench.predictions_test
            test_labels = bench.test_labels
            representation_names = bench.representations
            y_score = bench.y_score
            mask_value = bench.mask_value
            if not mask_value:
                mask_value = -1000
            repres_name = representation_names[0]
            model_names = bench.models_params["model_names"]

            if best_grid[repres_name]:
                model_name = model_names[0]  # we should have only one model at a time

                if not isinstance(test_labels[repres_name][model_name], pd.DataFrame):
                    test_labels_df = pd.DataFrame(test_labels[repres_name][model_name].tolist())
                else:
                    test_labels_df = test_labels[repres_name][model_name]

                for index, col in enumerate(test_labels_df.columns):
                    predictions_df = pd.DataFrame(predictions_test[repres_name][model_name]).loc[
                        :, index
                    ]
                    ytest_tmp = test_labels_df[col]
                    ytest = test_labels_df
                    index_ = ytest[ytest[col] != mask_value][col].index.values
                    ytest_tmp = ytest[col].iloc[index_]
                    ytest_tmp = ytest_tmp.reset_index(drop=True)
                    predictions_df = predictions_df.iloc[index_]
                    predictions_df = predictions_df.reset_index(drop=True)
                    if len(benchs) == 1:
                        ax = axes[index]
                    else:
                        if len(test_labels_df.columns) == 1:
                            ax = axes[index_bench]
                        else:
                            ax = axes[index_bench][index]

                    xmin = min(predictions_df.values)
                    xmax = max(predictions_df.values)
                    ymin = min(ytest_tmp.values)
                    ymax = max(ytest_tmp.values)
                    min_ = min([xmin, ymin])
                    max_ = max([xmax, ymax])
                    ax.set(xlim=(min_, max_), ylim=(min_, max_))
                    ax.plot([min_, max_], [min_, max_], "r", alpha=0.2)
                    m, b = np.polyfit(predictions_df, ytest_tmp, 1)
                    ax.plot(predictions_df, m * predictions_df + b, alpha=0.2)
                    ax.legend(["y=x", "interpolation"])

                    ax.scatter(predictions_df, ytest_tmp, alpha=0.15, marker=".")

                    ax.set_xlabel("predictions", fontsize=12, fontweight="bold")
                    ax.set_ylabel("real value", fontsize=12, fontweight="bold")
                    ax.set_title(df_name + ": " + model_name + " - " + repres_name)
                    ax.spines["right"].set_visible(False)
                    ax.spines["top"].set_visible(False)
            else:
                continue

        fig.tight_layout(pad=3.0)
        save_png_as_html(fig, dir_ + "/" + name + ".html")
        fig.savefig(
            dir_ + "/" + name + ".png", transparent=True,
        )

    elif model_type == "classification":
        fig, axes = plt.subplots(
            len(benchs) * 2, max_cols, figsize=((15 * max_cols) / 2, (30 * len(benchs) * 2) / 8)
        )
        index_bench = 0
        for index_bench_mv, bench in enumerate(benchs):
            df_name = bench.experiment_name
            best_grid = bench.best_grid
            predictions_test = bench.predictions_test
            test_labels = bench.test_labels
            representation_names = bench.representations
            y_score = bench.y_score
            mask_value = bench.mask_value
            if not mask_value:
                mask_value = -1000
            repres_name = representation_names[0]
            model_names = bench.models_params["model_names"]

            if best_grid[repres_name]:
                model_name = model_names[0]  # we should have only one model at a time
                y_score_only = pd.DataFrame(y_score[repres_name][model_name])
                # check dimension consistency
                rows = y_score_only.shape[0]
                columns = y_score_only.shape[1]
                if rows < columns:
                    y_score_only = y_score_only.T
                if not isinstance(test_labels[repres_name][model_name], pd.DataFrame):
                    test_labels_df = pd.DataFrame(test_labels[repres_name][model_name].tolist())
                else:
                    test_labels_df = test_labels[repres_name][model_name]
                y_test_only = test_labels_df
                for index, col in enumerate(test_labels_df.columns):
                    predictions_df = pd.DataFrame(predictions_test[repres_name][model_name]).loc[
                        :, index
                    ]

                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    fpr, tpr, _ = roc_curve(
                        y_test_only.iloc[:, index].values[
                            y_test_only.iloc[:, index].values != mask_value
                        ],
                        y_score_only.iloc[:, index].values[
                            y_test_only.iloc[:, index].values != mask_value
                        ],
                    )
                    roc_auc = auc(fpr, tpr)
                    lw = 2

                    if len(test_labels_df.columns) == 1:
                        ax = axes[index_bench + 1]
                    else:
                        ax = axes[index_bench + 1][index]
                    ax.plot(
                        fpr,
                        tpr,
                        color="darkorange",
                        lw=lw,
                        label="ROC curve (area = %0.2f)" % roc_auc,
                    )
                    ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
                    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
                    ax.set_xlabel("False Positive Rate", fontsize=18.0)
                    ax.set_ylabel("True Positive Rate", fontsize=18.0)
                    ax.set_title(
                        df_name + ":" + model_name + "_" + repres_name,
                        fontsize=20,
                        fontweight="bold",
                    )
                    matrix = confusion_matrix(
                        y_test_only.iloc[:, index].values[
                            y_test_only.iloc[:, index].values != mask_value
                        ],
                        predictions_df[y_test_only.iloc[:, index].values != mask_value],
                    )
                    sns.set(font_scale=2)
                    if len(test_labels_df.columns) == 1:
                        sns.heatmap(
                            matrix,
                            annot=True,
                            xticklabels=["Pred_N", "Pred_P"],
                            yticklabels=["Real_N", "Real_P"],
                            ax=axes[index_bench],
                        )
                    else:
                        sns.heatmap(
                            matrix,
                            annot=True,
                            xticklabels=["Pred_N", "Pred_P"],
                            yticklabels=["Real_N", "Real_P"],
                            ax=axes[index_bench][index],
                        )
                plt.tight_layout()

            else:
                continue
            index_bench = index_bench + 2
        fig.tight_layout(pad=3.0)
        save_png_as_html(fig, dir_ + "/" + name + ".html")
        fig.savefig(
            dir_ + "/" + name + ".png", transparent=True,
        )

    else:
        raise ValueError(
            f"{model_type} is not a valid name. You can only have regression or classification"
        )

    return
