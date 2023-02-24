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


"""helpers for post processing and optimization of the benchmarking results"""
import os
import sys


# Pandas, numpy, scipy, pyplot and seaborn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from rdkit import rdBase
import warnings

sns.set()
plt.style.use("fivethirtyeight")
rdBase.DisableLog("rdApp.error")
warnings.filterwarnings("ignore")
tf.keras.backend.clear_session()


class nested(dict):

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


def create_comparison_table_with_path(path_to_tables):
    """
    This function is used to create the final comparison table among different datasets (distinguishing regression and classification problems)
    Input:
    - path_to_tables: path where the table metrices from the benchmarking objects are stored

    Output:
    - experiments_dict: dictionary where the name of the experiments with the corresponding metric of interest are stored
    - tmp_dict: dictionary where for each experiment, model and representation, the metric value is stored
    """
    files = [f for f in os.listdir(path_to_tables) if ".ipynb_checkpoints" not in f]
    experiments_dict = {}
    tmp_dict = nested()
    df_list = []
    for file in files:
        df = pd.read_csv(path_to_tables + "/" + file)
        df_list.append(df)
        experiment_name = df.columns[1].split(":")[0]
        if "R2" in df.Metrics.values:  # regression
            metric = "R2"
        else:
            metric = "ROC_AUC"
        stringA = df[df.Metrics == metric][df.columns[1]].values[0]
        res = stringA.strip("][").split(", ")
        if len(res) == 1:
            multitasking = False
        else:
            multitasking = True
        df = df.set_index("Metrics")
        if multitasking:
            cols = df.columns
            experiment_name = df.columns[1].split(":")[0]
            for col in cols:
                col_tmp = col.split(":")[1]
                representation = col_tmp.split(",")[0]
                model = col_tmp.split(",")[1]
                a_string = df.loc[metric, col].strip("][").replace(",", "")
                a_list = a_string.split()
                map_object = map(float, a_list)

                list_of_integers = list(map_object)
                df.loc[metric, col] = list_of_integers
                df = df.loc[metric, :]
                df = df.to_frame().T
            cols = [x for x in df.columns.values]
            for col in cols:
                col_tmp = col.split(":")[1]
                representation = col_tmp.split(",")[0]
                model = col_tmp.split(",")[1]
                df.loc[
                    metric, experiment_name + "_1" + ":" + representation + "," + model
                ] = df.loc[metric, col][0]
                df.loc[
                    metric, experiment_name + "_2" + ":" + representation + "," + model
                ] = df.loc[metric, col][1]
            experiments_dict[experiment_name + "_1"] = metric
            experiments_dict[experiment_name + "_2"] = metric
            df.drop(cols, axis=1, inplace=True)
        else:
            df = df.loc[metric, :]
            df = df.to_frame().T
            experiments_dict[experiment_name] = metric
        for col in df.columns:
            experiment_name = col.split(":")[0]
            col_tmp = col.split(":")[1]
            representation = col_tmp.split(",")[0]
            model = col_tmp.split(",")[1]
            df[col] = pd.to_numeric(df[col], downcast="float")
            tmp_dict[experiment_name][representation][model + "_" + experiment_name] = df.loc[
                metric, col
            ]

    return experiments_dict, tmp_dict


def plt_heat_map(dataframe, title, vmin=0.0, vmax=0.0, dir_to_save="."):
    """
    Function to plot the heat map of the input dataframe
    Inputs:
    - dataframe: dataframe to plot as heat map
    - title: string
    - dir_to_save: string indicating the directory where to save the plots
    """
    f, ax = plt.subplots(figsize=(12, 4))
    if vmin == 0.0 and vmax == 0.0:
        ax = sns.heatmap(dataframe, cmap="YlGnBu", linewidths=5)
    else:
        ax = sns.heatmap(dataframe, cmap="YlGnBu", linewidths=5, vmin=vmin, vmax=vmax)
    ax.axes.set_title(title, fontsize=20)
    ax.tick_params(labelsize=15)
    if dir_to_save:
        if dir_to_save.endswith("/"):  # Normalise away trailing slashes
            dir_to_save = dir_to_save[:-1]
        plt.savefig(
            dir_to_save + "/" + title + "_heatMap.png",
            transparent=True,
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()


def create_heat_map(experiments_dict, tmp_dict, dir_to_save=None, name=""):
    """
    Function to create the heat map given the output of the create_comparison_table function.
    Inputs:
    - experiments_dict: dictionary where the name of the experiments with the corresponding metric of interest are stored
    - tmp_dict: dictionary where for each experiment, model and representation, the metric value is stored
    - dir_to_save: string indicating the directory where to save the plots
    """
    df_list_regr = []
    df_list_class = []
    if len(experiments_dict.keys()) > 1:
        for experiment in experiments_dict.keys():
            if experiments_dict[experiment] in ["R2", "RMSE_test_Norm"]:
                df_list_regr.append(pd.DataFrame.from_dict(tmp_dict[experiment]).T)
            elif experiments_dict[experiment] in ["ROC_AUC", "Balanced_Accuracy", "F1_score", "deltaAUPRC"]:
                df_list_class.append(pd.DataFrame.from_dict(tmp_dict[experiment]).T)
        if df_list_regr != []:
            df_all_regr = pd.concat(df_list_regr, axis=1)
            df_all_regr = df_all_regr.fillna(0)
            plt_heat_map(
                df_all_regr, f"df_all_regr_{name}", vmin=0, vmax=0.8, dir_to_save=dir_to_save
            )
        if df_list_class != []:
            df_all_class = pd.concat(df_list_class, axis=1)
            df_all_class = df_all_class.fillna(0)
            plt_heat_map(
                df_all_class, f"df_all_class_{name}", vmin=0.5, vmax=1, dir_to_save=dir_to_save
            )

    else:
        for experiment in experiments_dict.keys():
            if experiments_dict[experiment] in ["R2", "RMSE_test_Norm"]:
                df_all_regr = pd.DataFrame.from_dict(tmp_dict[experiment]).T
                df_all_regr = df_all_regr.fillna(0)
                plt_heat_map(
                    df_all_regr, f"df_all_regr_{name}", vmin=0, vmax=0.8, dir_to_save=dir_to_save
                )
            elif experiments_dict[experiment] in ["ROC_AUC", "Balanced_Accuracy", "F1_score"]:
                df_all_class = pd.DataFrame.from_dict(tmp_dict[experiment]).T
                df_all_class = df_all_class.fillna(0)
                plt_heat_map(
                    df_all_class, f"df_all_class_{name}", vmin=0.5, vmax=1, dir_to_save=dir_to_save
                )
    return


