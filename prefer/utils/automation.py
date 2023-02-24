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
import sys
import tempfile

import pandas as pd


from prefer.src.benchmarking import Benchmarking
from prefer.molecule_representations.fingerprints_representations_builder import (
    FingerprintsRepresentationsBuilder,
)

from prefer.src.vector_molecule_representations import VectorMoleculeRepresentations
from prefer.molecule_representations.descriptors2D_representations_builder import (
    Descriptors2DRepresentationsBuilder,
)
from prefer.utils.save_load import saving_procedure_autosklearn
from prefer.utils.data_preparation import prepare_data
from prefer.utils.post_processing_and_optimization_helpers import nested

import time


def run_prefer(
    molecular_descriptors, problem_type, final_folder_path="./results_prefer", model_instance=None,
):
    """
    inputs:
    molecular_descriptors: list of molecular representations to test
    models_params: dictionary of parameters for different models to evaluate

    outputs:
    bench_list: list of benchmarking objects
    """
    bench_list = []
    dir_destination = None
    for molecular_descriptor in molecular_descriptors:
        representation_name = molecular_descriptor.representation_name
        with tempfile.TemporaryDirectory() as tmpdirname:
            print(f"---The current representation is: {representation_name}---")
            print("Check multitasking ...")
            if "Property_2" in molecular_descriptor.df.columns.values:
                print("Multitasking set")
            else:
                print("Singletask set")
            bench = Benchmarking(
                problem_type=problem_type,
                working_directory=tmpdirname,
                model_instance=model_instance,
            )
            bench.benchmark(
                [molecular_descriptor], experiment_name=molecular_descriptor.experiment_name
            )
            bench_list.append(bench)
            # saving procedure
            timestr = time.strftime("%Y%m%d-%H%M%S")
            name = representation_name
            try:
                if not os.path.exists(final_folder_path):
                    os.mkdir(final_folder_path)
            except OSError as e:
                print(e)
                logging.error("Creation of the directory %s failed", final_folder_path)
            else:
                logging.info("Successfully created the directory %s ", final_folder_path)
            dir_destination = final_folder_path + "/" + name + "_" + timestr
            saving_procedure_autosklearn(bench, dir_destination)
    return bench_list, dir_destination


def merge_table_metrics(bench_list):
    collect_df = {}
    for bench in bench_list:
        bench.create_summary_table()
        for index, cols in enumerate(bench.table_metrics.columns.values):
            bench.table_metrics.rename(
                columns={
                    bench.table_metrics.columns[index]: bench.experiment_name
                    + ":"
                    + bench.table_metrics.columns[index]
                },
                inplace=True,
            )
        if bench.experiment_name not in collect_df:
            collect_df[bench.experiment_name] = [bench.table_metrics]
        else:
            collect_df[bench.experiment_name].append(bench.table_metrics)
    for experiment_name in collect_df.keys():
        merged = pd.concat(collect_df[experiment_name], axis=1)
    return merged


def data_preparation(data_info):

    # Read your .csv files
    try:
        df = pd.read_csv(data_info["path_to_data"])
    except Exception:
        df = pd.read_csv(data_info["path_to_data"], sep=";")
    id_column_name = data_info[
        "id_column_name"
    ]  # name of the column where the molecules IDs are stored
    smiles_column_name = data_info[
        "smiles_column_name"
    ]  # name of the column where the SMILEs are stored
    properties_column_name_list = data_info[
        "properties_column_name_list"
    ]  # name of the columns where the properties to model are stored

    if data_info["temporal_info_column_name"]:
        temporal_info_column_name = data_info["temporal_info_column_name"]
        # in prepare_data now the dataset is both prepared and filtered
        df = prepare_data(
            df,
            id_column_name,
            smiles_column_name,
            properties_column_name_list,
            temporal_info_column_name,
        )
    else:
        # in prepare_data now the dataset is both prepared and filtered
        df = prepare_data(
            df, id_column_name, smiles_column_name, properties_column_name_list
        )  
    return df


def generate_molecular_representations(
    df, split_type, experiment_name, list_of_model_based_representations_paths=None
):
    """
    Function to generate molecular representation objects, both traditional and model_based (these should be already computed and store in location appended in the list_of_model_based_representations_paths)
    """
    fing_representation = FingerprintsRepresentationsBuilder()
    _2d_descriptors = Descriptors2DRepresentationsBuilder()

    fingerprints = fing_representation.build_representations(df, split_type=split_type)
    fingerprints.experiment_name = experiment_name

    _2dd = _2d_descriptors.build_representations(df, split_type=split_type)
    _2dd.experiment_name = experiment_name
    dict_of_representations = dict()
    dict_of_representations["FINGERPRINTS"] = fingerprints
    dict_of_representations["2DDESCRIPTORS"] = _2dd

    if list_of_model_based_representations_paths:
        for path in list_of_model_based_representations_paths:
            model_name = path.split("_")[0]

            model_name = model_name.replace(".", "")
            model_name = model_name.replace("/", "")
            vector_repr = VectorMoleculeRepresentations(
                df=pd.DataFrame(), representation_name="", split_type=" "
            )
            model_based_representation = vector_repr.load(path)
            model_based_representation.experiment_name = experiment_name
            model_based_representation.representation_name = model_name
            model_based_representation.split_type = split_type

            dict_of_representations[model_name] = model_based_representation

    return dict_of_representations


def run(
    representations, problem_type, model_instance=None, final_folder_path="./PREFER_results",
):
    # Create main folder to store results if not yet in place
    try:
        if not os.path.exists(final_folder_path):
            os.mkdir(final_folder_path)
    except OSError as e:
        print(e)
        logging.error("Creation of the directory %s failed", final_folder_path)
    else:
        logging.info("Successfully created the directory %s ", final_folder_path)

    if final_folder_path.endswith("/"):
        final_folder_path = final_folder_path[:-1]

    timestr = time.strftime("%Y%m%d-%H%M%S")
    final_folder_path = f"{final_folder_path}/session_{timestr}"
    molecular_descriptors_list = []
    representation_name_list = []
    for key, item in representations.items():
        molecular_descriptors_list.append(item)
        representation_name_list.append(key)
    # run PREFER
    bench_list, dir_destination = run_prefer(
        molecular_descriptors_list,
        problem_type,
        final_folder_path=final_folder_path,
        model_instance=model_instance,
    )

    return bench_list, final_folder_path


def create_comparison_table(df, metric_regression = "R2", metric_classification = "ROC_AUC"):
    import ast

    if df.index.name != "Metrics":
        if "Metrics" in df.columns.values:
            df.index = df.Metrics
            df.drop(columns = ['Metrics'], inplace = True)
        else:
            raise ValueError(
                f"Indices of the passed dataframe do not contain the Metrics used in the evaluation."
            )
            
    if metric_regression in df.index:
        metric = metric_regression
    elif metric_classification in df.index:
        metric = metric_classification
    else:
        raise ValueError(
            f"Metric =  {df.index} do not contain neither {metric_classification} nor {metric_regression}. Cannot compute comparison table."
        )


    experiments_dict = {}
    tmp_dict = nested()
    if df.columns.values.shape[0] == 1:
        experiment_name = df.columns[0].split(":")[0]
    else:
        experiment_name = df.columns[1].split(":")[0]

    # Convert indices into column of the dataframe
    df = df.reset_index()
    value = df[df.Metrics == metric][df.columns[1]].values[0]
    # convert the list into a string to reorganize the results
    stringA = str(value)

    res = stringA.strip("][").split(", ")
    if len(res) == 1:
        multitasking = False
    else:
        multitasking = True
        n_tasks = len(res)

    # back to indices
    df = df.set_index("Metrics")
    if multitasking:
        experiment_name = df.columns[1].split(":")[0]
        cols = df.columns
        for col in cols:
            col_tmp = col.split(":")[1]
            representation = col_tmp.split(",")[0]
            a_string = str(df.loc[metric, col]).strip("][").replace(",", "")
            a_list = a_string.split()
            map_object = map(float, a_list)

            list_of_integers = list(map_object)
            df.loc[metric, col] = str(list_of_integers)
            df = df.loc[metric, :]
            df = df.to_frame().T
        cols = [x for x in df.columns.values]
        for col in cols:
            col_tmp = col.split(":")[1]
            representation = col_tmp.split(",")[0]
            for n in range(0, n_tasks):
                elem = ast.literal_eval(df.loc[metric, col])
                df.loc[metric, experiment_name + f"_{n+1}" + ":" + representation] = elem[n]
        for n in range(0, n_tasks):
            experiments_dict[experiment_name + f"_{n+1}"] = metric
        df.drop(cols, axis=1, inplace=True)
    else:
        df = df.loc[metric, :]
        df = df.to_frame().T
        experiments_dict[experiment_name] = metric

    for col in df.columns:
        experiment_name = col.split(":")[0]
        col_tmp = col.split(":")[1]
        representation = col_tmp.split(",")[0]
        if not multitasking:
            df[col] = pd.to_numeric(df[col], downcast="float")

        tmp_dict[experiment_name][representation][experiment_name] = df.loc[metric, col]

    return experiments_dict, tmp_dict
