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


#!/usr/bin/python

import shutil

from prefer.src.benchmarking import Benchmarking
from prefer.utils.models_evaluation import save_html
from prefer.utils.save_load import save_combined_results
import pandas as pd
import os
import argparse
import json


def extract_folders(original_flder):

    dirfiles = os.listdir(original_flder)
    fullpaths = map(lambda name: os.path.join(original_flder, name), dirfiles)
    dirs = []
    for file in fullpaths:
        if (os.path.isdir(file)) and (file[0] not in [".", "_"]):
            dirs.append(file)
    return dirs[0]


def combine_results_from_args(
    store_result_folder,
    problem_type,
    benchs_folder1,
    benchs_folder2,
    benchs_folder3,
    benchs_folder4,
    experiment_name=None,
    save_json=True,
):

    """
    Script to combine all the results stored as separate bench objects in different folders, colelcted in the input list, computed from different PREFER runs.
    """

    collect_df = {}
    results = []
    collect_bench = []
    # create final folder
    os.makedirs(store_result_folder, exist_ok=True)

    if store_result_folder.endswith("/"):  # Normalise away trailing slashes
        store_result_folder = store_result_folder[:-1]

    all_folders = [benchs_folder1, benchs_folder2, benchs_folder3, benchs_folder4]

    for folder in all_folders:
        if folder:
            end = folder.split("/")[-1]
            if "." in end:
                folder = folder.split("/")[:-1]
                folder = "/".join(folder)
            folder = extract_folders(folder)
            # folder of interests
            final_dir = folder

            if not final_dir.endswith("/"):  # Normalise away trailing slashes
                final_dir = final_dir + "/"

            tmp = Benchmarking(problem_type=problem_type)
            try:
                tmp.load(final_dir)
                print("bench loaded")
                tmp.create_summary_table()
                print("summary_table computed")

                if experiment_name:
                    experiment_name_tmp = experiment_name
                else:
                    experiment_name_tmp = tmp.experiment_name
                tmp.table_metrics.rename(
                    columns={
                        tmp.table_metrics.columns[0]: experiment_name_tmp
                        + ":"
                        + tmp.table_metrics.columns[0]
                    },
                    inplace=True,
                )

                if experiment_name_tmp not in collect_df:
                    collect_df[experiment_name_tmp] = [tmp.table_metrics]
                else:
                    collect_df[experiment_name_tmp].append(tmp.table_metrics)
                collect_bench.append(tmp)

            except Exception as e:
                # WARNING?
                raise ValueError(
                    f"An error occurred with folder: {final_dir}. Benchmarking object cannot be imported. In particular: {e}",
                )

            # dump metrics for every model:
            metrics_dict = tmp.table_metrics.to_dict()
            print("metrics_dict created")
            # Note: As the df with the metric has only <representation>+<model type> as identificator
            # so we add there experiment name and in the body attach problem type.
            experiment_id = next(iter(metrics_dict))
            new_experiment_id = experiment_name_tmp + "," + experiment_id
            metrics_dict[new_experiment_id] = metrics_dict[experiment_id]
            metrics_dict[new_experiment_id]["Problem type"] = tmp.problem_type
            del metrics_dict[experiment_id]
            results.append(metrics_dict)
            print("metrics_dict appended")
        else:
            continue

    # Then save one json with all the experiments (dataset x model type):
    if save_json:
        with open(store_result_folder + "/" + "PREFER_comparison_table.json", "w") as jsonfile:
            json.dump(results, jsonfile)

    for key in collect_df.keys():
        merged = pd.concat(collect_df[key], axis=1)
        save_html(
            merged, df_name=key, path=store_result_folder + "/" + "PREFER_comparison_table.html",
        )
        merged.to_csv(store_result_folder + "/" + "PREFER_comparison_table.csv")
        merged.to_pickle(store_result_folder + "/" + "PREFER_comparison_table.pkl")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="combine results of different PREFER runs")
    parser.add_argument(
        "-bf1",
        "--benchs_folder1",
        type=str,
        help="path of the folder where results are stored",
        required=True,
    )

    parser.add_argument(
        "-bf2", "--benchs_folder2", type=str, help="path of the folder where results are stored",
    )

    parser.add_argument(
        "-bf3", "--benchs_folder3", type=str, help="path of the folder where results are stored",
    )

    parser.add_argument(
        "-bf4", "--benchs_folder4", type=str, help="path of the folder where results are stored",
    )

    parser.add_argument(
        "-srf",
        "--store_result_folder",
        type=str,
        help="path of the folder where results are stored",
    )

    parser.add_argument(
        "-pt", "--problem_type", type=str, help="problem_type: regression or classification",
    )

    args = parser.parse_args()
    combine_results_from_args(
        benchs_folder1=args.benchs_folder1,
        benchs_folder2=args.benchs_folder2,
        benchs_folder3=args.benchs_folder3,
        benchs_folder4=args.benchs_folder4,
        store_result_folder=args.store_result_folder,
        problem_type=args.problem_type,
    )
