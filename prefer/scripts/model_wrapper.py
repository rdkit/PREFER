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
import os
import argparse
import json
import sys
import time
import pickle
import ast
import logging
import numpy as np


from prefer.src.benchmarking import Benchmarking
from prefer.src.prefer_model_wrapper import PreferModelWrapper
from prefer.scripts.run_PREFER import retrieve_type_of_molecular_representation

logger = logging.getLogger(__name__)


import shutil


def copy_file(src, dest):
    try:
        shutil.copy(src, dest)
    except shutil.Error as e:
        print("Error: %s" % e)
    except IOError as e:
        print("Error: %s" % e.strerror)


def store_metadata(
    path_to_df,
    problem_type,
    experiment_name,
    id_column_name,
    smiles_column_name,
    properties_column_name_list,
    representation_name,
    final_folder_path,
    property_model_folder_path,
    repr_dir,
    desirability_scores,
    path_to_model=None,
):
    # TODO store info related to the entire hyperparams grids used to obtain results
    # Extract representation object
    repr_type = retrieve_type_of_molecular_representation(representation_name)
    list_if_files = os.listdir(repr_dir)
    if not repr_dir.endswith("/"):  # Normalise away trailing slashes
        repr_dir = repr_dir + "/"

    repr_ = repr_type.load(repr_dir + list_if_files[0])
    Xtrain, ytrain, Xtest, ytest = repr_.split()

    # create final folder
    os.makedirs(final_folder_path, exist_ok=True)

    print("In store_metadata for representation name: " + representation_name)
    print(
        "Final folder: "
        + final_folder_path
        + " mentre property_model_folder_path: "
        + property_model_folder_path
    )

    # normalize paths
    path_to_df = normalize_path(path_to_df)
    final_folder_path = normalize_path(final_folder_path)
    property_model_folder_path = normalize_path(property_model_folder_path)

    # Set property_model_folder_path correctly
    arr = os.listdir(property_model_folder_path)
    property_model_folder_path = property_model_folder_path + "/" + arr[0]

    # check if path_to_model
    path_to_model_dict = {}
    if path_to_model:
        path_to_model = normalize_path(path_to_model)
        path_to_model_dict = {representation_name: path_to_model}

    arg_dict = dict(
        datapath=path_to_df,
        friendly_model_name=experiment_name,
        id_column_name=id_column_name,
        smiles_column_name=smiles_column_name,
        properties_column_name_list=properties_column_name_list,
        problem_type=problem_type,  # Can be regression or classification
        best_model_output_dir=final_folder_path,
        representations=[representation_name],
        path_to_model=path_to_model_dict,  # this should be set
        project_code="",
    )
    final_meta_data = arg_dict
    bm_rep = representation_name
    final_meta_data["best_model_representation"] = bm_rep
    final_meta_data["desirability_scores"] = desirability_scores
    logger.info(f"desirability_scores is: {desirability_scores}")

    timestr = time.strftime("%Y%m%d_%H%M%S")

    # Extra useful info to store
    (
        final_meta_data["rep_model_id"],
        model,
        [
            final_meta_data["features_scaling_type"],
            final_meta_data["features_means_vect"],
            final_meta_data["features_stds_vect"],
        ],
    ) = get_model_id_from_bench(
        path_to_bench_obj=property_model_folder_path,
        problem_type=problem_type,
        representation_name=representation_name,
    )  # function to retrieve the generator model id from teh benchmarking object (stored in the input folder)

    # Refit model
    # concatenate train and test sets
    X_fin = np.concatenate((Xtrain, Xtest), 0)
    y_fin = np.concatenate((ytrain, ytest), 0)
    print("Refitting AutoSklearn model...")
    model.refit(X_fin, y_fin)

    wrapper = PreferModelWrapper(model=model, metadata=final_meta_data)

    # Take df_complete from property_model_folder_path
    print("copy_file df_complete")
    copy_file(
        src=f"{property_model_folder_path}/df_complete.csv",
        dest=f"{final_folder_path}/df_complete.csv",
    )
    copy_file(
        src=f"{property_model_folder_path}/df_complete.json",
        dest=f"{final_folder_path}/df_complete.json",
    )
    # Save wrapper in final location
    metadata_name = f"{final_folder_path}/{experiment_name}_{representation_name}_{timestr}"
    with open(metadata_name + ".pkl", "wb") as output:
        pickle.dump(wrapper, output)

    return


def get_model_id_from_bench(path_to_bench_obj, problem_type, representation_name):

    """
    Function to retrieve the model representation id and the model from the benchmarking object
    Inputs:
    - path_to_bench_obj: string related to the folder path where teh benchmarking object is stored
    - problem_type: string related to the type of the problem; classification or regression are possible choises
    - representation_name: string related to the molecular representation selected. This can be a classical molecular representation (like FINGERPRINTS or DESCRIPTORS2D)
    or datadriven molecular representation
    """
    tmp = Benchmarking(problem_type=problem_type)
    try:
        tmp.load(path_to_bench_obj, path_to_bench_obj)
    except Exception as e:
        logger.error(e)
        raise

    try:
        return (
            tmp.models_ids[representation_name],
            tmp.best_estimator[representation_name],
            [
                tmp.features_scaling_type[representation_name],
                tmp.features_means_vect[representation_name],
                tmp.features_stds_vect[representation_name],
            ],
        )
    except Exception as e:
        logger.error(e)
        raise


def normalize_path(path):
    if path.endswith("/"):  # Normalise away trailing slashes
        path = path[:-1]
    return path


def _get_desirability_scores(desirability_scores: str) -> dict:
    # unpack desirability_scores
    try:
        # TODO:  is this sensible?  input is an array, and we take the first item?
        desirability_scores = json.loads(desirability_scores[0])

    except Exception:
        desirability_scores_json_format = json.dumps(desirability_scores)
        desirability_scores = json.loads(desirability_scores_json_format)

    is_str = isinstance(desirability_scores, str)
    if is_str:
        desirability_scores = ast.literal_eval(desirability_scores)

    return desirability_scores


def run():
    parser = argparse.ArgumentParser(description="model wrapper")

    parser.add_argument(
        "-ptd",
        "--path_to_df",
        type=str,
        help="The entire path to the dataframe used for this experiment. The dataframe should be stored as .csv, "
        "should use semicolomn as separator and should contain information about the SMILE representation "
        "of each molecule, an ID of the molecules and the property/ies one want to model.",
        required=True,
    )

    parser.add_argument(
        "-ptm",
        "--path_to_model",
        type=str,
        help="path to the model that should be used to convert smiles into embeddings",
    )

    parser.add_argument(
        "-en",
        "--experiment_name",
        type=str,
        help="name of the experiment one would like to perform. E.g. logD",
        required=True,
    )

    parser.add_argument(
        "-icn",
        "--id_column_name",
        type=str,
        help="name of the dataframe column where the id of each molecule is stored",
        required=True,
    )

    parser.add_argument(
        "-scn",
        "--smiles_column_name",
        type=str,
        help="name of the dataframe column where the smile representation of each molecule is stored",
        required=True,
    )

    parser.add_argument(
        "-pcn",
        "--properties_column_name",
        action="append",
        help="list of names of the dataframe columns where the property/ies of each molecule is stored",
        required=True,
    )

    parser.add_argument(
        "-pt",
        "--problem_type",
        type=str,
        help="whether this is a <regression> or a <classification> problem",
        required=True,
    )

    parser.add_argument(
        "-rn",
        "--representation_name",
        type=str,
        help="name of the rapresentation to compute or path to the generator which is used to map smiles into embeddings",
        required=True,
    )  # here you can have a list representations so that is case this list has a lenght >1 then the first step is to combine the representations

    parser.add_argument(
        "-pmfp",
        "--property_model_folder_path",
        type=str,
        help="directory where the property model together with the benchmarking object are stored."
        "current directory.",
        required=True,
    )

    parser.add_argument(
        "-ffp",
        "--final_folder_path",
        type=str,
        help="directory where the results will be stored. If not specified results will be store in the "
        "current directory.",
        required=True,
    )

    parser.add_argument(
        "-rdir",
        "--repr_dir",
        type=str,
        help="directory where the molecular representation is stored. ",
        required=True,
    )

    parser.add_argument(
        "-ds",
        "--desirability_scores",
        type=str,
        help="model desirability_scores as a json that looks like a dict",
        required=True,
    )

    args = parser.parse_args()

    # unpack properties_column_name
    try:
        properties_column_name = json.loads(args.properties_column_name[0])

    except Exception:
        properties_column_name_json_format = json.dumps(args.properties_column_name)
        properties_column_name = json.loads(properties_column_name_json_format)

    desirability_scores = _get_desirability_scores(args.desirability_scores)

    store_metadata(
        path_to_df=args.path_to_df,
        path_to_model=args.path_to_model,
        problem_type=args.problem_type,
        experiment_name=args.experiment_name,
        id_column_name=args.id_column_name,
        smiles_column_name=args.smiles_column_name,
        properties_column_name_list=properties_column_name,
        representation_name=args.representation_name,
        final_folder_path=args.final_folder_path,
        property_model_folder_path=args.property_model_folder_path,
        repr_dir=args.repr_dir,
        desirability_scores=desirability_scores,
    )


if __name__ == "__main__":
    run()
