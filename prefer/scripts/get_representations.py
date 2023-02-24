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
import argparse
import json
import logging
import os
import sys

import pandas as pd
import shutil


from prefer.utils.data_preparation import prepare_data
from prefer.utils.mapping import mapping_representations

logger = logging.getLogger(__name__)


def compute_representations_from_args(
    path_to_df,
    representation_to_compute,
    path_to_model,
    output_dir,
    experiment_name,
    id_column_name,
    smiles_column_name,
    splitting_strategy,
    temporal_info_column_name,
    properties_column_name_list,
):
    """
    we assume that the dataset has the semicolomn as separator and that the dataset
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Read your .csv files
    if path_to_df.endswith("/"):  # Normalise away trailing slashes
        path_to_df = path_to_df[:-1]

    try:
        arr = os.listdir(path_to_df)
        path_to_df = path_to_df + "/" + arr[0]
    except Exception:
        logger.info("Already a file")

    try:
        df = pd.read_csv(path_to_df)
    except Exception:
        df = pd.read_csv(path_to_df, sep=";")

    # in prepare_data now the dataset is both prepared and filtered

    # Manipulate dataframe such that it is in the right shape fo being used as input of the DataStorage class
    # ¦ ID ¦ Smiles ¦ Property_1 ¦ Property_2 ¦ ... ¦ Property_N ¦
    # -------------------------------------------------------------
    # This is done by specifying the experiment_name, the name of column where the ID information and SMILES representation of each sample is stored, and finally
    # the list of the columns' names of the properties to model.
    df = prepare_data(
        df=df,
        id_column_name=id_column_name,
        smiles_column_name=smiles_column_name,
        properties_column_name_list=properties_column_name_list,
        temporal_info_column_name=temporal_info_column_name,
    )

    mapping_representations(
        representation_name=representation_to_compute,
        df=df,
        output_dir=output_dir,
        path_to_model=path_to_model,
        experiment_name=experiment_name,
        path_to_df=path_to_df,
        split_type=splitting_strategy,
    )
    logger.info("Representation Computed")
    
    return output_dir


if __name__ == "__main__":
    """
    Example of usage:
    %run get_representations.py -ptd "/path/to/dataframe/dataframe.csv" -rtc "FINGERPRINTS"
    -od "/path/to/representation/PREFER_automation_branch/" -en "logD" -icn "Molecule ChEMBL ID"
    -scn "Smiles" -pcn "Standard Value"
    """
    parser = argparse.ArgumentParser(description="compute molecule representation")
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
        "-rtc",
        "--representation_to_compute",
        type=str,
        help="name of the rapresentation to compute or path to the generator which is used "
        "to map smiles into embeddings. If a model-based representation is selected then a "
        "path to model should be indicated",
        required=True,
    )

    parser.add_argument(
        "-ptm",
        "--path_to_model",
        type=str,
        help="path to the model that should be used to convert smiles into embeddings",
    )

    parser.add_argument(
        "-od",
        "--output_dir",
        type=str,
        help="path to the directory where to store the molecule representation computed",
        required=True,
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
        "-ss",
        "--splitting_strategy",
        type=str,
        help="name of splitting startegy selected [random, temporal, cluster]",
        required=True,
    )

    parser.add_argument(
        "-ticn",
        "--temporal_info_column_name",
        type=str,
        help="name of the column where the temporal information is stored",
    )

    parser.add_argument(
        "-pcn",
        "--properties_column_name",
        action="append",
        help="list of names of the dataframe columns where the property/ies of each molecule is stored",
        required=True,
    )
    # if multiple tasks -pcn "Task1" -pcn "Task2" -pcn "Task3"

    args = parser.parse_args()
    if (
        args.representation_to_compute not in ["FINGERPRINTS", "DESCRIPTORS2D", "TF2_GNN"]
        and not args.path_to_model
    ):
        raise RuntimeError(
            f"Please specify a path_to_model for molecular representations which are not in the default ones "
            f"[FINGERPRINTS, DESCRIPTORS2D, TF2_GNN]"
        )

    try:
        properties_column_name = json.loads(args.properties_column_name[0])

    except Exception:
        properties_column_name_json_format = json.dumps(args.properties_column_name)
        properties_column_name = json.loads(properties_column_name_json_format)

    compute_representations_from_args(
        path_to_df=args.path_to_df,
        representation_to_compute=args.representation_to_compute,
        path_to_model=args.path_to_model,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        id_column_name=args.id_column_name,
        smiles_column_name=args.smiles_column_name,
        splitting_strategy=args.splitting_strategy,
        temporal_info_column_name=args.temporal_info_column_name,
        properties_column_name_list=properties_column_name,
    )
