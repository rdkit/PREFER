#  Copyright (c) 2023, Novartis Institutes for BioMedical Research Inc.
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
#     * Neither the name of Novartis Institutes for BioMedical Research Inc. 
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

import warnings
import pandas as pd
warnings.filterwarnings('ignore')
from prefer.utils.data_preparation import prepare_data
import logging
import argparse
import yaml

root = logging.getLogger()
root.setLevel(logging.DEBUG)

from prefer.molecule_representations.model_representations_builder import ModelRepresentationsBuilder
import json


def compute_representations_from_args(
    args,
    path_to_model,
    model_name = None,
):
    """
    NB: we assume that the dataset has the semicolomn as separator and that the dataset
    """
    path_to_df=args.path_to_df
    id_column_name=args.id_column_name
    smiles_column_name=args.smiles_column_name
    split_type=args.splitting_strategy
    temporal_info_column_name=args.temporal_info_column_name
    
    supported_models = ['CDDD', 'MOLER']
    if model_name == None:
        model_name = 'MODELBASED'
        print(f'WARNING: PREFER supports only {supported_models}, but other models can be used')
    elif(model_name not in supported_models):
        print(f'WARNING: PREFER supports only {supported_models}, but other models can be used')
    
    try:
        properties_column_name = json.loads(args.properties_column_name[0])

    except Exception:
        properties_column_name_json_format = json.dumps(args.properties_column_name)
        properties_column_name = json.loads(properties_column_name_json_format)
        
    properties_column_name_list=properties_column_name
    
    # Read your .csv files
    if path_to_df.endswith("/"):  # Normalise away trailing slashes
        path_to_df = path_to_df[:-1]

    try:
        arr = os.listdir(path_to_df)
        path_to_df = path_to_df + "/" + arr[0]
    except Exception:
        logging.info("Already a file")

    try:
        df = pd.read_csv(path_to_df)
    except Exception:
        df = pd.read_csv(path_to_df, sep=";")

    # in prepare_data now the dataset is both prepared and filtered
    try:
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

    except Exception:
        logging.error(
            "ERROR in preparing data. One of id_column_name, smiles_column_name, properties_column_name_list may be wrong."
        )
        sys.exit(1)

    #For model based representations
    model_based_representations = ModelRepresentationsBuilder(path_to_model = path_to_model, limit_def = args.limit_def)
    model_based = model_based_representations.build_representations(df, split_type = split_type)
    
    # save representations
    import os

    # define the name of the directory to be created
    experiment_name = args.experiment_name
    path = f"./{model_name}_representations_{experiment_name}"

    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
        
    model_based.representation_name = model_name
    model_based.save(path)
    
    print(f'{model_name} representation correctly saved in {path}')
    return 



if __name__ == "__main__":
    '''
    Script to compute the model_based representations of a set of molecules in a dataframe. 
    '''
    parser = argparse.ArgumentParser(
        description=f"Compute model_based-representations",
    )
    parser.add_argument(
        "--prefer_args",
        type=str,
        help="path to the .yaml file where configuration parameters are stored.",
    )

    parser.add_argument(
        "--path_to_model",
        type=str,
        help="path to model_based model that has been previously downloaded",
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        help="string of model_name, e.g. CDDD or MOLER",
    )

    args = parser.parse_args()
    a_yaml_file = open(args.prefer_args)
    parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)

    args.path_to_df = parsed_yaml_file["path_to_df"]
    args.experiment_name = parsed_yaml_file["experiment_name"]
    args.id_column_name = parsed_yaml_file["id_column_name"]
    args.smiles_column_name = parsed_yaml_file["smiles_column_name"]
    args.properties_column_name = parsed_yaml_file["properties_column_name_list"]
    args.problem_type = parsed_yaml_file["problem_type"]
    args.splitting_strategy = parsed_yaml_file["splitting_strategy"]
    
    if 'limit_def' in parsed_yaml_file:
        args.limit_def = parsed_yaml_file["limit_def"]
    else:
        args.limit_def = None


    if "temporal_info_column_name" in parsed_yaml_file:
        args.temporal_info_column_name = parsed_yaml_file["temporal_info_column_name"]
    else:
        args.temporal_info_column_name = None
        
    compute_representations_from_args(args,args.path_to_model, args.model_name)
