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
import unittest
import os
import ast

import numpy as np
import json
import yaml
import pandas as pd


from prefer.scripts.get_representations import compute_representations_from_args
from prefer.scripts.run_PREFER import run_PREFER_from_args
from prefer.scripts.model_wrapper import store_metadata


class TestScripts(unittest.TestCase):
    def test_get_representations(self):
        path_to_df = "./data_for_test/logDPublic.csv"
        representation_to_compute = "FINGERPRINTS"
        path_to_model = None
        output_dir = "./representations_dir/"
        experiment_name = "test_logDPublic"
        id_column_name = "Molecule ChEMBL ID"
        smiles_column_name = "Smiles"
        splitting_strategy = "random"
        temporal_info_column_name = None
        properties_column_name_list = ["Standard Value"]
        try:
            os.makedirs(output_dir, exist_ok=True)
            print("Directory '%s' created successfully" % output_dir)
        except OSError as error:
            print("Directory '%s' can not be created")

        output_dir_new = compute_representations_from_args(
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
        )
        self.assertTrue(output_dir == output_dir_new)

    def test_run_PREFER(self):
        problem_type = "regression"
        representation_name = "FINGERPRINTS"
        repr_dir = "./representations_dir/"
        final_folder_path = "./output_dir/"
        experiment_name = "test_logDPublic"
        try:
            os.makedirs(final_folder_path, exist_ok=True)
            print("Directory '%s' created successfully" % final_folder_path)
        except OSError as error:
            print("Directory '%s' can not be created")

        run_PREFER_from_args(
            problem_type, representation_name, repr_dir, final_folder_path, experiment_name,
        )

    def test_store_metadata(self):
        path_to_df = "./data_for_test/logDPublic.csv"
        path_to_model = None
        problem_type = "regression"
        experiment_name = "test_logDPublic_wrapper"
        id_column_name = "Molecule ChEMBL ID"
        smiles_column_name = "Smiles"
        properties_column_name_list = ["Standard Value"]
        representation_name = "FINGERPRINTS"
        final_folder_path = "./wrappers_dir/"
        try:
            os.makedirs(final_folder_path, exist_ok=True)
            print("Directory '%s' created successfully" % final_folder_path)
        except OSError as error:
            print("Directory '%s' can not be created")

        property_model_folder_path = "./output_dir/"
        repr_dir = "./representations_dir/"
        with open("./file_for_test/logD_desirability_scores.yaml") as file:
            try:
                parsed_yaml_file = yaml.safe_load(file)
            except yaml.YAMLError as exception:
                print(exception)

            desirability_scores = json.dumps(parsed_yaml_file["desirability_scores"])
            is_str = isinstance(desirability_scores, str)
            if is_str:
                desirability_scores = ast.literal_eval(desirability_scores)
            store_metadata(
                path_to_df=path_to_df,
                path_to_model=path_to_model,
                problem_type=problem_type,
                experiment_name=experiment_name,
                id_column_name=id_column_name,
                smiles_column_name=smiles_column_name,
                properties_column_name_list=properties_column_name_list,
                representation_name=representation_name,
                final_folder_path=final_folder_path,
                property_model_folder_path=property_model_folder_path,
                repr_dir=repr_dir,
                desirability_scores=desirability_scores,
            )


if __name__ == "__main__":
    unittest.main()
