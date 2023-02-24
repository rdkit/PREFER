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
import numpy as np
import pandas as pd


from prefer.utils.check_input_dataframe import (
    check_dataframe,
    check_fields,
    check_fields_types,
    check_final_structure,
)


class TestCheckDataStorage(unittest.TestCase):
    def setUp(self):
        """Executed before every test case"""
        mol_representation_df = pd.DataFrame(
            np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD")
        )
        mol_representation_df.iloc[0, 0] = np.nan
        mol_representation_df = mol_representation_df.dropna()
        self.globalvar = mol_representation_df

    def tearDown(self):
        """Executed after every test case"""
        print("\ntearDown executing after the test case. Result:")

    def test_check_dataframe(self):
        self.assertFalse(check_dataframe(self.globalvar))

    def test_check_fields(self):
        df = pd.DataFrame(
            np.random.randint(0, 100, size=(100, 4)),
            columns=list(["Smiles", "ID", "Property_2", "Property_3"]),
        )
        self.assertFalse(check_fields(df))

    def test_check_fields_types(self):
        df = pd.DataFrame(
            np.random.randint(0, 100, size=(100, 4)),
            columns=list(["Smiles", "ID", "Property_2", "Property_3"]),
        )
        experiment_name = "experim_1"
        index_of_separation = 55
        split_type = "wrong_split_type"
        mask = False
        mask_value = -1
        problem_type = "regression"
        self.assertFalse(
            check_fields_types(
                df, experiment_name, problem_type, mask, mask_value, split_type, index_of_separation
            )
        )

    def test_check_fields_types_2(self):
        df = pd.DataFrame(
            np.random.randint(0, 100, size=(100, 4)),
            columns=list(["Smiles", "ID", "Property_2", "Property_3"]),
        )
        experiment_name = "experim_1"
        index_of_separation = 55
        split_type = "temporal"
        mask = False
        mask_value = -1
        problem_type = "regression"
        self.assertTrue(
            check_fields_types(
                df, experiment_name, problem_type, mask, mask_value, split_type, index_of_separation
            )
        )

    def test_check_final_structure(self):
        df = pd.DataFrame(
            np.random.randint(0, 100, size=(100, 4)),
            columns=list(["Smiles", "ID", "Property_1", "Property_2"]),
        )
        df["Property_1"][0] = np.nan
        self.assertFalse(check_final_structure(df))

    def test_check_final_structure_1(self):
        df = pd.DataFrame(
            np.random.randint(0, 100, size=(100, 4)),
            columns=list(["Smiles", "ID", "Property_1", "Property_2"]),
        )
        self.assertTrue(check_final_structure(df))


if __name__ == "__main__":
    unittest.main()
