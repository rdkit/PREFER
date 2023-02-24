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


"""
Helper functions to check the validity of the input dataframe.
"""
import logging

import pandas as pd


def check_dataframe(df):
    """
    This function helps to evaluate whether a dataframe has a correct indices of the rows
    """
    if df.index[-1] != (df.shape[0] - 1):
        print("Indices of the dataframe are not correct")
    return df.index[-1] == (df.shape[0] - 1)


def check_fields(df):
    cols = df.columns.values
    property_to_eval = df.columns[["Property" in str(x) for x in df.columns.values]].values
    expected_cols = ["Smiles", "ID"] + [
        "Property_" + str(x + 1) for x, elem in enumerate(property_to_eval)
    ]
    return all([x in cols for x in expected_cols])


def check_fields_types(
    df, experiment_name, problem_type, mask, mask_value, split_type, index_of_separation
):
    """
    This function helps to evaluate whether the dataframe fields are of the correct type.
    """
    return all(
        [
            isinstance(df, pd.DataFrame),
            isinstance(experiment_name, str),
            isinstance(problem_type, str),
            isinstance(split_type, str),
            split_type in ["random", "cluster", "temporal"],
            problem_type in ["regression", "classification"],
            isinstance(index_of_separation, int),
            isinstance(mask, bool),
            isinstance(mask_value, (int, float, complex)),
        ]
    )


def check_final_structure(df):
    """
    Function to check if the dataframes are proper for the building of the models
    """
    property_to_eval = df.columns[["Property" in str(x) for x in df.columns.values]].values
    for prop in property_to_eval:
        if df[prop].isnull().sum() > 0:
            logging.error(
                "ERROR --> some labels are NaN. Please check your dataframes before running eval_.BenchMoleProp()"
            )
            return False
    return True
