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

import sys


from prefer.utils.filtering import filter_and_normalize_mols
import pandas as pd


def prepare_data(
    df,
    id_column_name: str,
    smiles_column_name: str,
    properties_column_name_list: list,
    temporal_info_column_name: str = None,
    filter_flag: bool = True,
):

    """
    Function to prepare datasets.
    The inputs are:
    df: dataframe to be manipulated
    id_column_name: string of the name of the column where the ID is stored
    smiles_column_name: string of the name of the column where the smile representation is stored
    properties_column_name_list: list of the strings/names of the property/ies to evaluate

    """
    if not isinstance(properties_column_name_list, list):
        raise ValueError('properties_column_name_list should be a list of names of the selected labels')
    # Evaluate whether unique labels
    if len(properties_column_name_list) > len(set(properties_column_name_list)):
        raise ValueError('Duplicates in the labels list cannot be handled by PREFER - please provide unique labels names')
        
    # Check if consistent
    check = list()
    check.append(all([x in df.columns.values for x in properties_column_name_list]))
    check.append(id_column_name in df.columns.values)
    check.append(smiles_column_name in df.columns.values)
    if temporal_info_column_name:
        check.append(temporal_info_column_name in df.columns.values)
    if not all(check):
        raise ValueError("ERROR: columns name not found in the dataframe")

    cols = list()
    df.rename(columns={id_column_name: "ID"}, inplace=True)
    cols.append("ID")
    df.rename(columns={smiles_column_name: "Smiles"}, inplace=True)
    cols.append("Smiles")

    if temporal_info_column_name:
        df[temporal_info_column_name] = pd.to_datetime(df[temporal_info_column_name])
        df.rename(columns={temporal_info_column_name: "Time"}, inplace=True)
        cols.append("Time")

    # TO DO extend AutoSklearn in the case of sparsity of the label matrx. For now we need to remove nans
    print(
        "WARNING: Autosklearn does not handle for now label matrix sparsity, thus nan values will be removed both for single task and multitasking cases"
    )
    for index, _ in enumerate(properties_column_name_list):
        df = df[df[properties_column_name_list[index]].notna()]
        df = df.reset_index(drop=True)

    for index, properties_column_name in enumerate(properties_column_name_list):
        df.rename(columns={properties_column_name: "Property_" + str(index + 1)}, inplace=True)
        cols.append("Property_" + str(index + 1))
    if filter_flag:
        return filter_and_normalize_mols(df[cols])
    else:
        return df[cols]
