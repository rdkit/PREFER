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
import sys
from typing import Optional
from dataclasses import dataclass


import pandas as pd


from prefer.utils.data_utils import convert
from prefer.utils.splitting_strategies import (
    random_split,
    temporal_split,
    cluster_split,
)
from prefer.utils.features_scaling import scale_features
from prefer.src.molecule_representations import MoleculeRepresentations


@dataclass
class VectorMoleculeRepresentations(MoleculeRepresentations):
    repr_type: str = "vector"
    scale_type: Optional[str] = None
    seed: int = 1
    features_means: Optional[pd.Series] = None
    features_stds: Optional[pd.Series] = None

    def split(self, return_indices: bool = False):
        """
        method to extract the indices used to split the original dataset and obtain the final dataframes
        """
        print("Splitting the dataset according to: " + self.split_type + " split")

        indices = self.extract_indices()

        # In case of One shot one could add another split_type that should split alond the tasks and not on the samples
        if not indices:
            raise ValueError("Empty indices for splitting dataset")
        else:
            if len(indices) == 2:
                logging.debug("No Validation Set")
                index_train = indices[0]
                index_test = indices[1]
                Xtrain, ytrain, Xtest, ytest = self.extract_matrices(index_train, index_test)
                if self.scale_type:
                    print("Scaling features according to: " + self.scale_type)
                    Xtrain, Xtest, self.features_means, self.features_stds = scale_features(
                        Xtrain, Xtest, scaling_type=self.scale_type
                    )
            else:
                raise ValueError("Validation set cannot be computed for the moment")

            if return_indices:
                return Xtrain, ytrain, Xtest, ytest, index_train, index_test
            else:
                return Xtrain, ytrain, Xtest, ytest

    def extract_matrices(self, index_train, index_test):
        """
        method used to convert the test/train datasets, obtained by splitting the original dataset, into numpy arrays and store them into Xtrain and Xtest.
        """
        if max(index_train) > (self.df.shape[0] - 1) or max(index_test) > (self.df.shape[0] - 1):
            raise ValueError("ERROR with indices")

        properties = self.df.columns[["Property" in str(x) for x in self.df.columns.values]].values
        if properties.size == 0:
            properties = self.df.columns[["true_label_" in str(x) for x in self.df.columns.values]].values 
        elif properties.size == 0:
            raise ValueError('Columns with either Property or true_label_ cannot be found in the dataset. Cannot understand where labels are stored.')
        df_train = self.df.iloc[index_train]
        df_train = df_train.reset_index()
        df_test = self.df.iloc[index_test]
        df_test = df_test.reset_index()
        if "molecule_representation" in df_train.columns.values:
            repr_name = "molecule_representation"
        else:
            repr_name = self.representation_name

        Xtrain, ytrain = convert(df_train, repr_name, properties)
        Xtest, ytest = convert(df_test, repr_name, properties)

        return Xtrain, ytrain, Xtest, ytest

    def extract_indices(self):
        """
        method to extract the indices used to split the original dataset. They are computed according to the strategy required by the user.
        """
        if self.split_type == "random":
            return random_split(self.df, self.seed, limit_def=self.limit_def)
        elif self.split_type == "cluster":
            return cluster_split(df=self.df)
        elif self.split_type == "temporal":
            return temporal_split(df=self.df)
        else:
            raise ValueError(
                f"Split method {self.split_type} is not valid. Allowed options are random, cluster, temporal"
            )
