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

from pandas import DataFrame


from prefer.utils.data_utils import check_if_nan
from prefer.src.molecule_representations_builder import MoleculeRepresentationsBuilder
from prefer.src.molecule_representations import MoleculeRepresentations
from prefer.src.vector_molecule_representations import VectorMoleculeRepresentations
from prefer.model_based_representations.model_based_representations_factory import (
    load_model_from_directory,
)


class ModelRepresentationsBuilder(MoleculeRepresentationsBuilder):
    def __init__(
        self, path_to_model: str = None, representation_name: str = None, limit_def: int = None,
    ):
        self.path_to_model = path_to_model
        # here instance models
        self.model_instance = load_model_from_directory(path_to_model)
        self.representation_name = representation_name
        self.limit_def = limit_def

    def build_representations(
        self,
        molecule_data_orig: DataFrame,
        embedding_types: str = "vector",
        split_type: str = "random",
        padding_size: int = 100,
        seed=1,
    ) -> MoleculeRepresentations:
        """
        generic generator model to convert smile to embeddings
        Input:
        - molecule_data: this is a dataframe of the shape
        | ID | Smiles | Property_1 | Property_2 | ... | Property_N |
        ------------------------------------------------------------
        - split_type: string related to the type of test/train split one want to apply.
            Possible split_type are random, temporal and cluster. One can add new splitting strategies
            in utils.splitting_strategies
        - padding_size: max dimension of the final list of vectors (max number of atoms per molecule)

        Output:
        - MoleculeRepresentations object
        """
    
            
            
        if self.representation_name is None:
            self.representation_name = "model_based_representation"

        if embedding_types not in ["vector"]:
            raise ValueError("ERROR: embedding_types not known, only vector is possible.")

        molecule_data = molecule_data_orig.copy()
        logging.info("Generate Model based Representation")

        try:

            if embedding_types == "vector":
                with self.model_instance as model:
                    smiles_embedding = model.encode(molecule_data.Smiles.to_list())
                    version_model_ID = model.get_model_id()
                list_of_smiles_embedding = [x for x in smiles_embedding]
            else:
                raise ValueError(f"{embedding_types} not known. Only vector is possible.")
        except Exception as e:
            raise ValueError(
                f"ERROR: the model directory for the model based representation might be incorrect or another error occurred: ValueError exception thrown{e}"
            )

        if embedding_types == "vector":
            molecule_data["molecule_representation"] = list_of_smiles_embedding
            molecule_data = self.remove_nan(molecule_data)
            return VectorMoleculeRepresentations(
                df=molecule_data,
                representation_name=self.representation_name,
                split_type=split_type,
                seed=seed,
                model_id=version_model_ID,
                limit_def = self.limit_def,
            )
        else:
            raise ValueError(
                f"embedding_types: {embedding_types} not known. Only vector is supported"
            )

    def remove_nan(self, molecule_data: DataFrame):
        """
        method use to check whetehr a representation has nan values and in case remove the corresponding row.

        input: representation_to_add is the representation to check
        """
        nan_rows = check_if_nan(molecule_data["molecule_representation"])
        if nan_rows:
            logging.warning(
                "Found nan in the representation"
                + self.representation_name
                + ". The following sample/s should be removed from the dataframe:"
                + str(nan_rows)
            )
            molecule_data = molecule_data.drop(molecule_data.index[nan_rows])
            # Reset indices
            molecule_data = molecule_data.reset_index(drop=True)
        return molecule_data
