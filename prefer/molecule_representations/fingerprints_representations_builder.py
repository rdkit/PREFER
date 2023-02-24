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


from prefer.utils.data_utils import check_if_nan, generate_fingerprints, generate_molecule
from prefer.src.molecule_representations_builder import MoleculeRepresentationsBuilder
from prefer.src.molecule_representations import MoleculeRepresentations
from prefer.src.vector_molecule_representations import VectorMoleculeRepresentations


class FingerprintsRepresentationsBuilder(MoleculeRepresentationsBuilder):
    def __init__(
        self, limit_def: int = None,
    ):
        self.limit_def = limit_def
        
    def build_representations(
        self, molecule_data_orig: DataFrame, split_type: str = "random", seed=1,
    ) -> MoleculeRepresentations:
        """
        method to compute Morgan Fingerprints as implemented in RDKit

        Input:
        - molecule_data: this is a dataframe of the shape
        | ID | Smiles | Property_1 | Property_2 | ... | Property_N |
        ------------------------------------------------------------
        - split_type: string related to the type of test/train split one want to apply. Possible split_type are random, temporal and cluster. One can add new splitting strategies in utils.splitting_strategies
        Output:
        - MoleculeRepresentations object
        """
            
            
        molecule_data = molecule_data_orig.copy()
        logging.info("Generate Morgan Fingerprints")
        molecules = generate_molecule(molecule_data)
        molecule_data["molecule_representation"] = generate_fingerprints(molecules)
        molecule_data = self.remove_nan(molecule_data)

        return VectorMoleculeRepresentations(
            df=molecule_data, representation_name="FINGERPRINTS", split_type=split_type, seed=seed, limit_def = self.limit_def,
        )

    def remove_nan(self, molecule_data: DataFrame):
        """
        method use to check whetehr a representation has nan values and in case remove the corresponding row.

        input: representation_to_add is the representation to check
        """
        nan_rows = check_if_nan(molecule_data["molecule_representation"])
        if nan_rows:
            logging.warning(
                "Found nan in the representation:"
                + "fingerprints"
                + ". The following sample/s should be removed from the dataframe:"
                + str(nan_rows)
            )
            molecule_data = molecule_data.drop(molecule_data.index[nan_rows])
            # Reset indices
            molecule_data = molecule_data.reset_index(drop=True)
        return molecule_data
