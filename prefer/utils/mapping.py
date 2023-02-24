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


from pandas import DataFrame


from prefer.molecule_representations.descriptors2D_representations_builder import (
    Descriptors2DRepresentationsBuilder,
)
from prefer.molecule_representations.fingerprints_representations_builder import (
    FingerprintsRepresentationsBuilder,
)
from prefer.molecule_representations.model_representations_builder import (
    ModelRepresentationsBuilder,
)


def mapping_representations(
    representation_name: str,
    df: DataFrame,
    output_dir: str,
    path_to_model: str = "",
    path_to_df: str = "",
    experiment_name: str = "",
    split_type: str = "random",
):  # obj should be the object of the class for generic model
    """
    Function to map representation names to the corresponding molecule representation builder. The function generate the representation and it save it in a
    define directory.
    The function returns the directory name (string) and the representation type (MoleculeRepresentations object)
    """

    if representation_name == "DESCRIPTORS2D":
        builder = Descriptors2DRepresentationsBuilder()
    elif representation_name == "FINGERPRINTS":
        builder = FingerprintsRepresentationsBuilder()
    else:
        builder = ModelRepresentationsBuilder(
            path_to_model=path_to_model, representation_name=representation_name
        )

    representations = builder.build_representations(molecule_data_orig=df, split_type=split_type)
    representations.save(output_dir, representation_name, experiment_name, path_to_df)


def representations_supported():
    """
    Function to return the names of the representations currently supported by PREFER
    """

    return ["CDDD", "DESCRIPTORS2D", "MOLER", "FINGERPRINTS"]
