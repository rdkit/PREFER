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


import logging
import pickle
from abc import abstractmethod, ABC
import time
from dataclasses import dataclass
from typing import Optional
import os

from pandas import DataFrame


@dataclass
class MoleculeRepresentations(ABC):
    df: DataFrame
    representation_name: str
    split_type: str
    model_path: str = ""
    repr_type: str = ""
    model_id: str = "tmp_id"
    experiment_name: str = "new_experiment"
    path_to_df: str = ""
    limit_def: int = None

    @abstractmethod
    def split(self):
        pass

    def save(
        self,
        path: str,
        name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        path_to_df: Optional[str] = None,
    ):
        """

        method to save the MoleculeRepresentations object in the location specified by path

        Usage:
        mol_repr.save('../folder/')
        """

        if experiment_name is not None:
            self.experiment_name = experiment_name

        if path_to_df is not None:
            self.path_to_df = path_to_df

        timestr = time.strftime("%Y%m%d-%H%M%S")

        final_path = os.path.join(
            path,
            f"{self.experiment_name}_{name or self.representation_name}_{self.repr_type}_{timestr}.pkl",
        )

        with open(final_path, "wb",) as output:
            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)

        logging.info(f"Representation saved in {final_path}")

    @classmethod
    def load(cls, path: str):
        """
        Load MoleculeRepresentations from a .pkl file.
        """

        with open(path, "rb") as input:
            tmp = pickle.load(input)
        return cls(**tmp)
