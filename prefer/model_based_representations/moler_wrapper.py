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


import os
import pathlib
from typing import Any, List, Tuple, Union

import numpy as np

from prefer.model_based_representations.interface import LatentSpaceMoleculeGenerator
from molecule_generation.wrapper import VaeWrapper

Pathlike = Union[str, pathlib.Path]


class MoLeRGeneratorModel(VaeWrapper, LatentSpaceMoleculeGenerator):
    def __init__(
        self, dir: Pathlike, seed: int = 0, num_workers: int = 6, beam_size: int = 1, **kwargs: Any
    ):
        VaeWrapper.__init__(self, dir, seed=seed, num_workers=num_workers, beam_size=beam_size)
        LatentSpaceMoleculeGenerator.__init__(
            self, dir, model_id_file_patterns=("*_best.pkl", "*_best.hdf5"), **kwargs
        )

        self._can_decode_from_scaffold = True

    def get_name(self) -> str:
        return "MoLeR"

    @classmethod
    def is_valid_dir(cls, model_dir: str) -> object:
        files_in_dir = os.listdir(model_dir)
        return any(
            "_MoLeR__" in filename or "_MotifMoLeR__" in filename for filename in files_in_dir
        )
        return any(cls._is_moler_model_filename(filename) for filename in files_in_dir)

    def set_extra_args(self, **kwargs):
        workers = kwargs.get("num_workers")
        if workers is not None:
            self.num_workers = kwargs.get("num_workers")
        beam = kwargs.get("beam_size")
        if beam is not None:
            self.beam_size = kwargs.get("beam_size")
