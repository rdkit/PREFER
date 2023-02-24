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


import json
import os
from typing import Any, List, Tuple, Optional

import numpy as np


from prefer.utils.random_utils import set_random_seed
from prefer.model_based_representations.interface import LatentSpaceMoleculeGenerator


class CDDDGeneratorModel(LatentSpaceMoleculeGenerator):
    def __init__(self, dir: str, seed: int = 0, num_workers: int = 6, **kwargs: Any):
        super().__init__(dir, **kwargs)

        self.num_workers = num_workers

        set_random_seed(seed)

        self._set_inference_model(dir)

        self._dir = dir

        self._can_decode_from_scaffold = False
        # By default we will store the stats in the model dir (but a different loc might be passed to
        # sampling benchmark or to distribution matching benchmark, which will change it):
        self.latent_space_stats_file_dir = dir

    def _set_inference_model(self, dir):
        from cddd.inference import InferenceModel

        self._inference_model = InferenceModel(
            model_dir=dir,
            use_gpu=True,
            cpu_threads=self.num_workers,
            gpu_mem_frac=0.75,
            batch_size=4096,
        )
        self._latent_size = self._inference_model.hparams.emb_size

    def encode(self, smiles_list: List[str]) -> List[np.ndarray]:
        """See parent class."""
        return list(self._inference_model.seq_to_emb(smiles_list))

    def get_name(self) -> str:
        return "CDDD"

    @classmethod
    def is_valid_dir(cls, model_dir: str) -> bool:
        file_name = os.path.join(model_dir, "hparams.json")
        try:
            if not os.path.exists(file_name):
                return False

            with open(file_name, "rt") as fh:
                # Bizarrely, the file contains a quoted JSON string, so we need a double-load here:
                hparams = json.loads(json.load(fh))

            return hparams["model"] == "NoisyGRUSeq2SeqWithFeatures"
        except Exception:  # Parse errors, key error, etc.
            return False

    def set_extra_args(self, **kwargs):
        workers = kwargs.get("num_workers")
        if workers is not None:
            self.num_workers = kwargs.get("num_workers")
        latent_space_stats = kwargs.get("latent_space_stats_file_dir")
        if latent_space_stats is not None:
            self.latent_space_stats_file_dir = kwargs.get("latent_space_stats_file_dir")
