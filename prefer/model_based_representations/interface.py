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


"""Abstract base classes for molecule encoders, generators, etc."""
from abc import ABC, abstractmethod

from typing import ContextManager, List, Optional, Tuple, Iterable, Union

import numpy as np
from dirhash import dirhash
from pathlib import Path

Pathlike = Union[str, Path]


class MoleculeSampler(ABC):
    """
    A molecule generator that can sample random molecules.
    """

    def sample(self, num_samples: int) -> List[str]:
        """
        Sample SMILES strings using the wrapped model.

        Args:
            num_samples: Number of results to return.

        Returns:
            List of SMILES strings.
        """
        # The below is a default implementation, can be overwritten for specific models.
        return [smiles for smiles, _ in self.sample_with_emb(num_samples)]


class AbstractModelRepresentation(ContextManager):
    """
    Base class for all molecule encoders, decoders, and samplers, providing
    - Default implementations for ContextManager
    - A model_id based on hash of files in the directory where the model is saved
    - Model name identifying the type of model
    """

    def __init__(self, dir: Pathlike, model_id_file_patterns: Iterable[str] = ("*",), **kwargs):
        # As `dir_hash` takes a `str`, we need to explicitly cast it
        self._model_id = dirhash(str(dir), "sha256", match=model_id_file_patterns)

        # Any arguments that make their way into here were passed into a model, but not understood
        # by it. We intentionally allow this, since it allows the user to provide preferred choices
        # for arguments without checking if a given model supports them. However, we print a warning
        # to make this more explicit.
        if kwargs:
            print("The following arguments were provided and ignored:", list(kwargs.keys()))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def get_model_id(self) -> str:
        """

        Returns: the model id as a string

        """
        return self._model_id

    @abstractmethod
    def set_extra_args(self, **kwargs) -> None:
        pass

    @classmethod
    @abstractmethod
    def is_valid_dir(cls, model_dir):
        pass

    @abstractmethod
    def get_name(self) -> str:
        """

        Returns: a human-readable string to describe the model type (e.g. 'MoLeR').

        """
        raise NotImplementedError


class LatentSpaceMoleculeGenerator(AbstractModelRepresentation, MoleculeSampler):
    """
    Autoencoder / Latent Space based Generative Model
    """

    @abstractmethod
    def encode(self, smiles_list: List[str]) -> List[np.array]:
        """
        Map input molecules to points in vector space.
        Args:
            smiles_list: List of molecules as SMILES

        Returns: 2D array of molecules as vectors (latent space)
        TODO: should this be List[np.array]?

        """
        raise NotImplementedError
