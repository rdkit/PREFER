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
from typing import Any, List, Optional, Type

from prefer.model_based_representations.interface import (
    LatentSpaceMoleculeGenerator,
    AbstractModelRepresentation,
)

import tensorflow as tf

if tf.__version__ >= "2.0.0":
    # MoLeR environment
    from prefer.model_based_representations.moler_wrapper import MoLeRGeneratorModel

    latent_space_models = [MoLeRGeneratorModel]
else:
    # CDDD environment
    from prefer.model_based_representations.cddd_wrapper import CDDDGeneratorModel

    latent_space_models = [CDDDGeneratorModel]
    
# Add here new model based molecular representation


def load_latent_model_from_directory(model_dir: str, **kwargs: Any) -> LatentSpaceMoleculeGenerator:
    model: LatentSpaceMoleculeGenerator = load_model_from_directory(model_dir, [], **kwargs)
    return model


def load_model_from_directory(
    model_dir: str,
    extra_model_types: Optional[List[Type[AbstractModelRepresentation]]] = None,
    **kwargs: Any,
) -> AbstractModelRepresentation:
    """Loads a model from the given directory.

    Note:
        This method will figure out the exact type of model from the data.
        Both `args` and `kwargs` are passed to the model's `__init__` method.

    Returns:
        An object implementing the AbstractModelRepresentation interface.
    """
    if extra_model_types is None:
        extra_model_types = []
    all_models = latent_space_models + extra_model_types
    if not os.path.isdir(model_dir):
        raise ValueError(f"{model_dir} is not a directory!")

    for cls in all_models:
        if cls.is_valid_dir(model_dir):
            return cls(model_dir, **kwargs)
    raise ValueError(f"{model_dir} does not contain any of the recognised model types.")
