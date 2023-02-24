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


from __future__ import annotations
from typing import List, Dict, Optional
import yaml
import pydantic
from pydantic import Extra

"""pydantic Models defining structure of a PREFER config YAML file"""


class PreferConfig(pydantic.BaseModel):
    class Config:
        extra = Extra.forbid

    problem_type: str
    experiment_name: str
    smiles_column_name: str
    id_column_name: str
    desirability_scores: Optional[Dict[str, List[Dict[str, float]]]]
    splitting_strategy: str = "random"

    @classmethod
    def from_yaml_file(cls, path: str) -> PreferConfig:
        with open(path) as f:
            parsed_yaml = yaml.load(f, Loader=yaml.FullLoader)
        return cls.parse_obj(parsed_yaml)


class LocalConfig(PreferConfig):
    """Config for training a local (project-specific) property model"""

    assay_name: str
    project_code: str
    properties_column_name: str
    # TODO why do we have different field names 'datapath' and 'path_to_df' for local and global models?
    datapath: str


class GlobalConfig(PreferConfig):
    """Config for training a global (not project-specific) property model"""

    properties_column_name_list: List[str]
    path_to_df: str
    representations: Dict[str, str]
    temporal_info_column_name: Optional[str]
