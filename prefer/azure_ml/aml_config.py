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
import os
import json
from dataclasses import dataclass, fields
from prefer.azure_ml.exceptions import MissingEnvironmentVariable

ENV_PREFIX = "AML_CONFIG_"
ENV_PATH_NAMES = {
    "subscription_id": f"{ENV_PREFIX}SUBSCRIPTION_ID",
    "resource_group": f"{ENV_PREFIX}RESOURCE_GROUP",
    "workspace_name": f"{ENV_PREFIX}WORKSPACE_NAME",
    "compute_target_name": f"{ENV_PREFIX}COMPUTE_TARGET_NAME",
    "cpu_compute_target_name": f"{ENV_PREFIX}CPU_COMPUTE_TARGET_NAME",
    "datastore_name": f"{ENV_PREFIX}DATASTORE_NAME",
    "result_store_name": f"{ENV_PREFIX}RESULT_STORE_NAME",
    "keyvault_name": f"{ENV_PREFIX}KEYVAULT_NAME",
}


@dataclass
class AmlConfig:

    subscription_id: str
    resource_group: str
    workspace_name: str
    compute_target_name: str
    cpu_compute_target_name: str
    datastore_name: str
    result_store_name: str
    keyvault_name: str

    @classmethod
    def can_load_from_environment_variables(cls) -> bool:
        field_names = [x.name for x in fields(cls)]
        aml_config = {name: os.getenv(ENV_PATH_NAMES[name]) for name in field_names}
        if any(x is None for x in aml_config.values()):
            return False
        return True

    @classmethod
    def from_environment_variables(cls) -> AmlConfig:
        """Instantiate an AmlConfig from environment variables.
        If the relevant environment variables are not all set, raise MissingEnvironmentVariable"""
        field_names = [x.name for x in fields(cls)]
        aml_config = {name: os.getenv(ENV_PATH_NAMES[name]) for name in field_names}
        if not AmlConfig.can_load_from_environment_variables():
            raise MissingEnvironmentVariable
        return cls(**aml_config)

    @classmethod
    def from_file(cls, filename: str) -> AmlConfig:
        with open(filename, "rt") as fh:
            aml_config = json.load(fh)
            return cls(**aml_config)
