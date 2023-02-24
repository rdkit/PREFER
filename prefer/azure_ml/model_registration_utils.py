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


#! /usr/bin/env python3
import time
from typing import List, Optional, Any, Dict

from azureml.core import Workspace, RunConfiguration, Run, ComputeTarget
from azureml.pipeline.core import PipelineData, PipelineStep
from pathlib import Path

from dirhash import dirhash

from prefer.azure_ml.reproducibility import get_current_api_version, get_git_status
from prefer.azure_ml.utils import create_step
from prefer.azure_ml.aml_config import AmlConfig

from pyreporoot import project_root


def name_property_model_for_registration(experiment_name: str, representation_name: str) -> str:
    return experiment_name + "_" + representation_name


def add_tags_to_aml_run(aml_run: Run, tags: Dict[str, Any]) -> None:
    # Add the tags to the AML run as well.
    for key, value in tags.items():
        aml_run.tag(key, value)

    # If the run has a parent (which should be the case for pipelines), also add the tags there.
    if aml_run.parent is not None:
        for key, value in tags.items():
            aml_run.parent.tag(key, value)


def create_registration_step(
    aml_config: AmlConfig,
    workspace: Workspace,
    model_name: str,
    user_name: str,
    run_name: Optional[str],
    model_training_output: PipelineData,
    eval_outputs: List[PipelineData],
    run_config: RunConfiguration,
    tmpdir_to_use: str,
) -> PipelineStep:
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = run_name or f"{timestamp}-{user_name}"

    git_status = get_git_status()
    repo_hash = _dir_hash(str(project_root(Path(__file__))))
    api_version = get_current_api_version()

    return create_step(
        tmpdir_to_use=tmpdir_to_use,
        name="Model Registration",
        script_name="prefer/azure_ml/model_registration_prefer.py",
        arguments=[
            model_name,
            "--timestamp",
            timestamp,
            "--user-name",
            user_name,
            "--run-name",
            run_name,
            "--git_status",
            git_status,
            "--repo_hash",
            repo_hash,
            "--api_version",
            api_version,
            model_training_output,
        ]
        + eval_outputs,
        inputs=[model_training_output] + eval_outputs,
        outputs=[],
        compute_target=ComputeTarget(workspace, aml_config.cpu_compute_target_name),
        runconfig=run_config,
    )


def create_registration_prefer_step(
    aml_config: AmlConfig,
    workspace: Workspace,
    user_name: str,
    run_name: Optional[str],
    model_prefer_wrapped: PipelineData,
    run_config: RunConfiguration,
    tmpdir_to_use: str,
    local_or_global_model: str,
    conda_env_name: str,
) -> PipelineStep:
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = run_name or f"{timestamp}-{user_name}"

    git_status = get_git_status()
    repo_hash = _dir_hash(str(project_root(Path(__file__))))
    api_version = get_current_api_version()

    return create_step(
        tmpdir_to_use=tmpdir_to_use,
        name="Model Registration",
        script_name="prefer/azure_ml/model_registration_prefer.py",
        arguments=[
            model_prefer_wrapped,
            "--timestamp",
            timestamp,
            "--user-name",
            user_name,
            "--run-name",
            run_name,
            "--git_status",
            git_status,
            "--repo_hash",
            repo_hash,
            "--api_version",
            api_version,
            "--local_or_global_model",
            local_or_global_model,
            "--conda_env_name",
            conda_env_name,
        ],
        inputs=[model_prefer_wrapped],
        outputs=[],
        compute_target=ComputeTarget(workspace, aml_config.compute_target_name),
        runconfig=run_config,
    )


def _dir_hash(folder_name: str, **kwargs) -> str:
    """
    Calculate SHA256 hash of a an entire folder tree, recursively
    Note: The multi-threaded version seems to be throwing an error currently, but the single threaded version
            is fast enough given how infrequently we expect this to be used
    """
    return dirhash(folder_name, "sha256", **kwargs)
