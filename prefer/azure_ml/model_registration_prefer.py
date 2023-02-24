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
import json
import os
import pickle
import time
import logging

from azureml.core.run import Run

from prefer.azure_ml.model_registration_utils import (
    add_tags_to_aml_run,
    name_property_model_for_registration,
)
from prefer.azure_ml.telemetry_utils import (
    set_telemetry_handlers,
    function_span_metrics_decorator,
)

logger = logging.getLogger(__name__)
set_telemetry_handlers(logger, "Training")


@function_span_metrics_decorator("training.model_registration_prefer.run_from_args", "Training")
def run_from_args(args):
    aml_run = Run.get_context()

    model_dir = args.MODEL_PREFER_WRAPPED_DIR
    pkls = [name for name in os.listdir(model_dir) if name.endswith(".pkl")]
    assert len(pkls) == 1
    wrapped_model_name = pkls[0]

    with open(os.path.join(model_dir, wrapped_model_name), "rb") as in_file:
        wrapped_model = pickle.load(in_file)
    rep_model_id = wrapped_model.rep_model_id

    problem_type = wrapped_model.problem_type
    local_or_global_model = args.local_or_global_model

    is_best = local_or_global_model == "local"
    representation_name = wrapped_model.model_representation
    registration_name = name_property_model_for_registration(
        wrapped_model.friendly_model_name, representation_name
    )

    cloud_target_dir = "packaged_model"
    # # (1) Save the wrapped model:
    aml_run.upload_file(
        path_or_stream=os.path.join(model_dir, wrapped_model_name),
        name=os.path.join(cloud_target_dir, wrapped_model_name),
    )

    # (2) Save metadata
    metadata = {
        "git_status": args.git_status,
        "repo_hash": args.repo_hash,
        "api_version": args.api_version,
    }

    metadata_filename = "metadata.json"

    with open(metadata_filename, "wt") as fh:
        json.dump(metadata, fh)

    aml_run.upload_file(
        path_or_stream=metadata_filename, name=os.path.join(cloud_target_dir, metadata_filename),
    )

    # (3) Register the model with tags and properties.
    tags = {
        "is_best": is_best,
        # All property prediction models are active by default (i.e. not marked for future deletion) because:
        # - only the best local model is registered, therefore it makes sense that it will not be deleted by default.
        # - all global models are registered, and the data scientist could potentially pick any of them for production.
        "is_active": True,
    }

    # TODO why do we need to store the desirability curve with the property model?

    properties = {
        "generative_or_property_model": "property_model",
        "property_model_friendly_name": wrapped_model.friendly_model_name,
        "problem_type": problem_type,
        "representation_name": representation_name,
        "generative_model_id": rep_model_id,
        "local_or_global_model": args.local_or_global_model,
        "timestamp": args.timestamp,
        "user_name": args.user_name,
        "run_name": args.run_name,
        "git_status": args.git_status,
        "repo_hash": args.repo_hash,
        "api_version": args.api_version,
        "project_code": wrapped_model.project_code,
        "desirabilities": wrapped_model.desirability_scores,
        "env": args.conda_env_name,
    }

    aml_run.register_model(
        model_name=registration_name, model_path=cloud_target_dir, tags=tags, properties=properties,
    )

    add_tags_to_aml_run(aml_run, tags)
    aml_run.add_properties(properties)


def run():
    import argparse

    parser = argparse.ArgumentParser(description="Model registration script (PREFER).")
    parser.add_argument(
        "MODEL_PREFER_WRAPPED_DIR",
        type=str,
        help="Directory with the pickled PREFER model with metadata.",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=time.strftime("%Y-%m-%d_%H-%M-%S"),
        help="Timestamp of the run (used for model tagging).",
    )
    parser.add_argument(
        "--user-name",
        type=str,
        default="unknown_user",
        help="Name of the scheduling user (used for model tagging).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="unknown_run_name",
        help="Name of the pipeline run (used for model tagging).",
    )
    parser.add_argument(
        "--git_status",
        type=str,
        required=True,
        help="Git status (branch, commit, etc) at the time of training.",
    )
    parser.add_argument(
        "--repo_hash",
        type=str,
        required=True,
        help="Hash of the repo root at the time of training.",
    )
    parser.add_argument(
        "--api_version", type=str, required=True, help="API version supported by the model.",
    )
    parser.add_argument(
        "--local_or_global_model",
        type=str,
        choices=["local", "global"],
        help="local/global",
        required=True,
    )
    parser.add_argument(
        "--conda_env_name",
        type=str,
        default="moler-environment",
        help="Name of the conda environment used to build the model.",
    )
    args = parser.parse_args()

    run_from_args(args)


if __name__ == "__main__":
    run()
