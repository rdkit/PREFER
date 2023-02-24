
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


# # TODO .path_to_model handle the different cases

# ! /usr/bin/env python3
import argparse
import json
import os
import sys
import tempfile
import time
from azureml.pipeline.core.builder import PipelineStep, PipelineData
from pathlib import Path
from typing import List
import logging

from azureml.core import Workspace, ComputeTarget


from pyreporoot import project_root

logger = logging.getLogger(__name__)

from prefer.schema.config import GlobalConfig

from prefer.azure_ml.utils import (
    load_aml_config,
    get_workspace,
    get_authentication,
    make_pipeline_output_path,
    get_run_config,
    get_pipeline_scheduling_argparser,
    create_step,
    make_data_reference,
    submit_pipeline,
)
from prefer.azure_ml.aml_config import AmlConfig
from prefer.azure_ml.get_model_utils import make_model_reference
from prefer.azure_ml.model_registration_utils import create_registration_prefer_step


def mismatch(representation_to_compute, prefer_env):
    """
    Function to evaluate whether there is a mismatch between the representation_to_compute and the environment
    """
    if (representation_to_compute == "CDDD") and (prefer_env == "moler-environment.yml"):
        return True
    elif (representation_to_compute == "MOLER") and (prefer_env == "cddd-environment.yml"):
        return True
    else:
        return False


def create_prefer_pipeline_steps(
    args: argparse.Namespace, aml_config: AmlConfig, workspace: Workspace, tmpdir_to_use: str,
) -> List[PipelineStep]:

    pipeline_steps: List[PipelineStep] = []
    project_root_path = project_root(Path(__file__))
    input_data_path = make_data_reference(
        aml_config, workspace, args.path_to_df
    )  # it was inside of the for loop
    collect_run_PREFER_results_folders = []
    prefer_benches_common_folder = PipelineData(
        name="prefer_benches_common_folder",  # This is used later, see below
        datastore=workspace.datastores[aml_config.result_store_name],
        is_directory=True,
    )

    for representation_to_compute, path_to_model in zip(
        args.representations_to_compute, args.path_to_models
    ):
        print(f'>>>>>>> CURRENT REPRES: {representation_to_compute}')
        # check if mismatch
        # here somethinmg happen because apparently for CDDD it returned False and the cddd-env was not set at the really beginning (main of the code))
        if mismatch(representation_to_compute, args.prefer_env):
            print(f'Mismatch since prefer_env:{args.prefer_env}')
            continue
        else:

            aml_conda_environment = str(project_root_path.joinpath(args.prefer_env))

            run_config = get_run_config(aml_conda_environment, aml_config)
            properties_column_name_json_format = json.dumps(args.properties_column_name)

            experiment_name = args.experiment_name
            id_column_name = args.id_column_name
            desirability_scores = args.desirability_scores

            smiles_column_name = args.smiles_column_name
            problem_type = args.problem_type
            splitting_strategy = args.splitting_strategy
            properties_column_name = properties_column_name_json_format

            if path_to_model == "not_specified":
                input_model_path = path_to_model
                input_model_path_flag = False
            else:
                input_model_path = make_model_reference(
                    aml_config,
                    workspace,
                    path_to_model,
                    model_storage_aml_config_field="datastore_name",
                )
                input_model_path_flag = True

            representation_loc = make_pipeline_output_path(
                aml_config, workspace, representation_to_compute
            )

            conda_env_name = os.path.basename(args.prefer_env)
            conda_env_name = os.path.splitext(conda_env_name)[0]

            get_repr_args = [
                "--path_to_df",
                input_data_path,
                "--representation_to_compute",
                representation_to_compute,
                "--output_dir",
                representation_loc,
                "--experiment_name",
                experiment_name,
                "--id_column_name",
                id_column_name,
                "--smiles_column_name",
                smiles_column_name,
                "--properties_column_name",
                properties_column_name,
                "--splitting_strategy",
                splitting_strategy,
            ]

            # check if temporal info has been set
            if args.temporal_info_column_name is not None:
                temporal_info_column_name = args.temporal_info_column_name
                get_repr_args += ["--temporal_info_column_name", temporal_info_column_name]
            else:
                if splitting_strategy == "temporal":
                    raise ValueError(
                        "ERROR: temporal splitting startegy has been required but no temporal_info column has been indicated"
                    )

            pipeline_inputs = [input_data_path]

            # if input_model_path_check is not None:
            if input_model_path_flag:
                get_repr_args += ["--path_to_model", input_model_path]
                pipeline_inputs.append(input_model_path)
            else:
                get_repr_args += ["--path_to_model", path_to_model]

            print("PIPELINE get_representations")
            pipeline_steps.append(
                create_step(
                    tmpdir_to_use=tmpdir_to_use,
                    name=f"Get Representation for {representation_to_compute}",
                    script_name="prefer/scripts/get_representations.py",
                    arguments=get_repr_args,
                    inputs=pipeline_inputs,
                    outputs=[representation_loc],
                    compute_target=ComputeTarget(workspace, aml_config.compute_target_name),
                    runconfig=run_config,
                )
            )

            input_representation_path = representation_loc

            # Run PREFER
            prefer_results_loc = PipelineData(
                name=f"results_prefer_{representation_to_compute}",  # This is used later, see below
                datastore=workspace.datastores[aml_config.result_store_name],
                is_directory=True,
            )
            print("PIPELINE run_PREFER")
            pipeline_steps.append(
                create_step(
                    tmpdir_to_use=tmpdir_to_use,
                    name=f"run PREFER for {representation_to_compute}",
                    script_name="prefer/scripts/run_PREFER.py",
                    arguments=[
                        "--problem_type",
                        problem_type,
                        "--representation_name",
                        representation_to_compute,
                        "--repr_dir",
                        input_representation_path,
                        "--final_folder_path",
                        prefer_results_loc,
                        "--experiment_name",
                        experiment_name,
                    ],
                    inputs=[input_representation_path],
                    outputs=[prefer_results_loc],
                    compute_target=ComputeTarget(workspace, aml_config.compute_target_name),
                    runconfig=run_config,
                )
            )

            collect_run_PREFER_results_folders.append(prefer_results_loc)

            # Add creation of the wrapper

            path_to_df = args.path_to_df
            model_folder_path = prefer_results_loc  # where results are stored

            wrapper_results = PipelineData(
                name=f"wrapper_results_{representation_to_compute}",  # This is used later, see below
                datastore=workspace.datastores[aml_config.result_store_name],
                is_directory=True,
            )
            print("PIPELINE model_wrapper")
            pipeline_steps.append(
                create_step(
                    tmpdir_to_use=tmpdir_to_use,
                    name=f"Creating Wrapper for {representation_to_compute}",
                    script_name="prefer/scripts/model_wrapper.py",
                    arguments=[
                        "--path_to_df",
                        path_to_df,
                        "--path_to_model",
                        path_to_model,
                        "--experiment_name",
                        experiment_name,
                        "--id_column_name",
                        id_column_name,
                        "--smiles_column_name",
                        smiles_column_name,
                        "--properties_column_name",
                        properties_column_name,
                        "--problem_type",
                        problem_type,
                        "--representation_name",
                        representation_to_compute,
                        "--property_model_folder_path",
                        model_folder_path,
                        "--final_folder_path",
                        wrapper_results,
                        "--repr_dir",
                        input_representation_path,
                        "--desirability_scores",
                        desirability_scores,
                    ],
                    inputs=[model_folder_path, input_representation_path],
                    outputs=[wrapper_results],
                    compute_target=ComputeTarget(workspace, aml_config.compute_target_name),
                    runconfig=run_config,
                )
            )
            print("PIPELINE create_registration_prefer_step")
            registration_step = create_registration_prefer_step(
                aml_config=aml_config,
                workspace=workspace,
                user_name=args.user_name,
                run_name=args.run_name,
                model_prefer_wrapped=wrapper_results,
                run_config=run_config,
                tmpdir_to_use=tmpdir_to_use,
                local_or_global_model="global",
                conda_env_name=conda_env_name,
            )
            pipeline_steps.append(registration_step)

        # at this level use bench_folders_list to loop over all the folders where the bench objects have been stored for each representation, collect the final tables and combine the results
        prefer_combined_results_dir = PipelineData(
            name=f"prefer_combined_results_dir",  # This is used later, see below
            datastore=workspace.datastores[aml_config.result_store_name],
            is_directory=True,
        )
    print("PIPELINE combine_results")
    print("[WARNING] we assume here that the maximum number of representation is 4")
    # TODO (how to best combine the run_PREFER results in a dynamic way - e.g. without assuming the number of folders/representations that will be created?) - I have tried also with having a common folder where to move all the result in each loop, but this is not possible in azure
    arguments = []
    for index, _ in enumerate(collect_run_PREFER_results_folders):
        arguments.append(f"--benchs_folder{index+1}")
        arguments.append(collect_run_PREFER_results_folders[index])
    arguments = arguments + [
        "--store_result_folder",
        prefer_combined_results_dir,
        "--problem_type",
        problem_type,
    ]

    pipeline_steps.append(
        create_step(
            tmpdir_to_use=tmpdir_to_use,
            name=f"Combine PREFER results",
            script_name="prefer/scripts/combine_results.py",
            arguments=arguments,
            inputs=collect_run_PREFER_results_folders,
            outputs=[prefer_combined_results_dir],
            compute_target=ComputeTarget(workspace, aml_config.compute_target_name),
            runconfig=run_config,
        )
    )

    return pipeline_steps


def main(user_args: List[str]):

    parser = get_pipeline_scheduling_argparser(model_name="PREFER", default_raw_data_path="")

    parser.add_argument(
        "-pa",
        "--prefer_args",
        type=str,
        default=str(project_root(Path(__file__)).joinpath("config_logD_azure.yaml")),
        help="path to the .yaml file where configuration parameters are stored.",
    )

    parser.add_argument(
        "-env", "--prefer_env", type=str, help="name of the environment yaml file",
    )
    args = parser.parse_args(user_args)

    prefer_config = GlobalConfig.from_yaml_file(args.prefer_args)

    args.path_to_df = prefer_config.path_to_df
    args.experiment_name = prefer_config.experiment_name
    args.id_column_name = prefer_config.id_column_name
    args.smiles_column_name = prefer_config.smiles_column_name
    args.properties_column_name = prefer_config.properties_column_name_list
    args.problem_type = prefer_config.problem_type
    args.splitting_strategy = prefer_config.splitting_strategy
    args.desirability_scores = json.dumps(prefer_config.desirability_scores)
    args.temporal_info_column_name = prefer_config.temporal_info_column_name
    args.representations_to_compute = [repr_ for repr_ in prefer_config.representations.keys()]

    tempdir_vect = []
    pipeline_run_vect = []
    default_env = args.prefer_env
    args.prefer_env = default_env
    print(f"environment has been set to the default: {args.prefer_env}")
    if not args.prefer_env:
        print("environment is None")
        if (
            "CDDD" in args.representations_to_compute
        ):
            print("cddd-environment has been set for you.")
            args.prefer_env = "cddd-environment.yml"
        else:
            print("moler-environment has been set for you.")
            args.prefer_env = "moler-environment.yml"
    args.path_to_models = [
        model if model else "not_specified" for model in prefer_config.representations.values()
    ]
    # First try to parse pipeline parameters (to error out before we do anything else):
    pipeline_parameters = json.loads(args.pipeline_parameters)

    aml_config = load_aml_config(args.aml_config)
    aml_auth = get_authentication(
        args.tenant_id, args.service_principal_id, args.service_principal_password
    )
    workspace = get_workspace(aml_config, aml_auth)
    result_store = workspace.datastores[aml_config.result_store_name]

    # TODO add the temp_dir as optional input as follows:
    #   TMP_DIR_PREFIX = "prefer_test_"
    #   with tempfile.TemporaryDirectory(prefix=TMP_DIR_PREFIX, dir="/scratch") as tempdir:

    with tempfile.TemporaryDirectory() as tempdir:
        pipeline_steps = create_prefer_pipeline_steps(args, aml_config, workspace, tempdir)
        # This will print status info + a URL:
        pipeline_run = submit_pipeline(
            workspace=workspace,
            experiment_name=args.experiment_name,
            steps=pipeline_steps,
            pipeline_parameters=pipeline_parameters,
        )
        logger.info(f"Pipeline submitted:\n{pipeline_run.get_portal_url()}")
        timestr = time.strftime("%Y%m%d_%H%M%S")

    return pipeline_run


if __name__ == "__main__":
    main(sys.argv[1:])
