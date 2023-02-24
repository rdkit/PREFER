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


import argparse
import getpass
import json
import os
import shutil
import tempfile
import uuid
from typing import Callable, Any, List, Optional, Tuple
import logging

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azureml.core import Experiment, Workspace, Run
from azureml.core.authentication import (
    ServicePrincipalAuthentication,
    InteractiveLoginAuthentication,
    AbstractAuthentication,
)

from azureml.core.runconfig import RunConfiguration, CondaDependencies
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineStep
from azureml.pipeline.core.builder import PipelineData
from azureml.pipeline.steps import PythonScriptStep

from prefer.azure_ml.aml_config import AmlConfig

from pyreporoot import project_root
from pathlib import Path

logger = logging.getLogger(__name__)

INCLUDED_PREFIXES_JSON = "included_prefixes.json"


def add_aml_authentication_args(parser: argparse.ArgumentParser) -> None:
    default_path = (
        str(project_root(Path(__file__))) + "/prefer/azure_ml/aml_configuration/aml_config.json"
    )
    parser.add_argument(
        "--aml-config",
        type=str,
        default=default_path,
        help="JSON file storing details about AzureML bits to use.",
    )
    parser.add_argument(
        "--tenant-id",
        type=str,
        help="ID of the Azure tenant. Required for non-interactive authentication.",
    )
    parser.add_argument(
        "--service-principal-id",
        type=str,
        help="ID of the Azure service principal. Required for non-interactive authentication.",
    )
    parser.add_argument(
        "--service-principal-password",
        type=str,
        help="Password for the Azure service principal. Required for non-interactive authentication.",
    )
    parser.add_argument(
        "--wait-for-completion",
        action="store_true",
        help="Block until submitted AzureML run has completed. Useful for smoke-testing.",
    )


def get_pipeline_scheduling_argparser(
    model_name: str, default_raw_data_path: str,
) -> argparse.ArgumentParser:
    """
    Construct argument parser for pipeline scheduling tools, exposing standard arguments.

    Args:
        model_name: Name of the model handled by the pipeline.
        default_raw_data_path: Location of the default raw data for this model, which
            can then be preprocessed for training purposes.

    Returns:
        ArgumentParser object which can be used to process CLI arguments, potentially after
        adding more arguments.
    """
    parser = argparse.ArgumentParser(
        description=f"Schedule training & eval of {model_name} on AzureML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    return update_pipeline_scheduling_argparser(model_name, default_raw_data_path, parser)


def update_pipeline_scheduling_argparser(
    model_name: str, default_raw_data_path: str, parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """
    Construct argument parser for pipeline scheduling tools, exposing standard arguments.

    Args:
        model_name: Name of the model handled by the pipeline.
        default_raw_data_path: Location of the default raw data for this model, which
            can then be preprocessed for training purposes.

    Returns:
        ArgumentParser object which can be used to process CLI arguments, potentially after
        adding more arguments.
    """

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=f"{model_name}-Pipeline",
        help="Name for experiment to hold this pipeline run.",
    )
    parser.add_argument(
        "--user-name",
        type=str,
        default=getpass.getuser(),
        help="Name of the scheduling user (used for model tagging).",
    )
    parser.add_argument(
        "--run-name", type=str, help="Name of the pipeline run (used for model tagging).",
    )
    parser.add_argument(
        "--raw-data-path",
        type=str,
        default=default_raw_data_path,
        help="Path of the raw input data in the configured storage.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path of the processed input data in the configured storage. If not specified, will schedule preprocessing steps.",
    )
    parser.add_argument(
        "--pipeline-parameters",
        type=str,
        default="{}",
        help="JSON dictionary of pipeline parameter values, overriding the defaults.",
    )

    add_aml_authentication_args(parser)
    return parser


def schedule_pipeline_from_args(
    args: argparse.Namespace,
    step_creation_fn: Callable[[argparse.Namespace, AmlConfig, Workspace, str], List[PipelineStep]],
) -> Run:
    """
    Schedule a pipeline run based on the arguments. This function takes care of generic
    submission bits, and `step_creation_fn` is responsible for providing the actual
    steps of the pipeline.

    Args:
        args: Result of an argument parse using the ArgumentParser from
            `get_pipeline_scheduling_argparser`.
        step_creation_fn: Callable function that creates the actual pipeline steps.

    Returns:
        As a side-effect, submits an experiment and logs the URL to the monitoring
        page on STDOUT. It also returns the same URL for any downstream use.
    """
    aml_config = load_aml_config(args.aml_config)
    aml_auth = get_authentication(
        args.tenant_id, args.service_principal_id, args.service_principal_password
    )
    workspace = get_workspace(aml_config, aml_auth)
    return schedule_pipeline_from_args_and_aml_config(args, aml_config, workspace, step_creation_fn)


def schedule_pipeline_from_args_and_aml_config(
    args: argparse.Namespace,
    aml_config: AmlConfig,
    workspace: Workspace,
    step_creation_fn: Callable[[argparse.Namespace, AmlConfig, Workspace, str], List[PipelineStep]],
) -> Run:
    """
    Schedule a pipeline run based on the arguments. This function takes care of generic
    submission bits, and `step_creation_fn` is responsible for providing the actual
    steps of the pipeline.

    Args:
        args: Result of an argument parse using the ArgumentParser from
            `get_pipeline_scheduling_argparser`.
        aml_config: AmlConfig instance with AML configurations.
        workspace: AML Workspace instance.
        step_creation_fn: Callable function that creates the actual pipeline steps.

    Returns:
        As a side-effect, submits an experiment and logs the URL to the monitoring
        page on STDOUT. It also returns the same URL for any downstream use.
    """
    # First try to parse pipeline parameters (to error out before we do anything else):
    pipeline_parameters = json.loads(args.pipeline_parameters)

    with tempfile.TemporaryDirectory() as tempdir:
        pipeline_steps = step_creation_fn(args, aml_config, workspace, tempdir)

        # This will print status info + a URL:
        run = Experiment(workspace, args.experiment_name).submit(
            Pipeline(workspace=workspace, steps=pipeline_steps),
            pipeline_parameters=pipeline_parameters,
            continue_on_step_failure=True,
        )
        logger.info(f"run url: {run.get_portal_url()}")
        if args.wait_for_completion:
            run.wait_for_completion()
        return run


def load_aml_config(filename: str) -> AmlConfig:
    """
    Load AML Config from file for using AzureML.

    Args:
        filename: Path to a file in JSON format.
    """
    print(f"loading configuration from: {filename}")
    aml_config = AmlConfig.from_file(filename)
    return aml_config


def load_aml_config_from_env_or_file(fallback_filename: str) -> AmlConfig:
    """
    Load AML Config from environment variables or from file for using AzureML.
    All varibales have to be set for the config to load from the environment.
    Else config is loaded from file.

    Args:
        filename: Path to a file in JSON format.
    """
    if AmlConfig.can_load_from_environment_variables():
        logger.info("loading configuration from environment variables")
        aml_config = AmlConfig.from_environment_variables()
        return aml_config
    else:
        logger.info(f"loading configuration from: {fallback_filename}")
        return AmlConfig.from_file(fallback_filename)


def get_instrumentation_key_from_config(aml_config: AmlConfig) -> str:
    # TODO: Can we delete the feature that fetches the instrumentation key from os.environ()?
    # this would force it to be in the vault.
    if "INSTRUMENTATION_KEY" in os.environ:
        return os.environ["INSTRUMENTATION_KEY"]
    client = get_keyvault_client_from_config(aml_config)
    try:
        return client.get_secret("instrumentation-key").value
    except Exception:
        logger.warning("Could not find Application Insights instrumentation key in Key Vault")
        return "00000000-0000-0000-0000-000000000000"


def _try_get_service_principal_from_mount():
    # fetching usecase credentials which are mounted on filesystem through KV integration
    try:
        with open("/mnt/secrets/f1a-productsp-appid") as f:
            app_id = f.read().strip()
            os.environ["AZURE_CLIENT_ID"] = app_id
        with open("/mnt/secrets/f1a-productsp-pwd") as f:
            app_pwd = f.read().strip()
            os.environ["AZURE_CLIENT_SECRET"] = app_pwd
        with open("/mnt/secrets/f1a-tenantid") as f:
            tenant_id = f.read().strip()
            os.environ["AZURE_TENANT_ID"] = tenant_id
        logger.info("secret found in mount, running in F1A")
    except OSError:
        logger.info("secret not found in mount")


def get_keyvault_client_from_config(aml_config: AmlConfig):
    keyvault_name = aml_config.keyvault_name
    keyvault_uri = f"https://{keyvault_name}.vault.azure.net/"

    # DefaultAzureCredentials will try ServicePrincipal auth, Environment auth, ManagedIdentity auth or User auth,
    # whichever is available.
    _try_get_service_principal_from_mount()
    credential = DefaultAzureCredential()
    return SecretClient(vault_url=keyvault_uri, credential=credential)


def get_run_config(
    conda_env_path: str,
    aml_config: AmlConfig,
    python_path: Tuple[str, ...] = (
        "prefer/model_based_representations/models/cddd/",
        "prefer/model_based_representations/models/molecule-generation/", # here you can add other path to new submodules added
    ),
) -> RunConfiguration:
    """
    Args:
        conda_env_path: Path to a YAML file speciying a conda environment.

    Returns:
        A run configuration which can be used when scheduling AML jobs.
    """
    run_config = RunConfiguration(conda_dependencies=CondaDependencies(conda_env_path))
    # Default to a container with GPU support - we can also use this on compute targets
    # without GPUs.
    run_config.environment.docker.base_image = "mcr.microsoft.com/azureml/base-gpu:latest"
    # Add Application isights logs
    prova = ":".join(python_path)
    run_config.environment_variables = {
        "APPLICATIONINSIGHTS_CONNECTION_STRING": f"InstrumentationKey={get_instrumentation_key_from_config(aml_config)}",
        "PYTHONPATH": ":".join(python_path),
    }

    return run_config


def get_authentication(
    tenant_id: Optional[str] = None,
    service_principal_id: Optional[str] = None,
    service_principal_password: Optional[str] = None,
) -> AbstractAuthentication:
    """
    Select an appropriate authentication method and return the handler object for that.

    Args:
        tenant_id: Optional ID of the Azure tenant. Required for non-interactive authentication.
        service_principal_id: Optional ID of the Azure service principal. Required for
            non-interactive authentication.
        service_principal_password: Optional password for the Azure service principal. Required
            for non-interactive authentication.

    Returns:
        AbstractAuthentication object which can be used to access AzureML services.
    """
    if (
        tenant_id is not None
        and service_principal_id is not None
        and service_principal_password is not None
    ):
        # Do authentication via Service Principal (used in CI)
        return ServicePrincipalAuthentication(
            tenant_id, service_principal_id, service_principal_password
        )
    else:
        # This will force CLI authentication (through the devicelogin UX).
        logger.info(f"Using interactive authentication.")
        return InteractiveLoginAuthentication()


def get_workspace(
    aml_config: AmlConfig, authentication: Optional[AbstractAuthentication] = None,
):
    """
    Construct an AzureML workspace object.

    args:
        aml_config: Configuration of AML use (e.g., workspace/subscription details).
        authentication: Configured authentication method.
    """
    workspace = Workspace.get(
        name=aml_config.workspace_name,
        subscription_id=aml_config.subscription_id,
        resource_group=aml_config.resource_group,
        auth=authentication,
    )
    return workspace


def covert_path_on_datastore_to_name(path: str) -> str:
    return path.replace("/", "__").replace(".", "_").replace("-", "_")


def make_data_reference(
    aml_config: AmlConfig,
    workspace: Workspace,
    path: str,
    aml_config_field: Optional[str] = "datastore_name",
) -> DataReference:
    """
    Create a reference to a path in the datastore of the workspace.

    Args:
        aml_config: Configuration of AML use (e.g., workspace/subscription details).
        workspace: AzureML workspace object containing the datastore to use.
        path: Path in the datastore (as string).
        datastore_name: str with the datastore name
            The default is "datastore_name": "02s_preprocessed_data".
            For optimisation or benchmarking you can pass "optimisation_datastore_name"
            See aml_config_*.json files for more.

    Returns:
        DataReference to the path which can be used in AML runs/pipeline steps.
    """
    if path.endswith("/"):  # Normalise away trailing slashes
        path = path[:-1]
    return DataReference(
        datastore=workspace.datastores[getattr(aml_config, aml_config_field)],
        data_reference_name=covert_path_on_datastore_to_name(path),
        path_on_datastore=path,
    )


def make_pipeline_output_path(
    aml_config: AmlConfig, workspace: Workspace, name: str
) -> PipelineData:
    """
    Create a fresh output path for PipelineSteps.

    Args:
        aml_config: Configuration of AML use (e.g., workspace/subscription details).
        workspace: AzureML workspace object containing the datastore to use.
        name: Name of the output.

    Returns:
        PipelineData which can be used to tie AML runs/pipeline steps together.
    """
    return PipelineData(
        name=name, datastore=workspace.datastores[aml_config.result_store_name], is_directory=True,
    )


def copy_filtered_file_tree(
    source_dir: str, target_dir: str, include_filter_fn: Optional[Callable[[str], bool]] = None,
) -> None:
    """
    Copy contents of `source_dir` to `target_dir`, but only include those files for which
    `include_filter_fn` returns True when called on the path relative to `source_dir`.

    Args:
        source_dir: Source directory to copy from.
        target_dir: Target directory to copy from.
        include_filter_fn: Optional Callable, has to return True for every file that should
            be included. If not provided, all files will be copied.

    Returns:
        None, but will modify contents of `target_dir`.
    """
    # Normalise code_dir to include the final directory separator:
    source_dir = os.path.join(source_dir, "")

    if not os.path.exists(source_dir):
        raise Exception(f"Directory {source_dir} does not exist.")
    if include_filter_fn is None:
        include_filter_fun: Callable[[str], bool] = lambda _: True
    else:
        include_filter_fun = include_filter_fn

    for dir, _, filenames in os.walk(source_dir):
        relative_dir_name = dir[len(source_dir) :]
        for filename in filenames:
            relative_file_name = os.path.join(relative_dir_name, filename)
            # First check if we should include this:
            if include_filter_fun(relative_file_name):
                # Make sure target directory exists:
                os.makedirs(os.path.join(target_dir, relative_dir_name), exist_ok=True)

                # Copy file over (including metadata, hence copy2):
                shutil.copy2(
                    src=os.path.join(dir, filename),
                    dst=os.path.join(target_dir, relative_file_name),
                )



def create_step(
    tmpdir_to_use: str, *, script_name: str, **step_kwargs: Any,
):
    """
    Create a PythonScriptStep AML pipeline step using a subset of files in `source_dir`.

    Args:
        tmpdir_to_use: Temporary directory in which to stage job submission. Note that this
            needs to still exist when created steps are being submitted (hence we cannot
            create a fresh temporary directory here).
        script_name: Name of the entry script for the AzureML run
        step_kwargs: Other arguments which will be passed to the PythonScriptStep constructor.
    """

    # We just need to create fresh name in the process-wide tempdir - this does not need to be
    # secure because that tempdir is fully controlled by us:
    tempdir = None
    while tempdir is None:
        tempdir = os.path.join(tmpdir_to_use, uuid.uuid4().hex)
        if os.path.exists(tempdir):
            tempdir = None

    with open(os.path.dirname(os.path.abspath(__file__)) + f"/{INCLUDED_PREFIXES_JSON}") as f:
        included_prefixes_all_scripts = json.load(f)
        included_prefixes = included_prefixes_all_scripts[script_name]

    def _normalise_paths(paths: List[str]) -> List[str]:
        # Do Windows/Unix normalisation: We assume paths are specified in Unix convention with / as
        # separator. To make things work on windows, we need to fix things up:
        if os.path.sep == "\\":
            return [p.replace("/", "\\") for p in paths]
        else:
            return paths

    included_prefixes = {k: _normalise_paths(v) for k, v in included_prefixes.items()}

    # First copy only the source files we need to the temp directory:
    # Backend files

    project_root_dir = os.path.join(project_root(__file__), "")

    # Other files - basically just azure_ml files
    copy_filtered_file_tree(
        source_dir=str(os.path.join(project_root_dir)),
        target_dir=tempdir,
        include_filter_fn=_make_filter_fn_from_prefixes_and_blocked_suffixes(
            allowed_prefixes=included_prefixes["project_root"], blocked_suffixes=[],
        ),
    )
    # Then construct the step consuming these:
    return PythonScriptStep(script_name=script_name, **step_kwargs, source_directory=tempdir,)


def _make_filter_fn_from_prefixes_and_blocked_suffixes(
    allowed_prefixes: List[str], blocked_suffixes: List[str],
) -> Callable[[str], bool]:
    """
    Make a filter function for use with `copy_filtered_file_tree` by specifying allowed path
    prefixes and forbidden path suffixes.

    Args:
        allowed_prefixes: Generated function will only return True for strings starting with
            one of the entries of this argument.
        blocked_suffixes: Generated function will return False for strings containing any of
            the entries of this argument.

    Returns:
        Callable taking a string and returning a boolean value as described above.
    """

    def filter_fn(s: str):
        if any(s.startswith(prefix) for prefix in allowed_prefixes):
            if all(suffix not in s for suffix in blocked_suffixes):
                return True
        return False

    return filter_fn


def submit_pipeline(*, workspace, experiment_name, steps, pipeline_parameters):
    return Experiment(workspace, experiment_name).submit(
        Pipeline(workspace=workspace, steps=steps),
        pipeline_parameters=pipeline_parameters,
        continue_on_step_failure=True,
    )
