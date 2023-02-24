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


import re
from typing import Dict, List, NamedTuple, Optional
import logging

from azureml.core import Experiment, Workspace
from azureml.core.model import Model
from azureml.core.run import Run
from azureml.data.data_reference import DataReference

from prefer.azure_ml.utils import covert_path_on_datastore_to_name
from prefer.azure_ml.aml_config import AmlConfig

logger = logging.getLogger(__name__)


class DatastorePath(NamedTuple):
    datastore_name: str
    path_on_datastore: str


def get_model_path(model: Model, workspace: Workspace) -> DatastorePath:
    """Obtain the path corresponding to a registered model of any kind."""
    # Join up all tags and properties.
    tags_and_properties = {**model.tags, **model.properties}

    experiment = Experiment(workspace=workspace, name=tags_and_properties["run_name"])
    run = Run(experiment=experiment, run_id=model.run_id)

    # Take all input directories to the registration step...
    data_references = run.get_details()["runDefinition"]["dataReferences"].items()

    # ...and exclude those corresponding to eval results or the HTML report. We expect to see these
    # additional input paths for generator models, while for property prediction models we don't;
    # either way it doesn't hurt to filter them out.
    data_references = [
        value for (key, value) in data_references if "_eval" not in key and key != "report"
    ]

    # Hopefully, we're left with exactly one input path: the model itself.
    if len(data_references) != 1:
        raise ValueError(
            f"Got {len(data_references)} candidates for the model path: {data_references}."
        )

    [model_data_reference] = data_references

    return DatastorePath(
        datastore_name=model_data_reference["dataStoreName"],
        path_on_datastore=model_data_reference["pathOnDataStore"],
    )


def get_generator_model_path(name: str, version: int, workspace: Workspace) -> DatastorePath:
    """Obtain the path corresponding to a trained generator model (e.g. MoLeR:2)."""

    # Get the model, then follow the links to the run that registered it.
    if is_prod_alias(name):
        models = Model.list(
            workspace=workspace, properties=[[get_prod_alias_property_name(), f"{name}:{version}"]]
        )

        if len(models) != 1:
            raise ValueError(f"Got {len(models)} candidates for the prod alias: {name}.")

        [model] = models
    else:
        model = Model(workspace=workspace, name=name, version=version)

    return get_model_path(model, workspace)


def get_property_model_path(
    property_name: str, generative_model_id: str, workspace: Workspace
) -> DatastorePath:
    """Obtain the path corresponding to a property model (trained with a given generator)."""

    full_name = property_name
    logger.debug(f"Looking up property model for {property_name} under {full_name}.")
    models = Model.list(
        workspace,
        name=full_name,
        tags=[["is_best", "True"]]
    )

    if not models:
        raise ValueError(
            f"No property models found for {full_name} with generator id {generative_model_id}."
        )

    if len(models) > 1:
        all_versions = sorted([model.version for model in models])
        logger.debug(f"Found {len(models)} property model candidates with versions {all_versions}.")
    else:
        logger.debug(f"Found exactly one property model candidate.")

    model = max(models, key=lambda m: m.version)
    logger.debug(f"Selecting the latest model with version {model.version}.")

    return get_model_path(model, workspace)


class ModelNameAndVersion(NamedTuple):
    name: str
    version: str


def parse_registered_model_name(name: str) -> Optional[ModelNameAndVersion]:
    # Pattern to match a registered model name (e.g. MoLeR:2).
    if not name:
        return None
    pattern = "([a-zA-Z0-9\\-_]*):(\\d*)"
    match = re.match(pattern, name)

    if match:
        return ModelNameAndVersion(name=match.group(1), version=match.group(2))
    else:
        return None


def resolve_path_or_name(
    path_or_name: str,
    aml_config: AmlConfig,
    workspace: Workspace,
    model_storage_aml_config_field: str,
) -> DatastorePath:
    """Resolve a string which is either a model path or registered model name into a model path."""
    name_and_version = parse_registered_model_name(path_or_name)

    logger.info(f"Model NAME and VERSION {name_and_version}")

    if name_and_version is None:
        # Input doesn't look like a model name (so it must be a path).
        return DatastorePath(
            datastore_name=getattr(aml_config, model_storage_aml_config_field),
            path_on_datastore=path_or_name,
        )
    else:
        # Input can be parsed as a model name, so map it into a storage path.
        return get_generator_model_path(
            name_and_version.name, int(name_and_version.version), workspace
        )


def make_model_reference(
    aml_config: AmlConfig,
    workspace: Workspace,
    path_or_name: str,
    model_storage_aml_config_field: str,
    aml_graph_node_name: Optional[str] = "",
) -> DataReference:
    """Create a reference to a model in the datastore of the workspace.

    Args:
        aml_config: Configuration of AML use (e.g., workspace/subscription details).
        workspace: AzureML workspace object containing the datastore to use.
        path_or_name: Either a path in the datastore, or a registered model name (e.g. MoLeR:2).

    Returns:
        DataReference to the path which can be used in AML runs/pipeline steps.
    """
    path = resolve_path_or_name(path_or_name, aml_config, workspace, model_storage_aml_config_field)

    logger.info(f"Final AML PATH to model:  {path}")

    if aml_graph_node_name == "":
        aml_graph_node_name = covert_path_on_datastore_to_name(path.path_on_datastore)

    if path.path_on_datastore.endswith("/"):
        # Normalise away trailing slashes
        path = path._replace(path_on_datastore=path.path_on_datastore[:-1])

    return DataReference(
        datastore=workspace.datastores[path.datastore_name],
        data_reference_name=aml_graph_node_name,
        path_on_datastore=path.path_on_datastore,
    )


def make_property_model_reference(
    workspace: Workspace, property_name: str, generator_model_id: str
) -> DataReference:
    path = get_property_model_path(property_name, generator_model_id, workspace)

    if path.path_on_datastore.endswith("/"):  # Normalise away trailing slashes
        path.path_on_datastore = path.path_on_datastore[:-1]

    return DataReference(
        datastore=workspace.datastores[path.datastore_name],
        # Sensible AML graph node name
        data_reference_name=property_name,
        path_on_datastore=path.path_on_datastore,
    )


def make_all_property_model_references(
    workspace: Workspace, property_names: List[str], generator_model_id: str
) -> Dict[str, DataReference]:
    """
    Retrieve all property model references and return a dict of property model names and refs

    Args:
        workspace: AML workspace.
        property_names: List of property names.
        generator_model_id: Generator model id.

    Returns:
        Dict with property names as keys and property model data references as values.
    """
    return {
        property_name: make_property_model_reference(workspace, property_name, generator_model_id)
        for property_name in property_names
    }


def get_model(workspace: Workspace, model_type: str, model_version: str):
    """
    Construct an AzureML model object.

    args:
        workspace: Workspace where model is registered.
        model_type: AML Model type name.
        model_version: AML Model version.
    Returns:
        A Model if one exists in the workspace.
    Raises:
        ModelNotFoundException if model is not found in workspace.
    """
    model = Model(workspace, name=model_type, version=model_version)
    return model


def get_model_rep_name(gen_model_name: str):
    model_rep_name = ""
    name_version = parse_registered_model_name(gen_model_name)
    if name_version is not None:
        model_rep_name = name_version.name
        if is_prod_alias(model_rep_name):
            suff = get_prod_suffix()
            model_rep_name = model_rep_name.replace(suff, "")
    return model_rep_name


def get_prod_suffix() -> str:
    return "-Prod"


def get_prod_alias_property_name() -> str:
    return "prod_alias"


def is_prod_alias(name: str) -> bool:
    return name.endswith(get_prod_suffix())
