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


#!/usr/bin/python

import argparse
import os
import sys
import time
import logging


from prefer.src.vector_molecule_representations import VectorMoleculeRepresentations
from prefer.src.benchmarking import Benchmarking
from prefer.utils.save_load import saving_procedure_autosklearn

import tempfile

logger = logging.getLogger(__name__)


def run_PREFER_from_args(
    problem_type, representation_name, repr_dir, final_folder_path, experiment_name=None,
):

    os.makedirs(final_folder_path, exist_ok=True)

    if final_folder_path is None:
        final_folder_path = "."

    # Import saved representation
    repr_type = retrieve_type_of_molecular_representation(representation_name)
    list_if_files = os.listdir(repr_dir)
    if not repr_dir.endswith("/"):  # Normalise away trailing slashes
        repr_dir = repr_dir + "/"

    repr_ = repr_type.load(repr_dir + list_if_files[0])

    tasks_number = len([col for col in repr_.df.columns if "Property" in col])
    if tasks_number == 1:
        mask = False
    else:
        mask = True
    logger.info(tasks_number, mask)

    with tempfile.TemporaryDirectory() as tmpdirname:
        bench = Benchmarking(problem_type=problem_type, working_directory=tmpdirname,)
        try:
            bench.benchmark([repr_], experiment_name=experiment_name)
        except TypeError as e:
            logger.error("EXCEPTION during property model training: ", e)
            pass

        # saving procedure
        timestr = time.strftime("%Y%m%d-%H%M%S")
        name = representation_name
        if experiment_name is not None:
            name = name + "_" + experiment_name
        try:
            if not os.path.exists(final_folder_path):
                os.mkdir(final_folder_path)
        except OSError as e:
            logger.error("Creation of the directory %s failed", final_folder_path, e)
        else:
            logger.info("Successfully created the directory %s ", final_folder_path)
        dir_destination = final_folder_path + "/" + name + "_" + timestr

        saving_procedure_autosklearn(bench, dir_destination)
    return


def retrieve_type_of_molecular_representation(representation_name: str) -> type:
    return VectorMoleculeRepresentations


if __name__ == "__main__":
    """
    Example of usage:
    %run run_PREFER.py -pt "regression" -rn "FINGERPRINTS" -mn "RandomForest" -rd "/path/to/representation/PREFER_automation_branch/" -pg '{"max_depth": [10], "min_samples_leaf": [2], "n_estimators": [10]}' -pge "{}"
    """
    parser = argparse.ArgumentParser(description="run PREFER")
    parser.add_argument(
        "-pt",
        "--problem_type",
        type=str,
        help="whether this is a <regression> or a <classification> problem",
        required=True,
    )

    parser.add_argument(
        "-rn",
        "--representation_name",
        type=str,
        help="name of the rapresentation to compute or path to the generator which is used to map smiles into embeddings",
        required=True,
    )  # here you can have a list representations so that is case this list has a lenght >1 then the first step is to combine the representations

    parser.add_argument(
        "-rd",
        "--repr_dir",
        type=str,
        help="directory where the selected representation is stored",
        required=True,
    )

    parser.add_argument(
        "-ffp",
        "--final_folder_path",
        type=str,
        help="directory where the results will be stored. If not specified results will be store in the "
        "current directory.",
    )

    parser.add_argument(
        "-en", "--experiment_name", type=str, help="name of the current experiment",
    )

    args = parser.parse_args()
    run_PREFER_from_args(
        problem_type=args.problem_type,
        representation_name=args.representation_name,
        repr_dir=args.repr_dir,
        final_folder_path=args.final_folder_path,
        experiment_name=args.experiment_name,
    )
