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


import logging
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

directory_path = Path(__file__).parents[1]
SALTS_FILE = "docs/SaltsMod.txt"


from prefer.utils.check_input_dataframe import check_dataframe
from prefer.utils.data_utils import extract_representations


class MissingSmiles(ValueError):
    pass


class EmptyDataframe(ValueError):
    pass


class MissingRepresentation(ValueError):
    pass


def filter_and_normalize_smiles(smiles):

    uncharger = rdMolStandardize.Uncharger()
    # format is smarts, then flag to allow one to pick and choose which to include
    forbidden_elements = Chem.MolFromSmarts(
        "[Cu,Sb,As,Sn,Pt,Te,Pd,Lu,Ge,Zn,Cu,Co,Ni,Fe,Hg,Zr,Mn,Ag,Bi,Cd,Cr,Ti,Al,Au,Mo,V,Mg,In,Ga,Pb,Ca,W]"
    )
    remover = SaltRemover.SaltRemover(defnFilename=directory_path / SALTS_FILE)

    try:
        mol = Chem.MolFromSmiles(str(smiles))
    except Exception as e:
        logging.error(f"ERROR: Invalid SMILES {smiles}.{e}")

    if mol is None or mol.GetNumAtoms() > 100:
        return None

    res, deleted = remover.StripMolWithDeleted(mol)
    # add a flag in case you want to remove or keep the salts - Default we keep it without the salt
    if len(deleted) != 0:
        return None

    mol = uncharger.uncharge(mol)
    if mol is None:
        return None

    if Chem.SanitizeMol(mol, catchErrors=True):  # maybe the molecule has changed
        return None

    if mol.HasSubstructMatch(forbidden_elements):
        return None

    smi = Chem.MolToSmiles(mol, isomericSmiles=False)
    if len(smi) < 2 or "*" in smi or "R" in smi:
        return None

    return smi


def filter_and_normalize_mols(df):
    """
    function to filter the row dataset at the beginning of the benchmarking pipeline
    """

    # Turn off the warning
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    # Check if 'Smiles' columns is in the current dataframe
    if "Smiles" not in df.columns.values:
        raise MissingSmiles("ERROR: Smiles column not in the dataframe")

    uncharger = rdMolStandardize.Uncharger()
    # format is smarts, then flag to allow one to pick and choose which to include
    forbidden_elements = Chem.MolFromSmarts(
        "[Cu,Sb,As,Sn,Pt,Te,Pd,Lu,Ge,Zn,Cu,Co,Ni,Fe,Hg,Zr,Mn,Ag,Bi,Cd,Cr,Ti,Al,Au,Mo,V,Mg,In,Ga,Pb,Ca,W]"
    )
    # Define the list to store the indices of the rows to be dropped
    rows_to_drop = []

    for index, smile in enumerate(df["Smiles"]):

        mol = Chem.MolFromSmiles(str(smile))

        if mol is None:
            logging.warning("WARNING: mol is None for smile: " + str(smile))
            rows_to_drop.append(index)
            continue
        if mol.GetNumAtoms() > 100:
            rows_to_drop.append(index)
            continue
        remover = SaltRemover.SaltRemover(defnFilename=directory_path / SALTS_FILE)

        res, deleted = remover.StripMolWithDeleted(mol)
        # add a flag in case you want to remove or keep the salts - Default we keep it without the salt
        if len(deleted) != 0:
            rows_to_drop.append(index)
            continue

        mol = uncharger.uncharge(mol)
        if mol is None:
            rows_to_drop.append(index)
            continue

        if Chem.SanitizeMol(mol, catchErrors=True):  # maybe the molecule has changed
            rows_to_drop.append(index)
            continue

        if mol.HasSubstructMatch(forbidden_elements):
            rows_to_drop.append(index)
            continue

        smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        if len(smi) < 2 or "*" in smi or "R" in smi:
            rows_to_drop.append(index)
            continue

        # Update the smile at index index
        df["Smiles"][index] = smi

    print("Percentage of dropped molecule: " + str((len(rows_to_drop) * 100) / df.shape[0]))
    filtered_df = df.drop(rows_to_drop).reset_index(drop=True)
    if check_dataframe(
        filtered_df
    ):  # check whether the indices are all correct or something went wrong
        df = filtered_df
    else:
        raise ValueError(
            "ERROR: Problem with inidices. Maybe a reset_index() is needed. The dataset will not be updated."
        )
    return df


def find_nan(df, representation_to_evaluate=[], drop=False):
    """
    This function check for each dataset, each dataframe representations to evaluate if some rows contain nan values.
    The indices corresponding to the rows with nan values will be stored in molecules_to_drop variable and if drop is True
    the rows will be directly removed and the indices will be restored.
    """

    logging.info("filter nan values in the molecular representations")

    # Check if empty
    if df.empty:
        raise EmptyDataframe("ERROR: df is empty")

    # Extract representation_to_evaluate if empty
    if not representation_to_evaluate:
        representation_to_evaluate = extract_representations(df)
    elif not all([repr_ in df.columns.values for repr_ in representation_to_evaluate]):
        # Check if the representation is in the dataframe
        raise MissingRepresentation(
            "ERROR: One or more representations are not in the dataset stored. HINT: Run Molecules_Representations to compute the representations needed"
        )

    # find nan
    for representation in representation_to_evaluate:
        logging.info("For the representation " + representation)
        find_nan_vect = [np.isnan(x).any().sum() for x in df[representation]]
        if np.sum(find_nan_vect) > 0 and drop:
            indx_nan = [indx for indx, elem in enumerate(find_nan_vect) if elem > 0]
            logging.info("Drop Molecules at positions:" + str(indx_nan))
            df.drop(indx_nan, inplace=True)
            df = df.reset_index()
        else:
            logging.info(
                "No molecules need to be dropped for " + representation + " representation"
            )
