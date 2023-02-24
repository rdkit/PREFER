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


import sys

from typing import Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors

import io
import base64



from prefer.model_based_representations.model_based_representations_factory import (
    load_model_from_directory,
)

import pandas as pd
import logging
import scipy.spatial.distance as ssd
from scipy.cluster import hierarchy



        
def prepare_config_file(where_to_store_config_file, path_to_df, id_column_name, smiles_column_name, 
                       property_list_names):
    import yaml

    dict_file = {'path_to_df' : path_to_df, 
                 'experiment_name' : 'fake_experiment', 
                 'id_column_name': id_column_name, 
                 'smiles_column_name': smiles_column_name, 
                 'properties_column_name_list': property_list_names,
                  'problem_type': 'regression', 
                 'splitting_strategy': 'regression'
                }
    
    if(not where_to_store_config_file.endswith('/')):
        where_to_store_config_file = where_to_store_config_file+'/'

    with open(f'{where_to_store_config_file}config.yaml', 'w') as file:
        documents = yaml.dump(dict_file, file)
    return f'{where_to_store_config_file}config.yaml', dict_file


def cluster_fps(fps):
    nfps = len(fps)
    print("Hierarchical clustering is selected. Dataset size is:", nfps)

    average_cluster_size = 8
    dists = []
    # Generate dist matrix
    for i in range(0, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        dists.append([1 - x for x in sims])
    # Change format of dist matrix to package-recognizable one
    disArray = ssd.squareform(dists)
    # Build model
    Z = hierarchy.linkage(disArray)

    # Cut-Tree to get clusters
    cluster_amount = int(nfps / average_cluster_size)
    x = hierarchy.cut_tree(Z, n_clusters=cluster_amount)

    # change the output format to mimic the output of Butina
    x = list(x.transpose()[0])
    cs = [[] for _ in range(max(x) + 1)]

    for i in range(len(x)):
        cs[x[i]].append(i)
    return cs


def generate_fingerprints(listMols):
    """
    Given a list of molecules this function returns the corresponding Fingerprints representation
    """

    fps = []
    # test if we have any molecule objects
    if not len(listMols):
        return fps
    # generate count fingerprints output for sklearn
    for i, fp in enumerate(
        rdFingerprintGenerator.GetCountFPs(listMols, fpType=rdFingerprintGenerator.MorganFP)
    ):
        cv = np.zeros((0,), np.int8)
        DataStructs.ConvertToNumpyArray(fp, cv)
        fps.append(cv)
    return fps


def generate2DDesc(listMols, removeIPC=True):
    """
    Given a list of molecules this function returns the corresponding 2DD representation
    """
    desc = []
    # test if we have any molecule objects
    if not len(listMols):
        return desc
    numDescriptors = len(Descriptors._descList)
    if removeIPC:
        numDescriptors -= 1
    for mol in listMols:
        arr = []
        # test for invalid/"empty" molecules
        if (mol is None) or (mol.GetNumAtoms() == 0):
            arr = [np.nan] * numDescriptors
            desc.append(np.array(arr))
            continue
        for dname, calcD in Descriptors._descList:
            if removeIPC and dname == "Ipc":
                continue
            arr.append(calcD(mol))
        desc.append(np.array(arr))
    return desc


def extract_representations(df: pd.DataFrame):
    """
    This function is used to extract, from a dataframe with structure:

    ¦ ID ¦ Smiles ¦ Property1 ¦ Property2 ¦ ... ¦ PropertyN ¦ Represent1 ¦ ... ¦ RepresentM

    only the representation columns names.
    """

    standard_cols = ["ID", "Smiles"]
    cols = df.columns.values
    representation_to_evaluate = [
        col for col in cols if ((col not in standard_cols) and ("Property" not in col))
    ]
    return representation_to_evaluate


def generate_molecule(df: pd.DataFrame):
    """
    Function for the generation of valid molecules
    """
    # Generate valid molecules
    if "Smiles" not in df.columns.values:
        raise ValueError("Dataframe is missing `Smiles` column")
    else:
        return [Chem.MolFromSmiles(i) for i in df.Smiles]




def check_if_nan(df_col_representation):
    """
    function to check if one of the element of the selected representation is nan and
    return the index of the corresponding row
    """
    rows_with_nans = []
    for index, vect in enumerate(df_col_representation):
        if np.isnan(vect).any():
            rows_with_nans.append(index)
    return rows_with_nans


def convert(
    df: pd.DataFrame, representation: str, properties: list
) -> Tuple[np.ndarray, np.ndarray]:
    """
    function to convert dataframes, containing featurea and labels, into a feature and label numpy arrays
    """
    check_representations = representation not in df.columns.values
    check_properties = sum([x not in df.columns.values for x in properties])
    if (check_representations > 0) or (check_properties > 0):
        raise ValueError("ERROR: representations or properties not in df")

    X = np.stack(df[representation].tolist())
    if len(properties) == 1:
        y = df[properties[0]].to_numpy()
    else:  # multitasking
        y = df[properties].values
    return X, y



def create_latent_representation(smiles, model_name, model_path):
    """
    Create a latent given a list of smiles, generator model name and path to the trained model
    """
    try:
        model = load_model_from_directory(model_path)
        res = model.encode(smiles)
        return list(res)
    except Exception as e:
        raise ValueError(f"ERROR: Could not generate latent representation. {e}")


def create_molecule_representation(smiles, repName: str):
    """
    Create fingerprints or 2D descriptors given a list of smiles and representation name
    """
    rep = []
    try:
        molecules = [Chem.MolFromSmiles(i) for i in smiles]
        if repName.upper() == "FINGERPRINTS":
            rep = generate_fingerprints(molecules)
        elif repName.upper() == "DESCRIPTORS2D":
            rep = generate2DDesc(molecules)
        else:
            raise ValueError(f"ERROR: Representation {repName} unknown")
        return rep
    except Exception as e:
        raise ValueError(f"ERROR: Could not generate representation. {e}")
        
        
        


def normalize_smiles(smiles):
    """
    SMILES round tripping used for normalization of input smiles
    """
    norm_smiles = []
    for i in smiles:
        try:
            mol = Chem.MolFromSmiles(i)
        except Exception as e:
            logging.warning(f"WARNING: Invalid SMILES {i}. {e}")
        if mol is None:
            logging.warning(f"WARNING: Invalid SMILES {i}")
            norm_smiles.append("")
            continue
        nsmi = Chem.MolToSmiles(mol, isomericSmiles=False)
        norm_smiles.append(nsmi)
    return norm_smiles


def fig_to_base64_(fig):

    img = io.BytesIO()
    fig.savefig(img, format="png", x_inches="tight")
    img.seek(0)
    return base64.b64encode(img.getvalue())


def save_png_as_html(fig, fig_name: str):
    encoded = fig_to_base64_(fig)
    my_html = '<img src="data:image/png;base64, {}">'.format(encoded.decode("utf-8"))

    with open(fig_name, "w") as f:
        f.write(my_html)
