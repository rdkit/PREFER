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
# -*- coding: utf-8 -*-
import sys

import logging
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
import numpy as np


from rdkit import Chem
from rdkit.Chem import AllChem


from prefer.utils.data_utils import generate_molecule, generate_fingerprints, cluster_fps


def temporal_split(df, split_perc=0.8, seed=1):

    """
    Function to split dataset into train and test sets according to a temporal split
    """
    # check if df is empty
    if df.empty:
        raise ValueError("ERROR in temporal_split: df is an empty")

    # check if time column is in df
    if not any([pd.core.dtypes.common.is_datetime64_any_dtype(df[x].dtype) for x in df.columns]):
        raise ValueError(
            f"Temporal split is impossible: dataframe does not contain time information"
        )

    # check column with timestamp
    try:
        time_column = [
            x for x in df.columns if pd.core.dtypes.common.is_datetime64_any_dtype(df[x])
        ][0]
    except Exception as e:
        raise ValueError(f"Error: {e}")
    # order by timestamp
    df = df.sort_values(by=time_column)

    # Split dataset according to the split_perc
    N = len(df.index.values)
    limit = int(N * split_perc)
    index_train = df.index.values[0:limit]
    index_test = df.index.values[limit:]
    random.Random(seed).shuffle(index_train)
    random.Random(seed).shuffle(index_test)
    return index_train, index_test


def random_split(df, seed=1, perc=0.8, limit_def: int = None):
    # TODO possibly extend this for stratified random_split

    """
    Function to split dataset into train and test sets according to a random split
    """
    if df.empty:
        raise ValueError("Cannot split empty dataframe")
    else:
        x = list(range(df.shape[0]))
        random.Random(seed).shuffle(x)
        limit = int(len(x) * perc)

        if limit_def:
            if limit_def >= len(x):
                print(
                    f"ERROR: selected limit_def ({limit_def}) is >= to the total amount of samples ({len(x)}). A perc of {perc} will be set instead for test/train slitting"
                )
                print(
                    "Random split with training perc: "
                    + str(perc * 100)
                    + "% and seed: "
                    + str(seed)
                )
            else:
                print(
                    f"Default number of samples has been set for Random split. In particular {len(x[:limit_def])} samples for training and {len(x[limit_def:])} for testing. Seed set is {seed}."
                )
                limit = limit_def
        else:
            print(
                "Random split with training perc: " + str(perc * 100) + "% and seed: " + str(seed)
            )
        index_train = x[:limit]
        index_test = x[limit:]
        return index_train, index_test


def cluster_split(df, perc=0.8):
    # TODO possibly extend this for stratified cluster_split
    print("Cluster split")
    import math

    data = df.copy()

    # smiles column name
    if "Smiles" not in df.columns.values:
        raise ValueError("ERROR: Smiles column not in df")

    smiles = "Smiles"
    # remove rows with missing smiles
    data = data.loc[data[smiles].notnull()]
    molecules = generate_molecule(data)

    # generate array of fingerprints
    try:
        fps = [
            Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in molecules
        ]
    except Exception as e:
        print(f"problem in generating fingerprints: {e}")
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in molecules]

    # cluster & sort the clusters by size
    clusters = cluster_fps(fps)
    L = list(clusters)
    L.sort(key=lambda t: len(t), reverse=True)
    # concatenating all inner lists (clusters) in L
    indices = sum(L, [])

    size_train = int(math.ceil(len(data) * perc))
    index_train = indices[:size_train]
    index_test = indices[size_train:]
    index_train = np.array(index_train)
    index_test = np.array(index_test)

    random.Random(1).shuffle(index_train)
    random.Random(1).shuffle(index_test)
    return index_train, index_test


def cluster_split_euclidean(
    df, plot=False, perc=0.8, validation_index=False, validation_index_perc=0.1
):
    print("Cluster split")
    """
    Function to split dataset into train and test sets according to a cluster split
    """
    if df.empty:
        raise ValueError("Cannot split empty dataframe")
    # fingerprint representation is needed
    logging.info("Generate Morgan Fingerprints")
    df_copy = df.copy()
    molecules = generate_molecule(df_copy)
    fingerprints = generate_fingerprints(molecules)

    mol_representation = fingerprints
    if not mol_representation:
        raise ValueError("mol_representation is an empty vector")
    else:
        # Generate 100 clusters of the mol_representation representation
        # The definition of the number of clusters is done in an empirical way according to the dimension of the dataset
        # (~ 4000). In general this step can be improved by evaluating different clusters dimensions and eventually
        # check the corresponding silhouette
        # values, although it is not clear if this is a good indicator in the case of molecules mol_representation distribution.
        cluster = AgglomerativeClustering(n_clusters=100, affinity="euclidean", linkage="ward")
        clusters = cluster.fit_predict(mol_representation)

        # dataframe to associate mol_representation and clusters
        df_tmp = pd.DataFrame()
        df_tmp["mol_representation"] = mol_representation
        df_tmp["cluster"] = clusters

        # Clusters distribution
        if plot:
            width, height = plt.figaspect(1.68)
            plt.subplots(figsize=(10, 5))
            sns.set(font_scale=1.5)
            sns.distplot(df_tmp.cluster, bins=100, norm_hist=False)
            plt.ylabel("Density (# mols in the cluster)")
            plt.show()

        # Clusters distribution
        df_tmp_ord = (
            df_tmp.groupby(df_tmp["cluster"], as_index=False)
            .count()
            .sort_values(by=["mol_representation"], ascending=False)
            .reset_index()
            .drop(["index"], axis=1)
            .rename(columns={"mol_representation": "numb_of_mols"})
        )

        df_tmp["index"] = df_tmp.index
        merged = pd.merge(df_tmp_ord, df_tmp, how="left", on="cluster")

        # As in [https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.7b00166] the clusters were ordered in decreasing size
        # and training set compounds were selected starting from the largest cluster until at least 75% were accumulated
        # The remaining roughly 25% of singletons and small clusters became the held-out test set.
        # For some of the smallest data sets the cluster size had to be reduced to get ∼ 25% in small clusters.

        # Compute the size of the train and test set according to the size of the dataframe
        size_train = int(perc * len(mol_representation))
        index_train = []
        index_test = []

        index_train = merged["index"].values[:size_train]
        index_test = merged["index"].values[size_train:]
        index_train = np.array(index_train)
        index_test = np.array(index_test)
        random.Random(1).shuffle(index_train)
        random.Random(1).shuffle(index_test)
        return index_train, index_test
