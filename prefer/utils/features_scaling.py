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
import pandas as pd
import numpy as np


def scale_features(Xtrain, Xtest, scaling_type="standardization"):
    """
    method used to normalize features
    Xtrain and Xtest are 2D numpy arrays
    scaling_type is a tring that can be standardization or normalization
    """
    print("Scaling features")
    Xtrain_df = pd.DataFrame(Xtrain)
    Xtest_df = pd.DataFrame(Xtest)
    # compute mean and std of the train set
    means_ = Xtrain_df.mean()  # only from the training set
    stds_ = Xtrain_df.std()

    Xtrain_scaled = []
    for index, row in Xtrain_df.iterrows():
        scaled_row = apply_scaling(
            np.array(row), scaling_type=scaling_type, means=means_, stds=stds_
        )
        Xtrain_scaled.append(scaled_row)
    Xtrain_scaled = np.array(Xtrain_scaled)
    Xtrain_scaled_df = pd.DataFrame(Xtrain_scaled)

    Xtest_scaled = []
    for index, row in Xtest_df.iterrows():
        scaled_row = apply_scaling(
            np.array(row), scaling_type=scaling_type, means=means_, stds=stds_
        )
        Xtest_scaled.append(scaled_row)
    Xtest_scaled = np.array(Xtest_scaled)

    if scaling_type == "standardization":
        means_sc = Xtrain_scaled_df.mean()  # only from the training set
        stds_sc = Xtrain_scaled_df.std()

        if not (np.array(round(stds_sc, 1)) == 1.0).all():
            raise ValueError(
                "ERROR: when standardizing matrix; not all stds of the scaled matrix are 1.0"
            )
        if not (np.array(round(means_sc, 1)) == 0.0).all():
            raise ValueError(
                "ERROR: when standardizing matrix; not all means of the scaled matrix are 0.0"
            )

    if scaling_type == "normalization":
        means_sc = Xtrain_scaled_df.mean()  # only from the training set

        if not (np.array(round(means_sc, 1)) == 0.0).all():
            raise ValueError(
                "ERROR: when normalizing matrix; not all means of the scaled matrix are 0.0"
            )

    return Xtrain_scaled, Xtest_scaled, means_, stds_


def apply_scaling(features_vect, scaling_type="standardization", means=None, stds=None):
    """
    function to apply a specific scaling given means and stds.
    Inputs:
        - features_vect, must be a numpy array of the features related to a single sample
        - scaling_type, is a string that can be standardization or normalization
        - means, is a numpy array of the means (one for each feature)
        - stds, is a numpy array of the standard deviation values (one for each feature)
    Output:
        - numpy array of scaled features
    """

    # making sure the features_vect is numpy arrays:
    features_vect = np.array(features_vect)
    array_sum = np.sum(features_vect)
    array_has_nan = not np.isfinite(array_sum)
    # check if features_vect contains nan
    if array_has_nan:
        raise ValueError("features_vect provided to the apply_scaling contains nan - cannot scale.")

    if means is not None:
        # making sure the means is numpy arrays:
        means = np.array(means)

        # check the dimension
        if means.shape[0] != features_vect.shape[0]:
            raise ValueError(
                f"ERROR: features_vect dimension ({features_vect.shape[0]}) does not match with means dimension ({means.shape[0]})"
            )

        if stds is not None:
            # making sure the stds is numpy arrays:
            stds = np.array(stds)

            # check the dimension
            if stds.shape[0] != features_vect.shape[0]:
                raise ValueError(
                    "ERROR: features_vect dimension does not match with stds dimension"
                )

            if scaling_type == "standardization":

                features_vect = (features_vect - means) / stds
                # Replace possible inf with nans
                features_vect[features_vect == -np.inf] = np.nan
                features_vect[features_vect == np.inf] = np.nan
                # if zeros in stds we should also have nans in  features_vect
                if any(np.isnan(features_vect)) == (0 in stds):
                    if any(np.isnan(features_vect)):
                        # important check in case of zeros in stds
                        (stds_zeros,) = np.where(
                            stds == 0
                        )  # zeros in stds should correspond to nans in standardize array
                        features_vect_nans = [x[0] for x in np.argwhere(np.isnan(features_vect))]

                        if list(stds_zeros) == list(features_vect_nans):
                            return list(features_vect[~np.isnan(features_vect)])
                        else:
                            raise ValueError(
                                "ERROR: there is a problem with the standardization: no match between zeros in the stds vector and nans in the standardize features vector"
                            )
                    else:  # no zeros in stds and no nans in feature_vect
                        return list(features_vect)
                else:
                    raise ValueError(
                        "ERROR: found nans in standardize features_vect but no zeros in stds or viceversa"
                    )

            elif scaling_type == "normalization":
                features_vect = features_vect - means
                return list(features_vect)
            else:
                raise ValueError(
                    "ERROR: only standardization or normalization are possible scaling_type"
                )
        else:
            if scaling_type == "standardization":
                raise ValueError(
                    "ERROR: only normalization is possible since stds vector is not provided"
                )

            elif scaling_type == "normalization":
                features_vect = features_vect - means
                return list(features_vect)
            else:
                raise ValueError(
                    "ERROR: only standardization or normalization are possible scaling_type"
                )

    else:
        raise ValueError("ERROR: please provide a means vector (one value for each feature)")
