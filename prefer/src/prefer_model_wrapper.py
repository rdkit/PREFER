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
import logging

import numpy as np
import pandas as pd
from typing import Any, List, Optional, Tuple, Union, Dict


from prefer.utils.data_utils import create_molecule_representation, prepare_config_file
from prefer.utils.models_utils import create_model_based_molecule_representation
from prefer.utils.filtering import filter_and_normalize_smiles
from prefer.utils.features_scaling import apply_scaling


logger = logging.getLogger(__name__)


class PreferModelWrapper:

    """
    PreferModelWrapper class to create a wrapper for a trained model to able to use it "standalone" at predicition time.
    This includes data prepartaion as done during training and feature calculation.

    inputs:
    - model: trained model, e.g. RF or MLP
    - metadata: this is a dict of settings/paramters we need to store for the model:
        - mandatory:
            - problem_type: string defining the type of the problem; e.g. "regression" or "classification"
            - model_type: string defining the type of the model; e.g. "RandomForest" or "MLP"
            - model_representation: feature generator name e.g. "CDDD" or "FINGERPRINTS" or "Descriptors2D"
            - friendly_model_name: Model name to be exposed in a UI/client
            - desirability_scores: A list of dictionaries where each dictionary {"x": x, "y": y} defines
                                  a point on the desirability curve used to scale the output of the scoring
                                  function into the range [0, 1]. If None, a default desirability curve is
                                  used which is linear in the range [0, 1].
        - optional:
            - project_code: if applicable used to identify project specific models
            - rep_model_path: if applicable, path to a generator model for feature generation
            - features_scaling_type, is a string that can be standardization or normalization
            - features_means_vect, is a numpy array of the means (one for each feature)
            - features_stds_vect, is a numpy array of the standard deviation values (one for each feature)
    """

    def __init__(self, model, metadata: Dict[str, Any]):

        # non-optional attributes
        try:
            self.model = model
            self.problem_type = metadata["problem_type"]

            try:
                self.model_representation = metadata["best_model_representation"]
            except KeyError:
                # Old code had a typo
                self.model_representation = metadata["best_model_representation"]
            self.friendly_model_name = metadata["friendly_model_name"]
            # get desirability scores for all tasks of a model
            if(metadata["desirability_scores"]):
                self.desirability_scores = list(metadata["desirability_scores"].values())
            else:
                self.desirability_scores = None

        except Exception as e:
            raise ValueError(
                f"Please check your meta data, not all necessary settings are provided: {e}"
            )

        # optional attributes/settings
        if "project_code" in metadata:
            self.project_code = metadata["project_code"]
        else:
            self.project_code = None
        self.rep_model_id = metadata["rep_model_id"]
        try:
            self.rep_model_path = metadata["path_to_model"][self.model_representation]
        except Exception:
            self.rep_model_path = None

        if "features_scaling_type" in metadata:
            self.features_scaling_type = metadata["features_scaling_type"]
        else:
            self.features_scaling_type = None

        if "features_means_vect" in metadata:
            self.features_means_vect = metadata["features_means_vect"]
        else:
            self.features_means_vect = None

        if "features_stds_vect" in metadata:
            self.features_stds_vect = metadata["features_stds_vect"]
        else:
            self.features_stds_vect = None
            
        # this is for model based molecular representations         
            
        if 'prefer_path' in metadata:
            self.prefer_path = metadata["prefer_path"]
        else:
            self.prefer_path = None
            
            
        if 'dict_commands' in metadata:
            self.dict_commands = metadata["dict_commands"]
        else:
            self.dict_commands = None
            
        if 'probability_threshold' in metadata:
            self.probability_threshold = metadata["probability_threshold"]
        else:
            self.probability_threshold = None
            

    def _get_features(
        self,
        rep: Union[List[str], List[np.ndarray]],
        is_smiles_func: bool,
        rep_model_id: Optional[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert some molecules to np.ndarrays for scoring.

        If the molecules are given as SMILES strings, apply the featurisation specified by `self.model_representation`. Then one can 
        compute traditional molecular representation or model based representations on the fly. The supported ones are FINGERPRINTS, DESCRIPTORS2D,
        CDDD and MOLER. in case of other representations, one can still use the wrapper, but the molecules should be already featurized.

        If the molecules are already given as features, check they are not NaN and that the `rep_model_id` matches `self.rep_model_id`.

        Returns:
            (features, is_valid):
                features: float array of shape [num_mols, num_features] containing feature values.
                is_valid: boolean array of shape [num_mols] indicating which of the molecules were successfully featurised.

        """
    
        if is_smiles_func:
            rep: List[str]
            # create necessary representation first using smiles input
            norm_rep = [filter_and_normalize_smiles(smi) for smi in rep]
            print(f'We have a norm_rep == {norm_rep}')
            is_valid = np.array([x is not None for x in norm_rep], dtype=bool)
            print(f'We have a is_valid == {is_valid}')

            # Replace None with empty string so that featuriser doesn't crash.
            norm_rep = [x or "" for x in norm_rep]

            for smi, is_ok in zip(rep, is_valid):
                if not is_ok:
                    logger.warning(f"Invalid or filtered SMILES: {smi}")
            if (self.model_representation.upper() in ["CDDD", "MOLER"]) and self.dict_commands:
                if 'CDDD' in self.model_representation.upper():
                    current_model = 'CDDD'
                else:
                    current_model = 'MOLER'
                
                try:
                    # create and save fake dataframe where to store the smiles to convert
                    df_fake = pd.DataFrame()
                    df_fake['Smiles'] = norm_rep
                    df_fake['IDs'] = len(norm_rep)*1
                    df_fake['Property'] = len(norm_rep)*1
                    #create temporary folder
                    import os

                    # define the name of the directory to be created
                    if(self.prefer_path.endswith('/')):
                        self.prefer_path = self.prefer_path[:-1]
                    path = f"{self.prefer_path}/_tmp_PREFER/"

                    try:
                        os.mkdir(path)
                    except OSError:
                        print ("Creation of the directory %s failed" % path)
                    else:
                        print ("Successfully created the directory %s " % path)
                    df_fake.to_csv(f'{path}temporary_df.csv', index = False)
                    
                    config_path, dict_file = prepare_config_file(path, f'{path}temporary_df.csv', 'IDs', 'Smiles', ['Property'])
                    # create a new yaml confi file and use it in the commands with path_to_df = the new fake dataframe
                    
                    run_commands = self.dict_commands[current_model]['run']
                    
                    # Substitude prefer_args in the original commands
                    first_commands = run_commands.split(';')[:-1] # common commands to set up envs
                    last_commands = run_commands.split(';')[-1] # last command to call compute_model_based_representations
                    first_commands = ';'.join(first_commands) # connect common commands
                    index_to_substitude = last_commands.split().index("--prefer_args")+1 # in the call to the function check the position of the prefer_args input
                    last_commands = last_commands.split()
                    last_commands[index_to_substitude] = config_path # change the prefer_args input
                    last_commands = ' '.join(last_commands)
                    run_commands = first_commands+';'+last_commands # put everything together
                    print(f'Current command run in PREFER-model-wrapper is {run_commands}')
                    # path to representations that will be created
                    experiment_name_tmp = dict_file['experiment_name']
                    model_representations_path = f'./{self.model_representation}_representations_{experiment_name_tmp}'
                    
                    gen_rep = create_model_based_molecule_representation(run_commands, model_representations_path, dict_file['experiment_name'],dict_file['splitting_strategy'])
                    print(gen_rep)
                
                    import shutil

                    shutil.rmtree(f'{path}') 
                    return gen_rep, is_valid
                except Exception as e:
                    raise ValueError(e)
            elif self.model_representation.upper() in ["FINGERPRINTS", "DESCRIPTORS2D"]:
                gen_rep = create_molecule_representation(norm_rep, self.model_representation)
                return np.array(gen_rep), is_valid
            else:
                raise ValueError(f'Cannot compute the required featurization ({self.model_representation}) of the given smiles. Either the molecular representation required is not known (supported are MOLER, CDDD, FINGERPRINTS, DESCRIPTORS2D) or self.run_commands was not provided in the Wrapper initialization ({self.dict_commands})')
        else:
            # Molecules are already featurised.  Check the featurisation is the right one.
            if rep_model_id != self.rep_model_id or rep_model_id is None:
                raise ValueError("Embedding and property prediction model incompatible")

            is_valid = np.array([not np.isnan(x).any() for x in rep], dtype=bool)
            gen_rep = rep

            return np.array(gen_rep), is_valid

    @property
    def is_multi_task(self) -> bool:
        return len(self.desirability_scores) > 1

    def predict(
        self,
        rep: Union[List[str], List[np.ndarray]],
        is_smiles_func: bool = True,
        rep_model_id: Optional[str] = None,
    ) -> Tuple[List[float], np.ndarray]:
        """
        Given a list of representations (embeddings or smiles), return scores and an indication of which scores are valid.

        TODO currently does not support multi-task models.  For such models
        only the first score for each molecule is reported.

        """
        print(f"---> Model {self.friendly_model_name} | representation {self.model_representation}")

        gen_rep, is_valid = self._get_features(
            rep, is_smiles_func=is_smiles_func, rep_model_id=rep_model_id
        )
   
        if self.features_scaling_type is not None:
            # Scale features before prediction.
            gen_rep = [
                apply_scaling(
                    features_vect=x,
                    scaling_type=self.features_scaling_type,
                    means=self.features_means_vect,
                    stds=self.features_stds_vect,
                )
                for x in gen_rep
            ]

        # Scaling may yield NaNs if means are NaN or stds are zero.  In such cases, the score will not be valid.
        
        is_valid = is_valid & np.array([np.all(np.isfinite(x)) for x in gen_rep])
        model = self.model
        gen_rep = np.array([x for x in gen_rep])
        # Get the predictions
        if self.problem_type == "classification":
            if hasattr(model, "predict_proba"):
                # Classifiers from sklearn have a 'predict_proba' method.
                preds = model.predict_proba(gen_rep)[:, 1]
            else:
                preds = model.predict(gen_rep)
        elif self.problem_type == "regression":
            preds = model.predict(gen_rep)
        else:
            logger.error(f"Unknown problem type {self.problem_type}")
            preds = [np.nan for _ in rep]
            is_valid = np.zeros(len(rep), dtype=bool)

        return preds
