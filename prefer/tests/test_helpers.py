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
import unittest
import pandas as pd
import numpy as np


from prefer.utils.models_utils import output_dataframe_preparation


class TestHelpers(unittest.TestCase):
    def test_output_dataframe_preparation_singleTask(self):
        # list of lists
        data = [
            ["a1", "b1", "c1"],
            ["a2", "b2", "c2"],
            ["a3", "b3", "c3"],
            ["a4", "b4", "c4"],
            ["a5", "b5", "c5"],
        ]

        df = pd.DataFrame(data)
        df["Property_1"] = [
            "test_label1",
            "train_label1",
            "test_label2",
            "train_label2",
            "train_label3",
        ]
        index_train = [1, 3, 4]
        index_test = [0, 2]
        predictions_train = ["train_val1", "train_val2", "train_val3"]
        predictions_test = ["test_val1", "test_val2"]
        expected_df = df.copy()
        expected_df["model_predictions_property_1"] = [
            "test_val1",
            "train_val1",
            "test_val2",
            "train_val2",
            "train_val3",
        ]
        expected_df["is_train"] = [False, True, False, True, True]

        output_df = output_dataframe_preparation(
            df,
            index_train=index_train,
            index_test=index_test,
            predictions_train=predictions_train,
            predictions_test=predictions_test,
        )

        all_collect = []
        for col in output_df.columns:
            all_collect.append(all(output_df[0].values == expected_df[0].values))
        all_collect.append(all(output_df.columns.values == expected_df.columns.values))

        self.assertTrue(all(all_collect))

    def test_output_dataframe_preparation_multiTask(self):
        # list of lists
        data = [
            ["a1", "b1", "c1"],
            ["a2", "b2", "c2"],
            ["a3", "b3", "c3"],
            ["a4", "b4", "c4"],
            ["a5", "b5", "c5"],
        ]

        df = pd.DataFrame(data)
        df["Property_1"] = [
            "test_label1",
            "train_label1",
            "test_label2",
            "train_label2",
            "train_label3",
        ]

        df["Property_2"] = [
            "test_label1",
            "train_label1",
            "test_label2",
            "train_label2",
            "train_label3",
        ]
        index_train = [1, 3, 4]
        index_test = [0, 2]
        predictions_train = np.array(
            [["train_val1", "train_val2", "train_val3"], ["train_val1", "train_val2", "train_val3"]]
        )
        predictions_train = predictions_train.T
        predictions_test = np.array([["test_val1", "test_val2"], ["test_val1", "test_val2"]])
        predictions_test = predictions_test.T
        expected_df = df.copy()
        expected_df["model_predictions_property_1"] = np.array(
            ["test_val1", "train_val1", "test_val2", "train_val2", "train_val3"]
        )
        expected_df["model_predictions_property_2"] = np.array(
            ["test_val1", "train_val1", "test_val2", "train_val2", "train_val3"]
        )
        expected_df["is_train"] = [False, True, False, True, True]

        output_df = output_dataframe_preparation(
            df,
            index_train=index_train,
            index_test=index_test,
            predictions_train=predictions_train,
            predictions_test=predictions_test,
        )

        all_collect = []
        for col in output_df.columns:
            all_collect.append(all(output_df[0].values == expected_df[0].values))
        all_collect.append(all(output_df.columns.values == expected_df.columns.values))

        self.assertTrue(all(all_collect))


if __name__ == "__main__":
    unittest.main()
