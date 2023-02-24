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


from prefer.src.prefer_model_wrapper import PreferModelWrapper
from sklearn.linear_model import LinearRegression
import numpy as np


class TestPreferModelWrapper(unittest.TestCase):
    def test_prefer_model_wrapper(self):
        fingerprint_length = 2048  # Default value in `get_fingerprints`

        # Dummy model predicts 3.0 everywhere
        X = np.zeros((2, fingerprint_length))
        y = np.dot(X, np.ones(fingerprint_length)) + 3
        model = LinearRegression().fit(X, y)

        # When molecule is un-scoreable, PreferModelWrapper gives worst possible score
        worst_score = 0.32

        wrapper = PreferModelWrapper(
            model=model,
            metadata={
                "problem_type": "regression",
                "best_model_representation": "FINGERPRINTS",
                "friendly_model_name": "jan",
                "desirability_scores": {"junk": [{"x": 0, "y": 1.0}, {"x": worst_score, "y": 0.0}]},
                "rep_model_id": "the_rep_model",
            },
        )
        scores = wrapper.predict(
            ["CC", "CCC", "unparseable SMILES"], is_smiles_func=True, rep_model_id=None
        )

        assertion = scores == [3.0, 3.0, 3.0]
        self.assertTrue(all(assertion) == True)


if __name__ == "__main__":
    unittest.main()
