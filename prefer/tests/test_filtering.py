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

import numpy as np
import pandas as pd


from prefer.utils.filtering import filter_and_normalize_mols, find_nan


class TestFiltering(unittest.TestCase):
    def test_find_nan_1(self):
        # Empty df
        df = pd.DataFrame()
        representation_to_evaluate = ["Fingerprints", "_2DDescriptors", "Embedded_cddd"]
        with self.assertRaises(ValueError):
            find_nan(df, representation_to_evaluate)

    def test_find_nan_2(self):
        # Invalid representation
        df = pd.DataFrame(
            np.random.randint(0, 100, size=(100, 4)),
            columns=list(["Smiles", "ID", "Fingerprints", "_2DD"]),
        )
        representation_to_evaluate = ["invalid"]
        with self.assertRaises(ValueError):
            find_nan(df, representation_to_evaluate)

    def test_filter_salt(self):
        df = pd.DataFrame(
            np.random.randint(0, 100, size=(100, 3)), columns=list(["ID", "Fingerprints", "_2DD"])
        )
        with self.assertRaises(ValueError):
            filter_and_normalize_mols(df)


if __name__ == "__main__":
    unittest.main()
