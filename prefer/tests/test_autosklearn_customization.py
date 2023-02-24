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
import yaml


from autosklearn.regression import AutoSklearnRegressor
from prefer.utils.models_utils import (
    get_autosklearn_customized_model,
    convert_atype_to_btype,
    convert_list_into_dict,
)


class TestAutosklearnCustomization(unittest.TestCase):
    def test_get_autosklearn_customized_model(self):

        prefer_args = "./file_for_test/config_PREFER_test_custom_autosklearn.yaml"
        a_yaml_file = open(prefer_args)
        parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)

        if "model_instance" in parsed_yaml_file:
            model_instance = parsed_yaml_file["model_instance"]
        else:
            model_instance = None

        ml = get_autosklearn_customized_model(
            model_instance=model_instance, model_type="regression", working_directory="."
        )

        self.assertTrue(isinstance(ml, AutoSklearnRegressor))

    def test_convert_atype_to_btype(self):
        a = 1
        b = "test"
        new_a = convert_atype_to_btype(a, b)
        self.assertTrue(isinstance(new_a, str))

    def test_convert_list_into_dict(self):
        list_ = ["key1 = value1", "key2 : value2"]
        dict_test = {"key1": "value1", "key2": "value2"}
        dict_ = convert_list_into_dict(list_)
        self.assertTrue(dict_test == dict_)


if __name__ == "__main__":
    unittest.main()
