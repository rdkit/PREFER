The analysis_notebooks folder contains three notebooks to compute the performances of the PREFER models for each molecular representation and/or the performances of the best model against a RandomForest baseline.
In order to run the notebook, one should first run a PREFER job (e.g. through the Run_PREFER.ipynb notebook) in order to train PREFER models for each molecular representation. After that one can run the notebboks in the analysis_notebooks folder in the following order:

1. TestSet_Bootstrapping.ipynb
2. Plot_performance_distributions_for_representation.ipynb AND/OR Best_PREFER_model_VS_RF.ipynb 