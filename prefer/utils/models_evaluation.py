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
import sys
import copy


import plotly as py
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    precision_recall_curve,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)


# Add PREFER dependency
from prefer.utils.models_utils import convert_prediction_types

# Add ghostml dependency
import ghostml


def compute_model_performances(
    model_type, X_train, y_train, X_test, y_test, metrics_names, refit_automl,
):
    """
    Function to fit the model selected during the nested cross-validation with the entire training set and test it on the test set.

    inputs:
    - model_type, string equal to "classification" or "regression"
    - X_train, training features
    - y_train, training labels
    - X_test, test features
    - y_test, test labels
    - metrics_names, name of the metric to compute for the evaluation of the model
    - best_estimator, best estimator found in the SearchCV
    - best_param, best set of hyperparametersrelated to the best estimator found
    - representation, name of the representation used
    """
    # Check if multitasking for correctly computing the metrics' values
    if y_test.ndim == 1:
        properties_num = 1
    else:
        properties_num = y_test.shape[1]

    # predict 
    print(f'any nan in the feature matrix? :{np.isnan(X_test).any()}')
    print('---------')
    print(X_test)
    if model_type == "regression":
        predict_train = refit_automl.predict(X_train)
        predict_test = refit_automl.predict(X_test)
    elif model_type == "classification":
        try:  # predict_proba in case of classifier
            predict_train = refit_automl.predict_proba(X_train) #assumption: in case of multitasking autosklearn predict_proba will give the probability of being in class 1 for each compounds for each label
            predict_test = refit_automl.predict_proba(X_test)
            # in the case of single task predict_proba with an autosklearn model will always give the 2 prob per sample (being in class0 and class1) - as normal sklearn
            if properties_num == 1:  
                predict_train = predict_train[:, 1]
                predict_test = predict_test[:, 1]
        except Exception as e:
            print(f"it was not possible to compute predict_proba: {e}")
            predict_train = refit_automl.predict(X_train)
            predict_test = refit_automl.predict(X_test)

    if model_type == "regression":
        metrics = evaluate_regression(
            predict_train, y_train, predict_test, y_test, metrics_names, refit_automl
        )
    elif model_type == "classification":
        # check if predicted probabilities have been computed instead of predicted classes
        unique_predict_test = np.unique(predict_test)
        unique_predict_train = np.unique(predict_train)
        if (len(unique_predict_test) == 2) or (len(unique_predict_train) == 2):
            raise ValueError(
                "Predicted values for test and/or train sets are not probability. Classification cannot be evaluated!"
            )
        metrics = evaluate_classification(
            predict_train,  # probs
            y_train,  # binary
            predict_test,  # probs
            y_test,
            metrics_names,
            refit_automl,
        )

    else:
        raise ValueError(f"ERROR: model_type {model_type} not known")

    y_score_copy = predict_test.copy()
    return metrics, predict_train, predict_test, y_score_copy


def evaluate_regression(predict_train, y_train, predict_test, y_test, metrics_names, estimator):
    """
    Function to test the model selected during the cross-validation with the entire test set and to compute the metrics related to the regression problem.
    inputs:
    - predict_train, predictions related to the training set
    - y_train, training labels
    - predict_test, predictions related to the test set
    - y_test, test labels
    - metrics_names, name of the metric to compute for the evaluation of the model
    - estimator, final best estimator for the representation under analysis
    """
    logging.info("Model predictions on the test set - regression problem")
    np.random.seed(7)
    if tf.__version__ >= "2.0.0":
        tf.random.set_seed(7)

    metrics = dict((k, []) for k in metrics_names)
    # Check if multitasking for correctly computing the metrics' values
    if y_test.ndim == 1:
        properties_num = 1
    else:
        properties_num = y_test.shape[1]

    if properties_num == 1:  # no multitasking
        logging.debug("Evaluate regression for single task")
        # Let us check wheather the mask for NaN values has been activated and if yes,
        # let us mask the value to mask of predict_test and test_labels
        test_labels = y_test
        train_labels = y_train
        mask_value = -1000
        # test sets
        predict_test = predict_test[y_test != mask_value]
        test_labels = y_test[y_test != mask_value]
        # train sets
        predict_train = predict_train[y_train != mask_value]
        train_labels = y_train[y_train != mask_value]

        test_error = predict_test - test_labels
        max_test_label = np.max(test_labels)
        min_test_label = np.min(test_labels)

        R2 = round(r2_score(test_labels, predict_test), 3)
        regr_metrics = {
            "RMSE_test": round(np.sqrt(mean_squared_error(test_labels, predict_test)), 3),
            "RMSE_train": round(np.sqrt(mean_squared_error(train_labels, predict_train)), 3),
            "R2": R2,
            "RMSE_test_Norm": round(
                np.sqrt(mean_squared_error(test_labels, predict_test))
                / (max_test_label - min_test_label),
                3,
            ),
            "Mean test error value": round(np.mean(test_error), 3),
            "Max test error value": round(np.max(test_error), 3),
            "Min error value": round(np.min(test_error), 3),
            "25th percentile (error)": round(np.percentile(test_error, 25), 3),
            "50th percentile (error)": round(np.percentile(test_error, 50), 3),
            "75th percentile (error)": round(np.percentile(test_error, 75), 3),
        }
    else:
        logging.debug("Evaluate regression for multitasking")
        R2 = []
        RMSE_test = []
        RMSE_train = []
        RMSE_test_Norm = []
        Mean_test_error_value = []
        Max_test_error_value = []
        Min_error_value = []
        percentile25 = []
        percentile50 = []
        percentile75 = []

        for i in range(properties_num):
            predict_test_tmp = predict_test[:, i]
            test_labels_tmp = y_test[:, i]
            train_labels_tmp = y_train[:, i]
            predict_train_tmp = predict_train[:, i]
            logging.debug("Mask has been activated and the Metrics are computed accordingly")
            mask_value = -1000
            predict_test_tmp = predict_test_tmp[test_labels_tmp != mask_value]
            test_labels_tmp = test_labels_tmp[test_labels_tmp != mask_value]
            predict_train_tmp = predict_train_tmp[train_labels_tmp != mask_value]
            train_labels_tmp = train_labels_tmp[train_labels_tmp != mask_value]

            R2.append(round(r2_score(test_labels_tmp, predict_test_tmp), 3))
            test_error = predict_test_tmp - test_labels_tmp

            max_test_label = round(np.max(test_labels_tmp), 3)
            min_test_label = round(np.min(test_labels_tmp), 3)
            RMSE_test.append(
                round(np.sqrt(mean_squared_error(test_labels_tmp, predict_test_tmp)), 3)
            )
            RMSE_train.append(
                round(np.sqrt(mean_squared_error(train_labels_tmp, predict_train_tmp)), 3)
            )
            RMSE_test_Norm.append(
                round(
                    np.sqrt(mean_squared_error(test_labels_tmp, predict_test_tmp))
                    / (max_test_label - min_test_label),
                    3,
                )
            )
            Mean_test_error_value.append(round(np.mean(test_error), 3))
            Max_test_error_value.append(round(np.max(test_error), 3))
            Min_error_value.append(round(np.min(test_error), 3))
            percentile25.append(round(np.percentile(test_error, 25), 3))
            percentile50.append(round(np.percentile(test_error, 50), 3))
            percentile75.append(round(np.percentile(test_error, 75), 3))
        regr_metrics = {
            "RMSE_test": RMSE_test,
            "RMSE_train": RMSE_train,
            "R2": R2,
            "RMSE_test_Norm": RMSE_test_Norm,
            "Mean test error value": Mean_test_error_value,
            "Max test error value": Max_test_error_value,
            "Min error value": Min_error_value,
            "25th percentile (error)": percentile25,
            "50th percentile (error)": percentile50,
            "75th percentile (error)": percentile75,
        }

    if (np.isnan(predict_test).any()) or (np.isinf(predict_test).any()):
        logging.warning("there are NaN or Inf in the predition on the test set --> ERROR!")
        regr_metrics = {
            "RMSE_test": 10000000,
            "RMSE_train": 10000000,
            "R2": -1000000,
            "RMSE_test_Norm": 100000,
            "Mean test error value": 1000000000,
            "Max test error value": 100000000,
            "Min error value": 1000000000,
            "25th percentile (error)": 1000000,
            "50th percentile (error)": 1000000000,
            "75th percentile (error)": 100000000,
        }

    for metric_name in metrics:
        if metric_name in regr_metrics:
            logging.info("Metric %s is %s", metric_name, str(regr_metrics[metric_name]))
            metrics[metric_name] = regr_metrics[metric_name]
        else:
            logging.debug("Metric %s wasn't computed in <Evaluate_Regression>", metric_name)
    return metrics


def evaluate_classification(
    predict_train, y_train, predict_test, y_test, metrics_names, estimator
):  
    """
    Function to test the model selected during the cross-validation with the entire test set and to compute the metrics related to the classification problem

    inputs:
    - predict_train, predictions related to the training set
    - y_train, training classes
    - predict_test, predictions related to the test set
    - y_test, test classes
    - metrics_names, name of the metric to compute for the evaluation of the model
    - estimator, final best estimator for the representation under analysis
    - y_score, test probabilities
    """
    # check if multitasking
    if y_test.ndim == 1:
        properties_num = 1
    else:
        properties_num = y_test.shape[1]

    if properties_num == 1:
        # in case of a single task it is needed to convert
        # Convert array of list into list
        predict_train = convert_prediction_types(predict_train)  # probs
        predict_test = convert_prediction_types(predict_test)  # probs

    logging.info("Model predictions on the test set - classification problem")
    # Metrics to compute
    metrics = dict((k, []) for k in metrics_names)
    np.random.seed(7)
    if tf.__version__ >= "2.0.0":
        tf.random.set_seed(7)
    balanced_accuracy = []
    F1_score = []
    precision = []
    ROC_AUC = []
    recall = []
    y_score_tmp = []
    kappa = []
    deltaAUPRC = []
    prob_threshold_vect = []

    # compute best thrshold with ghostml
    try:
        thresholds = np.round(np.arange(0.05, 0.55, 0.05), 2)
        # convert
        prob_threshold = ghostml.optimize_threshold_from_predictions(
            y_train, predict_train, thresholds, ThOpt_metrics="Kappa"
        )
        print(f"Setting the best threshold to: {prob_threshold}, according to GHOSTML")
    except Exception as e:
        print(
            f"ERROR in optimizing the classification threshold using GHOSTML - default threshold of 0.5 will be set instead: {e}"
        )
        prob_threshold = 0.5

    y_score_tmp = copy.copy(predict_test)  # predicted probabilities of test set
    if properties_num == 1:  # no multitasking
        # prepare vector for default 0.5 threshold
        predict_test_default = copy.copy(predict_test)
        predict_train_default = copy.copy(predict_train)
        logging.info("Classification Single Task")

        # Convert into classes
        predict_train[predict_train < prob_threshold] = 0
        predict_train[predict_train >= prob_threshold] = 1
        predict_test[predict_test < prob_threshold] = 0
        predict_test[predict_test >= prob_threshold] = 1

        # default 0.5 threshold
        predict_train_default[predict_train_default < 0.5] = 0
        predict_train_default[predict_train_default >= 0.5] = 1
        predict_test_default[predict_test_default < 0.5] = 0
        predict_test_default[predict_test_default >= 0.5] = 1

        # compute metrics which do not rely on thresholds
        y_true = y_test
        precision, recall, _ = precision_recall_curve(y_true, y_score_tmp)
        auc_score = auc(recall, precision)
        deltaAUPRC = round(auc_score - (np.sum(y_train) / len(y_train)), 3)
        ROC_AUC = round(roc_auc_score(y_true, y_score_tmp), 3)

        # compute metrics which rely on thresholds
        # ghostml threshold
        y_pred = predict_test
        balanced_accuracy = round(balanced_accuracy_score(y_true, y_pred), 3)
        F1_score = round(f1_score(y_true, y_pred), 3)
        precision = round(precision_score(y_true, y_pred), 3)
        recall = round(recall_score(y_true, y_pred), 3)
        kappa = round(cohen_kappa_score(y_true, y_pred), 3)

        # default 0.5 threshold
        y_pred = predict_test_default
        balanced_accuracy_default = round(balanced_accuracy_score(y_true, y_pred), 3)
        F1_score_default = round(f1_score(y_true, y_pred), 3)
        precision_default = round(precision_score(y_true, y_pred), 3)
        recall_default = round(recall_score(y_true, y_pred), 3)
        kappa_default = round(cohen_kappa_score(y_true, y_pred), 3)
        prob_threshold_vect = prob_threshold

    else:  # it is a multitasking
        logging.info("Classification Multitasking")
        # Convert into classes
        y_score_copy = copy.copy(predict_test)  
        # default threshold to be improved as in GHOSTML
        # TO DO: extend ghostml also for multitasking
        predict_train[predict_train < 0.5] = 0
        predict_train[predict_train >= 0.5] = 1
        predict_test[predict_test < 0.5] = 0
        predict_test[predict_test >= 0.5] = 1
        # Assign to used variables
        y_true = y_test
        y_pred = predict_test

        for i in range(properties_num):

            mask_value = -1000
            test_labels_tmp = y_true[:, i]
            y_score_tmp = y_score_copy[:, i][test_labels_tmp != mask_value]

            balanced_accuracy.append(
                round(
                    balanced_accuracy_score(
                        test_labels_tmp[test_labels_tmp != mask_value],
                        y_pred[:, i][test_labels_tmp != mask_value],
                    ),
                    2,
                )
            )
            F1_score.append(
                round(
                    f1_score(
                        test_labels_tmp[test_labels_tmp != mask_value],
                        y_pred[:, i][test_labels_tmp != mask_value],
                    ),
                    2,
                )
            )
            precision.append(
                round(
                    precision_score(
                        test_labels_tmp[test_labels_tmp != mask_value],
                        y_pred[:, i][test_labels_tmp != mask_value],
                    ),
                    2,
                )
            )
            recall.append(
                round(
                    recall_score(
                        test_labels_tmp[test_labels_tmp != mask_value],
                        y_pred[:, i][test_labels_tmp != mask_value],
                    ),
                    2,
                )
            )
            ROC_AUC.append(
                round(
                    roc_auc_score(test_labels_tmp[test_labels_tmp != mask_value], y_score_tmp), 2,
                )
            )
            # TODO implement deltaAUPRC for multitasking scenario
            deltaAUPRC.append(0)
            prob_threshold_vect.append(0.5)
            kappa.append(
                round(
                    cohen_kappa_score(
                        test_labels_tmp[test_labels_tmp != mask_value],
                        y_pred[:, i][test_labels_tmp != mask_value],
                    ),
                    3,
                )
            )

        # NOTE : default metrics will be the same as for not deafault ones because not yet implemented for multitasking
        balanced_accuracy_default = balanced_accuracy
        F1_score_default = F1_score
        precision_default = precision
        recall_default = recall
        kappa_default = kappa
    class_metrics = {
        "Balanced_Accuracy": balanced_accuracy,
        "F1_score": F1_score,
        "Precision": precision,
        "Recall": recall,
        "ROC_AUC": ROC_AUC,
        "kappa_score": kappa,
        "deltaAUPRC": deltaAUPRC,
        "prob_threshold": prob_threshold_vect,
        "Balanced_Accuracy_default": balanced_accuracy_default,
        "F1_score_default": F1_score_default,
        "Precision_default": precision_default,
        "Recall_default": recall_default,
        "kappa_score_default": kappa_default,
    }

    for metric_name in metrics:
        if metric_name in class_metrics:
            logging.info("Metric %s is %s", metric_name, class_metrics[metric_name])
            metrics[metric_name] = class_metrics[metric_name]
        else:
            logging.debug("Metric %s wasn't computed in <Evaluate_Classification>", metric_name)

    return metrics


def show_results(df_name, metrics, metrics_names, representation_names, model_names, plot_res=True):

    """
    Function to show the results as a table.

    inputs:
    - df_name, name of the experiment
    - best_grid, best hyperparameters found in the SearchCV
    - metrics, metrics computed to evaluate the model performances
    - metrics_names, name of the metrics computed to evaluate the model performances
    - representation_names, list of names of the representations evaluated
    - plot_res, boolean to indicate to show or not show the plots

    """
    index = []
    columns = ["Molecular Representation", "Prediction Model", "Model Params"] + metrics_names
    dict_ = dict((k, []) for k in columns)

    for repres_name in representation_names:

        for model_name in model_names:
            dict_["Molecular Representation"].append(repres_name)
            dict_["Prediction Model"].append(model_name)
            dict_["Model Params"].append("  ")
            for metric in metrics_names:
                dict_[metric].append(metrics[repres_name][metric])

            index.append(repres_name + "," + df_name)
    # Create a table for the model performances
    table = pd.DataFrame(dict_, index=index)
    table = table.T
    colors = ["white" for i in range(table.shape[1] + 1)]
    colors[0] = "lightgrey"

    # Save it as a csv
    table.to_csv("table_Model_performances.csv")
    table.reset_index(inplace=True)
    table.rename(columns={"index": "Metrics"}, inplace=True)
    if plot_res:
        # Show it
        save_html(table.T, df_name)

    table = table.set_index("Metrics")
    return table


def save_html(merged, df_name, path=None):
    table = merged.T
    colors = ["white" for i in range(table.shape[1] + 1)]
    colors[0] = "lightgrey"
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(table.columns),
                    fill_color="grey",
                    align="left",
                    font=dict(color="white", size=14),
                ),
                cells=dict(
                    values=[table[i].astype(str).values for i in table.columns.values],
                    fill=dict(color=colors),
                    font=dict(color="black", size=14),
                    line_color="darkslategray",
                    align="left",
                ),
            )
        ]
    )
    fig.update_layout(
        title=df_name,
        autosize=False,
        width=1700,
        height=700,
        yaxis=go.layout.YAxis(
            title_text="Y-axis Title",
            ticktext=["Very long label", "long label", "3", "label"],
            tickvals=[1, 2, 3, 4],
            tickmode="array",
            titlefont=dict(size=30),
        ),
    )

    if path:
        py.offline.plot(fig, filename=path)
    else:
        fig.show()


def plot_results(
    df_name,
    model_type,
    metrics,
    predictions_test,
    test_labels,
    representation_names,
    mask_value=-1000,
    path=None,
):
    """
    Function to plot the results according to the type of problem (regression/classification)

    inputs:
    - df_name, name of the experiment
    - model_type, classification or regression models
    - predictions_test, predictions computed on the test set
    - test_labels, test labels
    - representation_names, list of names of the representations evaluated
    - y_score, test probabilities
    - mask, boolean value to indicate whether a multitasking with scatter label matrix (mask = True) or a single task (mask = False) has been selected
    - mask_value, value used to replace missing values in the case of mask = True
    """
    if model_type == "regression":
        for repres_name in representation_names:
            if not isinstance(test_labels[repres_name], pd.DataFrame):
                test_labels_df = pd.DataFrame(test_labels[repres_name].tolist())
            else:
                test_labels_df = test_labels[repres_name]
            fig = make_subplots(rows=1, cols=len(test_labels_df.columns))
            for index, col in enumerate(test_labels_df.columns):
                predictions_df = pd.DataFrame(predictions_test[repres_name]).loc[:, index]
                ytest = test_labels_df

                index_ = ytest[ytest[col] != mask_value][col].index.values
                ytest_tmp = ytest[col].iloc[index_]
                ytest_tmp = ytest_tmp.reset_index(drop=True)
                predictions_df = predictions_df.iloc[index_]
                predictions_df = predictions_df.reset_index(drop=True)
                if len(test_labels_df.columns) == 1:
                    xmin = min(predictions_df.values)
                    xmax = max(predictions_df.values)
                    ymin = min(ytest_tmp.values)
                    ymax = max(ytest_tmp.values)
                    min_ = min([xmin, ymin])
                    max_ = max([xmax, ymax])
                    fig.add_trace(
                        go.Scatter(x=[min_, max_], y=[min_, max_], name="x=y"), row=1, col=col + 1,
                    )
                    m, b = np.polyfit(predictions_df, ytest_tmp, 1)
                    fig.add_trace(
                        go.Scatter(
                            x=predictions_df, y=m * predictions_df + b, name="interpolation"
                        ),
                        row=1,
                        col=col + 1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=predictions_df,
                            y=ytest_tmp,
                            mode="markers",
                            marker=dict(size=4),
                            name="labels VS predictions",
                        ),
                        row=1,
                        col=col + 1,
                    )
                    fig.update_layout(
                        height=600, width=800, title_text=df_name + ": " + repres_name,
                    )

                else:
                    xmin = min(predictions_df.values)
                    xmax = max(predictions_df.values)
                    ymin = min(ytest_tmp.values)
                    ymax = max(ytest_tmp.values)
                    min_ = min([xmin, ymin])
                    max_ = max([xmax, ymax])
                    fig.add_trace(
                        go.Scatter(x=[min_, max_], y=[min_, max_], name="x=y"), row=1, col=col + 1,
                    )
                    m, b = np.polyfit(predictions_df, ytest_tmp, 1)
                    fig.add_trace(
                        go.Scatter(
                            x=predictions_df, y=m * predictions_df + b, name="interpolation"
                        ),
                        row=1,
                        col=col + 1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=predictions_df,
                            y=ytest_tmp,
                            mode="markers",
                            marker=dict(size=4),
                            name="labels VS predictions",
                        ),
                        row=1,
                        col=col + 1,
                    )
                    fig.update_layout(
                        height=600, width=800, title_text=df_name + ": " + repres_name,
                    )
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")

            if path:
                if not path.endswith("/"):
                    path = path + "/"
                fig.write_html(f"{path}{df_name}-{repres_name}.html")
                print(f"Plot has been saved as {path}{df_name}-{repres_name}.html")
            else:
                fig.write_html(f"{df_name}-{repres_name}.html")
                print(f"Plot has been saved as {df_name}-{repres_name}.html")
            fig.show()
    elif model_type == "classification":
        for repres_name in representation_names:
            # copy probabilities
            y_score = predictions_test[repres_name].copy()
            # convert probabilities into classes
            if isinstance(metrics[repres_name]["prob_threshold"], list):  # multitasking case
                # TO DO  according to the task different probability threshold may have been selected!
                print(
                    "WARNING: prob_threshold for multitasking has not been optimized with GHOSTML so far. Default threshold of 0.5 will be set instead."
                )
                prob_threshold = 0.5
            else:  # single task
                prob_threshold = metrics[repres_name]["prob_threshold"]
            predictions_test[repres_name][predictions_test[repres_name] < prob_threshold] = 0
            predictions_test[repres_name][predictions_test[repres_name] >= prob_threshold] = 1
            y_score_only = pd.DataFrame(y_score)
            # check dimension consistency
            rows = y_score_only.shape[0]
            columns = y_score_only.shape[1]
            if rows < columns:
                y_score_only = y_score_only.T
            if not isinstance(test_labels[repres_name], pd.DataFrame):
                test_labels_df = pd.DataFrame(test_labels[repres_name].tolist())
            else:
                test_labels_df = test_labels[repres_name]
            y_test_only = test_labels_df
            fig = make_subplots(rows=len(test_labels_df.columns), cols=2)
            add_index = 1
            annot_all = []
            for index, col in enumerate(test_labels_df.columns):

                predictions_df = pd.DataFrame(predictions_test[repres_name]).loc[:, index]

                fpr = dict()
                tpr = dict()
                fpr, tpr, _ = roc_curve(
                    y_test_only.iloc[:, index].values[
                        y_test_only.iloc[:, index].values != mask_value
                    ],
                    y_score_only.iloc[:, index].values[
                        y_test_only.iloc[:, index].values != mask_value
                    ],
                )

                fig.add_trace(go.Scatter(x=fpr, y=tpr), row=index + 1, col=1)

                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]), row=index + 1, col=1)
                matrix = confusion_matrix(
                    y_test_only.iloc[:, index].values[
                        y_test_only.iloc[:, index].values != mask_value
                    ],
                    predictions_df[y_test_only.iloc[:, index].values != mask_value],
                )

                a = round(matrix[0][0], 2)
                b = round(matrix[0][1], 2)
                c = round(matrix[1][0], 2)
                d = round(matrix[1][1], 2)
                x = ["Pred_N", "Pred_P"]
                y = ["Real_N", "Real_P"]
                z = [[a, b], [c, d]]
                z_text = [[str(y) for y in x] for x in z]
                fig1 = ff.create_annotated_heatmap(
                    z, x=x, y=y, annotation_text=z_text, colorscale="Viridis"
                )

                # adjust margins to make room for yaxis title
                fig1.update_layout(margin=dict(t=50, l=200))

                fig.add_trace(fig1.data[0], row=index + 1, col=2)
                annot1 = list(fig1.layout.annotations)

                for k in range(len(annot1)):
                    annot1[k]["xref"] = "x" + str(index + add_index + 1)
                    annot1[k]["yref"] = "y" + str(index + add_index + 1)
                add_index = add_index + 1
                annot_all = annot_all + annot1

            fig.update_layout(title_text="<i><b> ROC curve  -  Confusion matrix</b></i>",)
            # set showlegend property by name of trace
            for trace in fig["data"]:
                if trace["name"] != "B":
                    trace["showlegend"] = False
        fig.layout.annotations = annot_all  # annotations

        if path:
            if not path.endswith("/"):
                path = path + "/"
            fig.write_html(f"{path}{df_name}-{repres_name}.html")
            print(f"Plot has been saved as {path}{df_name}-{repres_name}.html")
        else:
            fig.write_html(f"{df_name}-{repres_name}.html")
            print(f"Plot has been saved as {df_name}-{repres_name}.html")

    else:
        raise ValueError(
            f"{model_type} is not a valid name. You can only have Regression or Classification"
        )
