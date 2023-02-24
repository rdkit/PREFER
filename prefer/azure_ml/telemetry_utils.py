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


import logging
from functools import wraps
from typing import Optional
from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.trace import config_integration
from opencensus.ext.azure.trace_exporter import AzureExporter
from opencensus.trace.samplers import ProbabilitySampler
from opencensus.trace.tracer import Tracer
from azureml.core import Run
from azureml.exceptions import RunEnvironmentException


def _callback_add_cloudrole(envelope):
    envelope.tags["ai.cloud.role"] = "Azure ML"
    return True


class AppFilter(logging.Filter):
    def __init__(self, run_type: str):
        self.run_type = run_type

    def filter(self, record):
        custom_dimensions = {}
        _set_run_context(custom_dimensions, self.run_type)
        record.custom_dimensions = custom_dimensions
        return True


def set_telemetry_handlers(logger, run_type: str):
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    try:
        handler = AzureLogHandler()
        handler.add_telemetry_processor(_callback_add_cloudrole)
        handler.addFilter(AppFilter(run_type))
        logger.addHandler(handler)

        # configure tracing for app insights integrations
        config_integration.trace_integrations(["requests"])
        _get_tracer(run_type)
    except Exception:
        # application insights connection string is not set up in environment.
        # Probably this doesn't run in AML, skipping setting up Application Insights for local run.
        pass


def _set_run_context(dictionary: dict, run_type: str):
    try:
        run = Run.get_context(allow_offline=False)
        dictionary["parent_run_id"] = run.parent.id
        dictionary["step_id"] = run.id
        dictionary["step_name"] = run.name
        dictionary["experiment_name"] = run.experiment.name
        dictionary["run_url"] = run.parent.get_portal_url()
        dictionary["run_type"] = run_type
    except RunEnvironmentException:
        # Not an AzureML run
        pass


def _callback_add_context(run_type: str, envelope):
    _set_run_context(envelope.data.baseData.properties, run_type)


def _get_tracer(run_type: str):
    try:
        app_insights_exporter = AzureExporter()
        app_insights_exporter.add_telemetry_processor(_callback_add_cloudrole)
        app_insights_exporter.add_telemetry_processor(
            lambda envelope: _callback_add_context(run_type, envelope)
        )
        return Tracer(exporter=app_insights_exporter, sampler=ProbabilitySampler(rate=1.0))
    except Exception:
        # application insights connection string is not set up in environment.
        # Probably this doesn't run in AML, returning tracer which reports locally.
        return Tracer(sampler=ProbabilitySampler(rate=1.0))


def function_span_metrics_decorator(span_name: str, run_type: Optional[str]):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = _get_tracer(run_type)
            with tracer.span(name=span_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator
