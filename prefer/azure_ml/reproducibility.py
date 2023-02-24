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


import subprocess
from hashlib import sha256
from time import time
import logging

from pathlib import Path

from pyreporoot import project_root
from semver import VersionInfo

logger = logging.getLogger(__name__)

GenChemVersion = VersionInfo


def get_current_api_version() -> GenChemVersion:
    """Returns the current version of the APIs."""
    with open(str(project_root(Path(__file__)).joinpath("api_version.txt")), "rt") as f:
        return GenChemVersion.parse(f.read().rstrip())


def timeit(fn):
    """
    *args and **kwargs are to support positional and named arguments of fn
    Use this as a decorator for the function you wish to time
        @timeit
        def my_func(args):
            ....
            return

        This produces output of the form "Time taken in my_func: 1.11111111s". The time is returned in seconds.
    """

    def get_time(*args, **kwargs):
        start = time()
        output = fn(*args, **kwargs)
        logger.info(f"Time taken in {fn.__name__}: {time() - start:.7f}s")
        return output  # make sure that the decorator returns the output of fn

    return get_time


def file_hash(filename: str) -> str:
    """
    Calculate SHA256 hash of a file
    """
    sha256_hash = sha256()
    with open(filename, "rb") as f:
        # Read and update hash string value in blocks of 16K
        for byte_block in iter(lambda: f.read(16384), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_git_short_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ASCII").strip()


def get_git_status() -> str:
    """
    Queries the git tree for current git revision hash, status of local changes, status of untracked files
    Output is of form:
        FULL_GIT_HASH@BRANCH_NAME+local_changes+untracked_files
        The git hash and branch name are always returned.
        '+local_changes' tag is added if any local changes are found
        '+untracked_files' tag is added if any untracked files are found
    Returns:
        (str) String describing the status of the git tree
    """
    try:

        def run_command(command):
            return (
                subprocess.check_output(command, cwd=project_root(Path(__file__)))
                .decode("ASCII")
                .strip()
            )

        # pylint: disable=unexpected-keyword-arg # For some reason, pylint doesn't like "cwd"
        head_ref_names = run_command(["git", "log", "--format=%D", "-1"])
        # head_ref_names returns output of the form
        # HEAD -> user_branch_name, origin/master, origin/HEAD, master
        # Parse this to recover the branch name if possible, else leave empty
        if " -> " in head_ref_names:
            branch = "@" + head_ref_names.split(" -> ")[1].split(",")[0]
        else:
            branch = ""

        # Recover the change hash, keep the full version
        change_hash: str = run_command(["git", "rev-parse", "HEAD"])

        # Identify if there are any uncommitted local changes
        local_changes: str = run_command(["git", "diff-index", "HEAD", "--"])
        change_status = "" if local_changes == "" else "+local_changes"

        # Identify if there are any untracked local changes
        untracked_files: str = run_command(["git", "status", "--short"])
        untracked_status = "" if untracked_files == "" else "+untracked_files"

        # This ony works when the remote is called origin, but we can't guarantee that
        # Find out if there is a way to query the name of the current remote tree
        # unpushed_changes: str = run_command(["git", "log", "origin.."])
        # unpushed_status = "" if unpushed_changes == "" else "+unpushed_changes"

        return "{}{}{}{}".format(change_hash, branch, change_status, untracked_status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "UNKNOWN GIT REVISION"
