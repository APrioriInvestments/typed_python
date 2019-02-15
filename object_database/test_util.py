#   Copyright 2019 Nativepython Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.from collections import defaultdict

import logging
import os
import psutil
import subprocess
import sys
import tempfile

from object_database.util import genToken


def currentMemUsageMb(residentOnly=True):
    if residentOnly:
        return psutil.Process().memory_info().rss / 1024 ** 2
    else:
        return psutil.Process().memory_info().vms / 1024 ** 2


def gc_object_types_histogram():
    """ Returns a map of types to the cound of how many such objects the GC is managing.

        Return Type: defaultdict( ObjectType: type -> count: int )
    """
    dd = defaultdict(int)
    gc_objects = gc.get_objects()
    for o in gc_objects:
        dd[type(o)] += 1

    total = sum([v for v in dd.values()])
    assert total == len(gc_objects), (total, len(gc_objects))

    return dd


def diff_object_types_histograms(new_histo, old_histo):
    """ Returns a new histogram that is the difference of it inputs """
    all_keys = set(new_histo.keys()).union(old_histo.keys())

    dd = {k: new_histo[k] - old_histo[k]
        for k in all_keys if new_histo[k] - old_histo[k] != 0
    }
    return dd


def sort_by_value(histogram, topK=None, filterFn=None):
    """ Return a sorted list of (value, tag) pairs from a given histogram.

        If filter is specified the results are filtered using that function
        If topK is specified, only return topK results
    """
    res = reversed(sorted(
        [(val, tag) for tag, val in histogram.items()],
        key=lambda pair: pair[0]
    ))

    if filter is not None:
        res = filter(filterFn, res)

    if topK is not None:
        return list(res)[:topK]
    else:
        return list(res)


def log_cells_stats(cells, logger, indentation=0):
    indent = " " * indentation
    def log(msg):
        logger(indent + msg)

    log("#####################################################")
    log("#  Cells structure DEBUG Log")
    log("#  - dirty: {}"
        .format(len(cells._dirtyNodes)))
    log("#  - need bcast: {}"
        .format(len(cells._nodesToBroadcast)))
    log("#  - cells: {}"
        .format(len(cells._cells)))
    log("#  - known children: {}"
        .format(len(cells._cellsKnownChildren)))
    log("#  - to discard: {}"
        .format(len(cells._nodesToDiscard)))
    log("#  - subscribed-to keys: {}"
        .format(len(cells._subscribedCells)))
    log("#####################################################")


ownDir = os.path.dirname(os.path.abspath(__file__))

def start_service_manager(tempDirectoryName, port, auth_token, loglevel_name="INFO", timeout=1.0,
                          verbose=True, own_hostname='localhost', db_hostname='localhost'):
    if not verbose:
        kwargs = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        kwargs = dict()

    server = subprocess.Popen(
        [sys.executable, os.path.join(ownDir, 'frontends', 'service_manager.py'),
            own_hostname, db_hostname, str(port), '--run_db',
            '--source', os.path.join(tempDirectoryName,'source'),
            '--storage', os.path.join(tempDirectoryName,'storage'),
            '--service-token', auth_token,
            '--shutdownTimeout', str(timeout / 2.0),
            '--log-level', loglevel_name
        ],
        **kwargs
    )
    try:
        # this should throw a subprocess.TimeoutExpired exception if the service did not crash
        server.wait(timeout)
    except subprocess.TimeoutExpired:
        pass
    else:
        raise Exception("Failed to start service_manager (retcode:{})"
            .format(server.returncode)
        )
    return server


def autoconfigure_and_start_service_manager(port=None, auth_token=None, loglevel_name=None, **kwargs):
    port = port or 8020
    auth_token = auth_token or genToken()

    if loglevel_name is None:
        loglevel = logging.getLogger(__name__).getEffectiveLevel()
        loglevel_name = logging.getLevelName(loglevel)

    tempDirObj = tempfile.TemporaryDirectory()
    tempDirectoryName = tempDirObj.name

    server = start_service_manager(
        tempDirectoryName,
        port,
        auth_token,
        loglevel_name,
        **kwargs
    )

    def cleanupFn(error=False):
        server.terminate()
        server.wait()
        if error:
            logging.getLogger(__name__).warning(
                "Exited with an error. Leaving temporary directory around for inspection: {}"
                .format(tempDirectoryName)
            )
        else:
            tempDirObj.cleanup()

    return server, cleanupFn
