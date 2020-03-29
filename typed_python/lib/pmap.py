#   Copyright 2017-2020 typed_python Authors
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
#   limitations under the License.

"""
Utilities for running operations in parallel.

We require operations to be compilable for this to work.
"""

import threading
from typed_python import Class, Final, Member, ListOf, TypeFunction, Tuple, NotCompiled, Entrypoint
from typed_python.typed_queue import TypedQueue
import os

_threads = []


class Job(Class):
    def execute(self, i: int) -> None:
        pass


work_queue = TypedQueue(Tuple(Job, int))()


@TypeFunction
def ListJob(ListT, FuncT, OutT):
    class ListJob(Job, Final):
        OutputType = OutT

        outputQueue = Member(TypedQueue(Tuple(OutT, int)))
        inputList = Member(ListT)
        f = Member(FuncT)

        def __init__(self, inputList, f):
            self.inputList = inputList
            self.f = f
            self.outputQueue = TypedQueue(Tuple(OutT, int))()

        def execute(self, i: int) -> None:
            output = Tuple(OutT, int)((self.f(self.inputList[i]), i))
            self.outputQueue.put(output)

    return ListJob


@Entrypoint
def workExecutor():
    while True:
        Job, ix = work_queue.get()

        if ix >= 0:
            Job.execute(ix)
        else:
            return


@NotCompiled
def ensureThreads():
    if not _threads:
        for i in range(os.cpu_count()):
            _threads.append(threading.Thread(target=workExecutor, daemon=True))
            _threads[-1].start()


@Entrypoint
def pmap(lst, f, OutT):
    ensureThreads()

    job = ListJob(type(lst), type(f), OutT)(lst, f)

    for i in range(len(lst)):
        tup = Tuple(Job, int)((job, i))
        work_queue.put(tup)

    results = job.outputQueue.getMany(len(lst), len(lst))

    res = ListOf(type(job).OutputType)()
    res.resize(len(results))

    for resTup in results:
        res[resTup[1]] = resTup[0]

    return res
