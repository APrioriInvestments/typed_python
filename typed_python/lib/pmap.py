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
from typed_python import Class, Final, Member, ListOf, TypeFunction, Tuple, NotCompiled, Entrypoint, PointerTo
from typed_python.typed_queue import TypedQueue
import os

_threads = []


isJobExecutor = threading.local()


class Job(Class):
    def execute(self, i: int) -> None:
        pass


work_queue = TypedQueue(Tuple(Job, int))()


@TypeFunction
def ListJob(InputT, FuncT, OutT):
    class ListJob(Job, Final):
        OutputType = OutT

        outputQueue = Member(TypedQueue(int))
        exceptionQueue = Member(TypedQueue(Tuple(int, object)))
        inputPtr = Member(PointerTo(InputT))
        isInitializedPtr = Member(PointerTo(bool))
        outputPtr = Member(PointerTo(OutT))
        jobGranularity = Member(int)
        maxIndex = Member(int)
        f = Member(FuncT)

        def __init__(self, inputPtr, f, outputPtr, isInitializedPtr, jobGranularity, maxIndex):
            self.inputPtr = inputPtr
            self.outputPtr = outputPtr
            self.isInitializedPtr = isInitializedPtr
            self.outputQueue = TypedQueue(int)()
            self.exceptionQueue = TypedQueue(Tuple(int, object))()
            self.jobGranularity = jobGranularity
            self.maxIndex = maxIndex
            self.f = f

        def execute(self, i: int) -> None:
            try:
                for jobIx in range(i * self.jobGranularity, min(self.maxIndex, (i + 1) * self.jobGranularity)):
                    (self.outputPtr + jobIx).initialize(self.f(self.inputPtr[jobIx]))
                    self.isInitializedPtr[jobIx] = True
            except Exception as e:
                self.exceptionQueue.put(Tuple(int, object)((jobIx, e)))

            self.outputQueue.put(i)

    return ListJob


@NotCompiled
def isExecutorThread() -> bool:
    return getattr(isJobExecutor, 'isExecutor', False)


@Entrypoint
def workExecutor():
    isJobExecutor.isExecutor = True

    while True:
        Job, ix = work_queue.get()

        if ix >= 0:
            Job.execute(ix)
        else:
            return

        Job = None


_maxPmapThreads = [1000]


def getMaxPmapThreads():
    return _maxPmapThreads[0]


def setMaxPmapThreads(count):
    assert count > 0
    _maxPmapThreads[0] = count


@NotCompiled
def ensureThreads():
    if not _threads:
        for i in range(min(_maxPmapThreads[0], os.cpu_count())):
            _threads.append(threading.Thread(target=workExecutor, daemon=True))
            _threads[-1].start()


@Entrypoint
def pmap(lst, f, OutT, minGranularity=1):
    """Apply 'f' to every element of 'lst' in parallel.

    Args:
        lst - a ListOf of some type
        f - a function from lst.ElementType to OutT
        OutT - the result type
        jobGranularity - how many items we should dispatch at once.
            If you have very small tasks, you'll spend far more time
            locking and unlocking the queue than you will actually
            doing work. If None, this will pick something that tries to
            avoid creating too many jobs
        minGranularity - the smallest batch size we'll allow.
            If this is 1, then each item in the list is a job. If
            greater than 1, then we will do no fewer than this many
            jobs per thread dispatch.
    """
    ensureThreads()

    jobGranularity = max(1, len(lst) // (int(os.cpu_count()) * 30), minGranularity)

    # make a list of objects but don't initialize any of them.
    # some objects don't have default constructors and we want to
    # still be able to pmap them.
    res = ListOf(OutT)()
    res.reserve(len(lst))

    # track which objects are initialized
    isInitialized = ListOf(bool)()
    isInitialized.resize(len(lst))

    # create the 'job'
    job = ListJob(lst.ElementType, type(f), OutT)(
        lst.pointerUnsafe(0),
        f,
        res.pointerUnsafe(0),
        isInitialized.pointerUnsafe(0),
        jobGranularity,
        len(lst)
    )

    # fill out the work queue
    jobCount = len(lst) // jobGranularity

    if jobCount * jobGranularity < len(lst):
        jobCount += 1

    for i in range(jobCount):
        tup = Tuple(Job, int)((job, i))
        work_queue.put(tup)

    # block until the work queue is complete
    # if we're an executor thread, then we are part of a
    # 'recursive' pmap call, and we need to do work!
    if isExecutorThread():
        while True:
            jobAndIndex = work_queue.getNonblocking()
            if jobAndIndex is None:
                # it's OK to block now because if
                # there are no outstanding jobs, then
                # all of _our_ jobs have been picked up
                job.outputQueue.getMany(jobCount, jobCount)
                break
            else:
                j = jobAndIndex[0]
                ix = jobAndIndex[1]

                if ix >= 0:
                    j.execute(ix)
    else:
        # we can just block on the queue
        job.outputQueue.getMany(jobCount, jobCount)

    # check if any of our threads excepted, and if so
    # raise the earliest one in the sequence.
    exceptionObj = None
    minI = len(lst)

    while job.exceptionQueue:
        i, eo = job.exceptionQueue.get()
        if i < minI:
            minI = i
            exceptionObj = eo

    if exceptionObj is not None:
        # if we're raising, we need to clean up our
        # temporary storage
        for i in range(len(lst)):
            if isInitialized[i]:
                res.pointerUnsafe(i).destroy()

        raise exceptionObj

    res.setSizeUnsafe(len(lst))

    return res
