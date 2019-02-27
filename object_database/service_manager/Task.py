#   Copyright 2019 Braxton Mckee
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

import logging
import traceback
import threading
import time

from object_database.web import cells as cells
from object_database.service_manager.ServiceSchema import service_schema
from object_database.service_manager.ServiceBase import ServiceBase
from object_database import Schema, Indexed, Index, core_schema
from object_database.view import revisionConflictRetry, DisconnectedException
from typed_python import OneOf, Alternative, ConstDict

task_schema = Schema("core.task")

# how many times our worker can disconnect in a row before we get marked 'Failed'
MAX_TIMES_FAILED = 10


@task_schema.define
class ResourceScope:
    pass


class TaskContext(object):
    """Placeholder for information about the current running task environment passed into tasks."""

    def __init__(self, db, storageRoot, codebase):
        self.db = db
        self.storageRoot = storageRoot
        self.codebase = codebase


class RunningTask(object):
    """Base class for a running Task's state. This must be serializable."""

    def __init__(self):
        pass

    def execute(self, taskContext, subtaskResults):
        """Step the task forward. Should return a TaskStatusResult. If we asked for results, they are passed back in subtaskResults"""
        raise NotImplementedError()


class TaskExecutor(object):
    """Base class for all Tasks. """

    def instantiate(self):
        """Return a RunningTask that represents us."""
        raise NotImplementedError()


class RunningFunctionTask(RunningTask):
    def __init__(self, f):
        self.f = f

    def execute(self, taskContext, subtaskResults):
        return TaskStatusResult.Finished(result=self.f(taskContext.db))


class FunctionTask(TaskExecutor):
    """A simple task that just runs a single function."""

    def __init__(self, f):
        self.f = f

    def instantiate(self):
        return RunningFunctionTask(self.f)


TaskStatusResult = Alternative(
    'TaskStatusResult',
    Finished={'result': object},
    Subtasks={'subtasks': ConstDict(str, TaskExecutor)},
    SleepUntil={'wakeup_timestamp': float}
)

TaskResult = Alternative(
    "TaskResult",
    Result={'result': object},
    Error={'error': str},
    Failure={}
)


@task_schema.define
class Task:
    service = Indexed(service_schema.Service)
    service_and_finished = Index('service', 'finished')

    resourceScope = Indexed(OneOf(None, ResourceScope))
    executor = TaskExecutor
    parent = OneOf(None, task_schema.Task)
    instance = OneOf(None, object)

    time_elapsed = float

    result = OneOf(None, TaskResult)

    finished = Indexed(OneOf(None, True))
    finished_timestamp = OneOf(None, float)

    @staticmethod
    def Create(service, executor):
        return TaskStatus(
            task=Task(
                service=service,
                executor=executor
            ),
            state="Unassigned"
        ).task


@task_schema.define
class TaskStatus:
    task = Indexed(Task)
    parentStatus = OneOf(None, task_schema.TaskStatus)
    resourceScope = Indexed(OneOf(None, ResourceScope))
    state = Indexed(OneOf("Unassigned", "Assigned", "Working", "Sleeping", "WaitForSubtasks", "DoneCalculating", "Collected"))
    wakeup_timestamp = OneOf(None, float)
    subtasks = OneOf(None, ConstDict(str, task_schema.TaskStatus))
    subtasks_completed = int
    times_failed = int
    worker = Indexed(OneOf(None, task_schema.TaskWorker))

    @revisionConflictRetry
    def finish(self, db, result, elapsed=0.0):
        with db.transaction():
            self._finish(result, elapsed)

    def _finish(self, result, elapsed=0.0):
        self.task.result = result
        self.task.time_elapsed += elapsed
        self.worker = None
        self.state = "DoneCalculating"


@task_schema.define
class TaskWorker:
    connection = Indexed(core_schema.Connection)
    hasTask = Indexed(bool)


class TaskService(ServiceBase):
    coresUsed = 1
    gbRamUsed = 8

    def initialize(self):
        self.db.subscribeToNone(TaskWorker)
        self.logger = logging.getLogger(__name__)

        with self.db.transaction():
            self.workerObject = TaskWorker(connection=self.db.connectionObject, hasTask=False)

        self.db.subscribeToIndex(task_schema.TaskStatus, worker=self.workerObject)

    def doWork(self, shouldStop):
        while not shouldStop.is_set():
            with self.db.view():
                tasks = TaskStatus.lookupAll(worker=self.workerObject)

            if not tasks:
                time.sleep(.01)
            else:
                if len(tasks) > 1:
                    raise Exception("Expected only one task to be allocated to us.")

                self.doTask(tasks[0])

                with self.db.transaction():
                    self.workerObject.hasTask = False

    @staticmethod
    def serviceDisplay(serviceObject, instance=None, objType=None, queryArgs=None):
        cells.ensureSubscribedType(TaskStatus, lazy=True)

        return cells.Card(
            cells.Subscribed(lambda: cells.Text("Total Tasks: %s" % len(TaskStatus.lookupAll()))) +
            cells.Subscribed(lambda: cells.Text("Working Tasks: %s" % len(TaskStatus.lookupAll(state='Working')))) +
            cells.Subscribed(lambda: cells.Text("WaitingForSubtasks Tasks: %s" % len(TaskStatus.lookupAll(state='WaitForSubtasks')))) +
            cells.Subscribed(lambda: cells.Text("Unassigned Tasks: %s" % len(TaskStatus.lookupAll(state='Unassigned'))))
        )

    def doTask(self, taskStatus):
        with self.db.view():
            task = taskStatus.task

        self.db.subscribeToObject(task)

        t0 = None

        try:
            with self.db.transaction():
                assert taskStatus.state == "Assigned", taskStatus.state
                taskStatus.state = "Working"
                task = taskStatus.task
                codebase = taskStatus.task.service.codebase
                subtaskStatuses = taskStatus.subtasks
                taskStatus.subtasks = {}
                taskStatus.wakeup_timestamp = None

                typedPythonCodebase = codebase.instantiate()

            self.db.setSerializationContext(typedPythonCodebase.serializationContext)

            if subtaskStatuses:
                self.db.subscribeToObjects(list(subtaskStatuses.values()))

                with self.db.view():
                    subtasks = {name: status.task for name, status in subtaskStatuses.items()}

                self.db.subscribeToObjects(list(subtasks.values()))

                with self.db.transaction():
                    for r in subtaskStatuses.values():
                        assert r.state == "Collected"

                    subtask_results = {name: t.result for name, t in subtasks.items()}
                    for t in subtasks.values():
                        t.delete()
                    for s in subtaskStatuses.values():
                        logging.info("Deleting subtask status %s", s)
                        s.delete()
            else:
                subtask_results = None

            with self.db.transaction():
                executor = taskStatus.task.executor
                instanceState = taskStatus.task.instance

            if instanceState is None:
                instanceState = executor.instantiate()

            t0 = time.time()
            context = TaskContext(self.db, self.runtimeConfig.serviceTemporaryStorageRoot, codebase)
            execResult = instanceState.execute(context, subtask_results)
            logging.info("Executed task %s with state %s producing result %s", task, instanceState, execResult)

            assert isinstance(execResult, TaskStatusResult), execResult

            if execResult.matches.Finished:
                taskStatus.finish(self.db, TaskResult.Result(result=execResult.result), time.time() - t0)

            if execResult.matches.Subtasks:
                with self.db.transaction():
                    taskStatus.state = "WaitForSubtasks"
                    taskStatus.worker = None

                    # create the new child tasks
                    newTaskStatuses = {}

                    for taskName, subtaskExecutor in execResult.subtasks.items():
                        newTaskStatuses[taskName] = TaskStatus(
                            task=Task(
                                service=task.service,
                                resourceScope=task.resourceScope,
                                executor=subtaskExecutor,
                                parent=task
                            ),
                            parentStatus=taskStatus,
                            resourceScope=task.resourceScope,
                            state="Unassigned",
                            worker=None
                        )

                    logging.info("Subtask %s depends on %s", task, [str(ts.task) + "/" + str(ts) for ts in newTaskStatuses.values()])

                    taskStatus.subtasks = newTaskStatuses

            if execResult.matches.SleepUntil:
                with self.db.transaction():
                    taskStatus.state = "Sleeping"
                    taskStatus.worker = None
                    taskStatus.wakeup_timestamp = execResult.wakeup_timestamp

        except Exception:
            self.logger.error("Task %s failed with exception:\n%s", task, traceback.format_exc())
            taskStatus.finish(self.db, TaskResult.Error(error=traceback.format_exc()), time.time() - t0 if t0 is not None else 0.0)


class TaskDispatchService(ServiceBase):
    coresUsed = 1
    gbRamUsed = 4

    def initialize(self):
        self.logger = logging.getLogger(__name__)
        self.db.subscribeToType(task_schema.TaskStatus)
        self.db.subscribeToType(task_schema.TaskWorker)

    def checkForDeadWorkers(self):
        toDelete = []
        with self.db.view():
            for w in TaskWorker.lookupAll():
                if not w.connection.exists():
                    toDelete.append(w)

        @revisionConflictRetry
        def deleteWorker(w):
            while True:
                with self.db.view():
                    taskStatuses = TaskStatus.lookupAll(worker=w)
                    tasks = [ts.task for ts in taskStatuses]

                if not taskStatuses:
                    return

                self.db.subscribeToObjects(tasks)

                with self.db.transaction():
                    for taskStatus in taskStatuses:
                        taskStatus.times_failed += 1
                        if taskStatus.times_failed > MAX_TIMES_FAILED:
                            taskStatus._finish(TaskResult.Failure(), 0.0)
                        else:
                            taskStatus.state = "Unassigned"
                            taskStatus.worker = None

        for d in toDelete:
            deleteWorker(d)

    def doWork(self, shouldStop):
        def checkForDeadWorkersLoop():
            while not shouldStop.is_set():
                try:
                    self.checkForDeadWorkers()
                    time.sleep(5.0)
                except DisconnectedException:
                    return
                except Exception:
                    self.logger.error("Unexpected exception in TaskDispatchService: %s", traceback.format_exc())

        def assignLoop():
            while not shouldStop.is_set():
                try:
                    if self.assignWork():
                        shouldStop.wait(timeout=.01)
                except DisconnectedException:
                    return
                except Exception:
                    self.logger.error("Unexpected exception in TaskDispatchService: %s", traceback.format_exc())

        def collectLoop():
            while not shouldStop.is_set():
                try:
                    if self.collectResults():
                        shouldStop.wait(timeout=.01)
                except DisconnectedException:
                    return
                except Exception:
                    self.logger.error("Unexpected exception in TaskDispatchService: %s", traceback.format_exc())

        threads = [threading.Thread(target=t) for t in [checkForDeadWorkersLoop, assignLoop, collectLoop]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    @revisionConflictRetry
    def assignWork(self):
        count = 0
        with self.db.view():
            workers = list(TaskWorker.lookupAll(hasTask=False))
            tasks = list(TaskStatus.lookupAll(state='Unassigned'))

        with self.db.transaction():
            while workers and tasks:
                worker = workers.pop(0)
                task = tasks.pop(0)

                worker.hasTask = True

                task.worker = worker
                task.state = "Assigned"

                count += 1
        return count

    def collectResults(self, maxPerTransaction=30):
        count = 0
        with self.db.view():
            statuses = list(TaskStatus.lookupAll(state="DoneCalculating"))[:maxPerTransaction]

        while count < 30 and statuses:
            self.collectTask(statuses.pop(0))
            count += 1

        return count

    @revisionConflictRetry
    def collectTask(self, taskStatus):
        with self.db.view():
            task = taskStatus.task
            state = taskStatus.state
            parentStatus = taskStatus.parentStatus

        if state == "DoneCalculating" and parentStatus is None:
            self.db.subscribeToObject(task)

            with self.db.transaction():
                # this is a root-level task. Mark it complete so it can be collected
                # by whoever kicked it off.
                task.finished = True
                task.finished_timestamp = time.time()
                logging.info("deleting root status %s", taskStatus)
                taskStatus.delete()

        elif state == "DoneCalculating":
            with self.db.transaction():
                parentStatus.subtasks_completed = parentStatus.subtasks_completed + 1

                if len(parentStatus.subtasks) == parentStatus.subtasks_completed:
                    parentStatus.state = "Unassigned"
                    parentStatus.times_failed = 0

                taskStatus.state = "Collected"
