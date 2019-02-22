"""A simple 'codebase' for testing purposes"""

from object_database.service_manager.ServiceBase import ServiceBase
from object_database.service_manager.Task import TaskExecutor, RunningTask, TaskStatusResult
from object_database import Schema

import time

schema = Schema("TestModule1")

@schema.define
class Record:
    x = int

def createNewRecord(db):
    db.subscribeToNone(Record)
    with db.transaction():
        Record(x=10)

class RunningTaskWithSubtasks(RunningTask):
    """A slow, simple task that runs for 1/20th of a second, and that fires off some subtasks.

    We use this for testing a task graph with dependencies.
    """
    def __init__(self, x):
        self.x = x

    def execute(self, taskContext, subtaskResults):
        if subtaskResults is None:
            time.sleep(0.05)

            if self.x <= 0:
                return TaskStatusResult.Finished(result=1)

            return TaskStatusResult.Subtasks({'A': TaskWithSubtasks(self.x-1), 'B': TaskWithSubtasks(self.x-2)})
        else:
            return TaskStatusResult.Finished(subtaskResults['A'].result + subtaskResults['B'].result)

class TaskWithSubtasks(TaskExecutor):
    def __init__(self, x):
        self.x = x

    def instantiate(self):
        return RunningTaskWithSubtasks(self.x)

class TestService1(ServiceBase):
    def initialize(self):
        pass

    def doWork(self, shouldStop):
        while not shouldStop.is_set():
            time.sleep(0.01)
