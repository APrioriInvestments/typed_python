class FunctionMetadata:
    def __init__(self):
        self._constantReturnValue = ()

    def setConstantReturnValue(self, value):
        self._constantReturnValue = (value,)

    def hasConstantReturnValue(self):
        return self._constantReturnValue

    def getConstantReturnValue(self):
        return self._constantReturnValue[0] if self._constantReturnValue else None
