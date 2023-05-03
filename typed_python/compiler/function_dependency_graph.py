from typed_python.compiler.directed_graph import DirectedGraph
from sortedcontainers import SortedSet


class FunctionDependencyGraph:
    def __init__(self):
        self._dependencies = DirectedGraph()

        # the search depth in the dependency to find 'identity'
        # the _first_ time we ever saw it. We prefer to update
        # nodes with higher search depth, so we don't recompute
        # earlier nodes until their children are complete.
        self._identity_levels = {}

        # nodes that need to recompute
        self._dirty_inflight_functions = set()

        # (priority, node) pairs that need to recompute
        self._dirty_inflight_functions_with_order = SortedSet(key=lambda pair: pair[0])

    def dropNode(self, node):
        self._dependencies.dropNode(node, False)
        if node in self._identity_levels:
            del self._identity_levels[node]
        self._dirty_inflight_functions.discard(node)

    def getNextDirtyNode(self):
        while self._dirty_inflight_functions_with_order:
            priority, identity = self._dirty_inflight_functions_with_order.pop()

            if identity in self._dirty_inflight_functions:
                self._dirty_inflight_functions.discard(identity)

                return identity

    def addRoot(self, identity):
        if identity not in self._identity_levels:
            self._identity_levels[identity] = 0
            self.markDirty(identity)

    def addEdge(self, caller, callee):
        if caller not in self._identity_levels:
            raise Exception(f"unknown identity {caller} found in the graph")

        if callee not in self._identity_levels:
            self._identity_levels[callee] = self._identity_levels[caller] + 1

            self.markDirty(callee, isNew=True)

        self._dependencies.addEdge(caller, callee)

    def getNamesDependedOn(self, caller):
        return self._dependencies.outgoing(caller)

    def markDirtyWithLowPriority(self, callee):
        # mark this dirty, but call it back after new functions.
        self._dirty_inflight_functions.add(callee)

        level = self._identity_levels[callee]
        self._dirty_inflight_functions_with_order.add((-1000000 + level, callee))

    def markDirty(self, callee, isNew=False):
        self._dirty_inflight_functions.add(callee)

        if isNew:
            # if its a new node, compute it with higher priority the _higher_ it is in the stack
            # so that we do a depth-first search on the way down
            level = 1000000 - self._identity_levels[callee]
        else:
            level = self._identity_levels[callee]

        self._dirty_inflight_functions_with_order.add((level, callee))

    def functionReturnSignatureChanged(self, identity):
        for caller in self._dependencies.incoming(identity):
            self.markDirty(caller)
