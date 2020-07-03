#   Copyright 2020 typed_python Authors
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


class DirectedGraph:
    """A simple class to model a directed graph.

    Nodes can be anything hashable.
    """

    def __init__(self):
        self.sourceToDest = {}
        self.destToSource = {}

    def addEdge(self, source, dest):
        if dest not in self.sourceToDest.setdefault(source, set()):
            self.sourceToDest[source].add(dest)
            self.destToSource.setdefault(dest, set()).add(source)

    def dropEdge(self, source, dest):
        if not self.hasEdge(source, dest):
            raise Exception(f"No edge between {source} and {dest}")
        self.discardEdge(source, dest)

    def discardEdge(self, source, dest):
        if source in self.sourceToDest:
            self.sourceToDest[source].discard(dest)
        if dest in self.destToSource:
            self.destToSource[dest].discard(source)

    def hasEdge(self, source, dest):
        if source not in self.sourceToDest:
            return False
        return dest in self.sourceToDest[source]

    def outgoing(self, node):
        return self.sourceToDest.get(node, set())

    def incoming(self, node):
        return self.destToSource.get(node, set())

    def dropNode(self, node, transitively):
        """Remove a node.  If the node has no edges, it's a no-op.

        if 'transitively', then find all the nodes we're connected to and
        connect them.
        """
        if transitively:
            for i in self.incoming(node):
                for o in self.outgoing(node):
                    self.addEdge(i, o)

        for i in list(self.incoming(node)):
            self.discardEdge(i, node)

        for o in list(self.outgoing(node)):
            self.discardEdge(node, o)

    def levels(self, reversed=False):
        """ Return a map of node to level (int>=0).

            Args:
                reversed (bool): when False, source nodes are level=0,
                    when True, sink nodes are level=0
        """
        levels = {}

        if reversed:
            children = lambda n: self.outgoing(n)
        else:
            children = lambda n: self.incoming(n)

        def walk(n):
            if n not in levels:
                levels[n] = -2
                levels[n] = 1 + max((walk(child) for child in children(n)), default=-1)

            if levels[n] == -2:
                raise Exception(f"Cycle detected at node {n}")

            return levels[n]

        for n in self.sourceToDest:
            walk(n)
        for n in self.destToSource:
            walk(n)

        return levels

    def topologicalSort(self, nodes, reversed=False):
        levels = self.levels(reversed=reversed)
        indices = {n: i for i, n in enumerate(nodes)}

        return sorted(nodes, key=lambda n: (levels.get(n, 0), indices[n]))

    def createsCycle(self, source, dest):
        """would adding 'source->dest' node make the graph cyclic?"""
        if source == dest:
            return False

        downstreamSet = set()

        def walk(x):
            if x in downstreamSet:
                return

            downstreamSet.add(x)

            for x in self.outgoing(x):
                walk(x)

        walk(dest)

        return source in downstreamSet
