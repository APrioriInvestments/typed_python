#   Copyright 2018 Braxton Mckee
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

import threading
import json

class InMemoryJsonStore(object):
    """Implements a string-to-json store in memory.

    Keys must be strings. Values may be anything that's json-compatible.

    This class is thread-safe.
    """
    def __init__(self, db=0):
        self.values = {}
        self.lock = threading.RLock()

    def get(self, key):
        with self.lock:
            if key not in self.values:
                return None
            val = self.values.get(key)

            assert isinstance(val, str), key

            return json.loads(val)

    def set(self, key, value):
        with self.lock:
            if value is None:
                if key in self.values:
                    del self.values[key]
            else:
                self.values[key] = json.dumps(value)

    def setAdd(self, key, values):
        with self.lock:
            if key not in self.values:
                self.values[key] = set()
            for value in values:
                self.values[key].add(json.dumps(value))

    def setRemove(self, key, values):
        with self.lock:
            s = self.values.get(key,None)
            assert isinstance(s,set), (key,s)
            for value in values:
                jv = json.dumps(value)
                assert jv in s, (jv, s)
                s.remove(json.dumps(value))

    def storedStringCount(self):
        return len([x for x in self.values.values() if isinstance(x,str)])

    def setMembers(self, key):
        with self.lock:
            s = self.values.get(key, None)
            if s is None:
                return []

            assert isinstance(s,set)

            return sorted([json.loads(x) for x in s])

    def listKeys(self, prefix, suffix):
        with self.lock:
            res = []
            for k in self.keys:
                if k.startswith(prefix) and k.endswith(suffix) and len(k) >= len(prefix) + len(suffix):
                    res.append(k)
            return sorted(res)

    def getSeveral(self, keys):
        with self.lock:
            return [self.get(k) for k in keys]

    def setSeveral(self, kvs, adds=None, removes=None):
        with self.lock:
            for k in (adds or []):
                assert not isinstance(self.values.get(k, None), str), k + " is already a string"
            for k in (removes or []):
                assert not isinstance(self.values.get(k, None), str), k + " is already a string"

            for k,v in kvs.items():
                self.set(k,v)

            if adds:
                for k,to_add in adds.items():
                    self.setAdd(k, to_add)

            if removes:
                for k,to_remove in removes.items():
                    self.setRemove(k, to_remove)

    def exists(self, key):
        with self.lock:
            return key in self.values

    def delete(self, key):
        with self.lock:
            if key in self.values:
                del self.values[key]

    def clearCache(self):
        pass