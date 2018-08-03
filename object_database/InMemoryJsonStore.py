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
            assert isinstance(s,set)
            for value in values:
                s.discard(json.dumps(value))

    def storedStringCount(self):
        return len([x for x in self.values.values() if isinstance(x,str)])

    def setMembers(self, key):
        with self.lock:
            s = self.values.get(key, None)
            if s is None:
                return []

            assert isinstance(s,set)
            return sorted(s)

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