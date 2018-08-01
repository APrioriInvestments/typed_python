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

            return json.loads(self.values[key])

    def set(self, key, value):
        with self.lock:
            if value is None:
                if key in self.values:
                    del self.values[key]
            else:
                self.values[key] = json.dumps(value)

    def setSeveral(self, kvs):
        with self.lock:
            for k,v in kvs.items():
                self.set(k,v)

    def exists(self, key):
        with self.lock:
            return key in self.values

    def delete(self, key):
        with self.lock:
            if key in self.values:
                del self.values[key]

    def clearCache(self):
        pass