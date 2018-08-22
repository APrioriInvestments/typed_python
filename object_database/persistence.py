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

import redis
import json
import time
import threading
import logging
import os

class InMemoryStringStore(object):
    """Implements a string-to-string store in memory.

    Keys must be strings. Values are also strings or sets of strings.

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

            return val

    def set(self, key, value):
        with self.lock:
            if value is None:
                if key in self.values:
                    del self.values[key]
            else:
                self.values[key] = value

    def setAdd(self, key, values):
        with self.lock:
            if key not in self.values:
                self.values[key] = set()
            for value in values:
                self.values[key].add(value)

    def setRemove(self, key, values):
        with self.lock:
            s = self.values.get(key,None)
            assert isinstance(s,set), (key,s)
            for value in values:
                assert value in s, (jv, s)
                s.remove(value)

    def storedStringCount(self):
        return len([x for x in self.values.values() if isinstance(x,str)])

    def setMembers(self, key):
        with self.lock:
            s = self.values.get(key, None)
            if s is None:
                return []

            assert isinstance(s,set)

            return sorted([x for x in s])

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


class RedisStringStore(object):
    """Implements a string-to-store store using Redis.

    Keys must be strings. Values are strings, or None.

    You may store values using the "set" and "get" methods. 

    You may store sets of values using the setAdd, setRemove, setMembers, methods.

    You may not use a value-style method on a set-style key or vice versa.

    Setting a key to None deletes it, and non-existent keys are implicity 'None'

    You may delete any kind of value.

    This class is thread-safe.
    """
    def __init__(self, db=0, port=None):
        self.lock = threading.RLock()
        kwds = {}
        
        if port is not None:
            kwds['port'] = port

        self.redis = redis.StrictRedis(db=db, decode_responses=True, **kwds)
        self.cache = {}

    def get(self, key):
        """Get the value stored in a value-style key, or None if no key exists.

        Throws an exception if the value is a set.
        """
        with self.lock:
            if key in self.cache:
                assert not isinstance(self.cache[key], set), "item is a set, not a string"
                return self.cache[key]

            success = False
            while not success:
                try:
                    result = self.redis.get(key)
                    success = True
                except redis.exceptions.BusyLoadingError:
                    logging.info("Redis is still loading. Waiting...")
                    time.sleep(1.0)

            if result is None:
                return result

            result = result

            self.cache[key] = result

            return result

    def getSeveral(self, keys):
        """Get the values (or None) stored in several value-style keys."""

        with self.lock:
            needed_keys = [k for k in keys if k not in self.cache]

            if needed_keys:
                success = False
                while not success:
                    try:
                        vals = self.redis.mget(needed_keys)
                        success = True
                    except redis.exceptions.BusyLoadingError:
                        logging.info("Redis is still loading. Waiting...")
                        time.sleep(1.0)

                for ix in range(len(needed_keys)):
                    if vals[ix] is not None:
                        self.cache[needed_keys[ix]] = vals[ix]

            return [self.cache.get(k, None) for k in keys]

    def listKeys(self, prefix, suffix):
        """List all the keys (value or otherwise) with a given prefix and suffix."""
        with self.lock:
            for special in "\\*[]":
                prefix = prefix.replace(special, "\\" + special)
                suffix = suffix.replace(special, "\\" + special)

            return sorted(self.redis.list(prefix + "*" + suffix))

    def setMembers(self, key):
        with self.lock:
            if key in self.cache:
                assert isinstance(self.cache[key], set), "item is a string, not a set"
                return self.cache[key]

            success = False
            while not success:
                try:
                    vals = self.redis.smembers(key)
                    success = True
                except redis.exceptions.BusyLoadingError:
                    logging.info("Redis is still loading. Waiting...")
                    time.sleep(1.0)

            self.cache[key] = set([k for k in vals])
            return self.cache[key]


    def setAdd(self, key, values):
        self.setSeveral({}, {key: values}, {})

    def setRemove(self, key, values):
        self.setSeveral({}, {}, {key: values})

    def setSeveral(self, kvs, setAdds=None, setRemoves=None):
        with self.lock:
            pipe = self.redis.pipeline()

            for key, value in kvs.items():
                if value is None:
                    pipe.delete(key)
                else:
                    pipe.set(key, value)

            for key, to_add in (setAdds or {}).items():
                for to_add_val in to_add:
                    pipe.sadd(key, to_add_val)

            for key, to_remove in (setRemoves or {}).items():
                for to_remove_val in to_remove:
                    pipe.srem(key, to_remove_val)

            pipe.execute()

            #update our cache _after_ executing the pipe
            for key, value in kvs.items():
                if value is None:
                    if key in self.cache:
                        del self.cache[key]
                else:
                    self.cache[key] = value

            for key, to_add in (setAdds or {}).items():
                if key not in self.cache:
                    self.cache[key] = set()
                for val in to_add:
                    self.cache[key].add(val)

            for key, to_remove in (setRemoves or {}).items():
                if key not in self.cache:
                    self.cache[key] = set()
                for val in to_remove:
                    self.cache[key].discard(val)

    def set(self, key, value):
        with self.lock:
            if value is None:
                self.redis.delete(key)
                if key in self.cache:
                    del self.cache[key]
            else:
                self.cache[key] = value
                self.redis.set(key, value)

    def exists(self, key):
        with self.lock:
            if key in self.cache:
                return True
            return self.redis.exists(key)

    def delete(self, key):
        with self.lock:
            if key in self.cache:
                del self.cache[key]
            self.redis.delete(key)

    def clearCache(self):
        self.cache = {}
    