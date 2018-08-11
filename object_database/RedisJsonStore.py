import redis
import ujson as json
import time
import threading
import logging
import os


class RedisJsonStore(object):
    """Implements a string-to-json store using Redis.

    Keys must be strings. Values may be anything that's json-compatible.

    You may store values using the "set" and "get" methods. 

    You may store sets of values using the setAdd, setRemove, setMembers, methods.

    You may not use a value-style method on a set-style key or vice versa.

    Setting a key to "none" deletes it, and non-existent keys are implicity 'None'

    You may delete any kind of value.

    This class is thread-safe.
    """
    def __init__(self, db=0, port=None):
        self.lock = threading.RLock()
        kwds = {}
        
        if port is not None:
            kwds['port'] = port

        self.redis = redis.StrictRedis(db=db, **kwds)
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

            result = json.loads(result)

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
                return sorted(self.cache[key])

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
                    pipe.set(key, json.dumps(value))

            for key, to_add in (setAdds or {}).items():
                for to_add_val in to_add:
                    pipe.sadd(key, json.dumps(to_add_val))

            for key, to_remove in (setRemoves or {}).items():
                for to_remove_val in to_remove:
                    pipe.srem(key, json.dumps(to_remove_val))

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
                for val in to_add:
                    self.cache[key].discard(val)

    def set(self, key, value):
        with self.lock:
            if value is None:
                self.redis.delete(key)
                if key in self.cache:
                    del self.cache[key]
            else:
                self.cache[key] = value
                self.redis.set(key, json.dumps(value))

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
    