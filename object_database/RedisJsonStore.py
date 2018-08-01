import redis
import json
import time
import threading
import logging
import os


class RedisJsonStore(object):
    """Implements a string-to-json store using Redis.

    Keys must be strings. Values may be anything that's json-compatible.

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
        with self.lock:
            if key in self.cache:
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

    def setSeveral(self, kvs):
        with self.lock:
            pipe = self.redis.pipeline()

            for key, value in kvs.items():
                if value is None:
                    pipe.delete(key)
                    if key in self.cache:
                        del self.cache[key]
                else:
                    self.cache[key] = value
                    pipe.set(key, json.dumps(value))

            pipe.execute()

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
    