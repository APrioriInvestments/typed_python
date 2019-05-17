"""
TestRedis

utilities for bringing up a temporary redis backend for testing purposes.
"""

import subprocess
import redis
import time
import tempfile
import os

CONNECT_TIMEOUT = 0.5


class TestRedis:
    def __init__(self, port=1115):
        self.redisProcess = None
        self.port = port

        try:
            self.tempDir = tempfile.TemporaryDirectory()
            self.tempDirName = self.tempDir.name

            # start
            redis_path = "/usr/bin/redis-server"
            if not os.path.isfile(redis_path):
                redis_path = "/usr/local/bin/redis-server"

            self.redisProcess = subprocess.Popen(
                [redis_path, '--port', str(self.port), '--logfile', os.path.join(self.tempDirName, "log.txt"),
                    "--dbfilename", "db.rdb", "--dir", os.path.join(self.tempDirName)]
            )

            t0 = time.time()

            while not self.connect() and time.time() - t0 < CONNECT_TIMEOUT:
                time.sleep(0.01)

            assert self.redisProcess.poll() is None
        except BaseException:
            self.tearDown()

    def connect(self):
        """Try to connect. Returns None if we can't."""
        try:
            conn = redis.StrictRedis(port=self.port)
            conn.echo(b"connect_test")
            return conn
        except Exception:
            return None

    def tearDown(self):
        if self.redisProcess:
            self.redisProcess.terminate()
            self.redisProcess.wait()
        self.redisProcess = None
        self.tempDir.cleanup()
