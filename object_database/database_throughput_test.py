#!/usr/bin/env python3
import threading
import argparse
import sys
import time
import object_database.database as database

def main(argv):
    parser = argparse.ArgumentParser("Run a database throughput test")

    parser.add_argument("host")
    parser.add_argument("port")
    parser.add_argument("--threads", default=1, type=int)

    parsedArgs = parser.parse_args(argv[1:])

    db = database.connect(parsedArgs.host, parsedArgs.port)

    @db.define
    class Counter:
        k = int

    def worker():
        with db.transaction():
            c = Counter.New()

        while True:
            t0 = time.time()
            
            with db.transaction():
                c.k = 0

            while time.time() - t0 < 1.0:
                with db.transaction():
                    c.k = c.k + 1

            with db.view():
                print(c.k, " transactions per second")

    threads = [threading.Thread(target=worker) for _ in range(parsedArgs.threads)]
    for t in threads:
        t.daemon = True
        t.start()

    while True:
        time.sleep(0.01)

if __name__ == '__main__':
    try:
    	main(sys.argv)
    except KeyboardInterrupt:
        pass
    except:
    	import traceback
    	traceback.print_exc()