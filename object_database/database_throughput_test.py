#!/usr/bin/env python3

import argparse
import sys
import time
import object_database.database as database

def main(argv):
    parser = argparse.ArgumentParser("Run a database throughput test")

    parser.add_argument("host")
    parser.add_argument("port")

    parsedArgs = parser.parse_args(argv[1:])

    db = database.connect(parsedArgs.host, parsedArgs.port)

    @db.define
    class Counter:
        k = int

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

if __name__ == '__main__':
    try:
    	main(sys.argv)
    except:
    	import traceback
    	traceback.print_exc()