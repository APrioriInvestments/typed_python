#!/usr/bin/env python3

import argparse
import sys
import time
import typed_python
import object_database.database as database
import object_database.InMemoryJsonStore as InMemoryJsonStore
import object_database.RedisJsonStore as RedisJsonStore

def main(argv):
    parser = argparse.ArgumentParser("Run an object_database server")

    parser.add_argument("host")
    parser.add_argument("port", type=int)
    parser.add_argument("--redis_port", type=int, default=None)
    parser.add_argument("--inmem", default=False, action='store_true')

    parsedArgs = parser.parse_args(argv[1:])

    if parsedArgs.inmem:
        mem_store = InMemoryJsonStore.InMemoryJsonStore()
    else:
        mem_store = RedisJsonStore.RedisJsonStore(port=parsedArgs.redis_port)

    db = database.Database(mem_store)
    databaseServer = database.DatabaseServer(db, parsedArgs.host, parsedArgs.port)

    databaseServer.start()

    try:
    	while True:
    		time.sleep(0.1)
    except KeyboardInterrupt:
    	return

if __name__ == '__main__':
    main(sys.argv)