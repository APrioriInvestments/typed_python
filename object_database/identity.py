import threading


class IdentityProducer:
    def __init__(self, ix):
        self.ix = ix
        self.count = 0
        self.lock = threading.Lock()

    def createIdentity(self):
        with self.lock:
            count = self.count
            self.count += 1

        return str(self.ix) + "_" + str(count)
