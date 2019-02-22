import psutil


def currentMemUsageMb(residentOnly=True):
    if residentOnly:
        return psutil.Process().memory_info().rss / 1024 ** 2
    else:
        return psutil.Process().memory_info().vms / 1024 ** 2
