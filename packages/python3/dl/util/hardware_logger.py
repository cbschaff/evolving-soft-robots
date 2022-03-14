"""
This code is adapted from https://github.com/anderskm/gputil.
"""
import psutil
from subprocess import Popen, PIPE
from collections import namedtuple
from threading import Thread
import time
import os


_GPU = namedtuple('GPU', 'id,util,memutil')


def getGPUs():
    def _cast_float(x):
        try:
            return float(x)
        except ValueError:
            return float('nan')

    # Get ID, processing and memory utilization for all GPUs
    try:
        p = Popen(["nvidia-smi",
                   "--query-gpu=index,utilization.gpu,memory.total,memory.used",
                   "--format=csv,noheader,nounits"], stdout=PIPE)
        stdout, stderror = p.communicate()
    except Exception:
        return []
    output = stdout.decode('UTF-8')
    lines = output.split(os.linesep)
    numDevices = len(lines)-1
    gpus = []
    for g in range(numDevices):
        line = lines[g]
        vals = line.split(', ')
        try:
            id = int(vals[0])
            util = _cast_float(vals[1])
            mem_total = _cast_float(vals[2])
            mem_used = _cast_float(vals[3])
            gpus.append(_GPU(id=id, util=util, memutil=100 * mem_used/mem_total))
        except Exception:
            pass
    return gpus


def get_stats():
    cpu_util = psutil.cpu_percent()
    mem_util = psutil.virtual_memory().percent
    gpus = getGPUs()
    cvd = os.getenv("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        ids = [int(d) for d in cvd.split(',')]
        gpus = [gpu for gpu in gpus if gpu.id in ids]

    return cpu_util, mem_util, gpus


def log_stats():
    from dl import logger
    cpu_util, mem_util, gpus = get_stats()
    timestamp = time.time()
    logger.add_scalar('hardware/cpu_util', cpu_util, walltime=timestamp)
    logger.add_scalar('hardware/mem_util', mem_util, walltime=timestamp)
    logger.add_scalar('hardware/cpu_util', cpu_util, walltime=timestamp)
    for gpu in gpus:
        logger.add_scalar(f'hardware/gpu{gpu.id}/util', gpu.util,
                          walltime=timestamp)
        logger.add_scalar(f'hardware/gpu{gpu.id}/mem_util', gpu.memutil,
                          walltime=timestamp)


class HardwareLogger(Thread):
    def __init__(self, delay):
        super().__init__()
        self.stopped = False
        self.delay = delay
        self.start()
        time.sleep(1)  # wait for thread to start

    def run(self):
        while not self.stopped:
            log_stats()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
