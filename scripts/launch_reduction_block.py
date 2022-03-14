import os
import glob
from subprocess import Popen
import shutil
import time

SCRIPT = "universal_leg_reduction"
PREFIX = "leg_reduction"
TOLMODES = [0.0032]
TOLGIE = [0.001]
NCPU = "32"
PART = "leg0225"
TMP_DIR = "assets/partial_reduction_cache"
ROOT = os.path.join(os.getcwd(), 'assets/reductions')
CPK_ROOT = '/code/src/evolving-soft-robots/assets/reductions'
VIDEO_ROOT = os.path.join(os.getcwd(), 'assets/reduction_videos')

class ProcManager():
    def __init__(self):
        self.procs = {}

    def run(self, cmd, block=True):
        proc = Popen(cmd, shell=True)
        self.procs[proc.pid] = proc
        if block:
            proc.wait()
            del self.procs[proc.pid]

    def wait_all(self):
        for pid, proc in self.procs.items():
            proc.wait()
        self.procs = {}

    def close_all(self):
        for pid, proc in self.procs.items():
            proc.terminate()
        self.procs = {}


def get_name(tolModes, tolGIE):
    return PREFIX + f'_tolm_{tolModes:1.4f}_tolg_{tolGIE:1.4f}'


def get_path(tolModes, tolGIE):
    return os.path.join(ROOT, get_name(tolModes, tolGIE) + f'_{PART}')


def get_reduction_cmd(phases, tolModes, tolGIE):
    name = get_name(tolModes, tolGIE)
    args = (f" --phases {phases} --tolModes {tolModes} --tolGIE {tolGIE} "
            + f"--ncpu {NCPU} --part {PART} {SCRIPT} {name}")

    return f'cpk run -n reduction_{name} -M -f -L reduce -A " {args}" -- -v {TMP_DIR}:/tmp -v {ROOT}:{CPK_ROOT}'


def get_rotate_cmd(tolModes, tolGIE):
    name = get_name(tolModes, tolGIE)
    return f'cpk run -n rotate_{name} -M -f -L rotate_reduction -A " {name}_{PART}" -- -v {TMP_DIR}:/tmp -v {ROOT}:{CPK_ROOT}'


def get_animate_cmd(tolModes, tolGIE):
    name = get_name(tolModes, tolGIE)
    return f'cpk run -n "animate_{name}" -M -f -L animate_reduction -A " {SCRIPT} {name} --part all" -- -v {TMP_DIR}:/tmp -v {ROOT}:{CPK_ROOT}'


def run_block(procman):
    procman.run(get_reduction_cmd("1", TOLMODES[0], TOLGIE[0]))
    path = get_path(TOLMODES[0], TOLGIE[0])
    tmp_path = os.path.join(TMP_DIR, 'phase1_cache')
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    shutil.copytree(path, tmp_path)
    for i, tm in enumerate(TOLMODES):
        path = get_path(tm, TOLGIE[0])
        if not os.path.exists(path):
            shutil.copytree(tmp_path, path)
        tmp_path_phase3 = os.path.join(TMP_DIR, 'phase3_cache')
        procman.run(get_reduction_cmd("2 3", tm, TOLGIE[0]))
        if os.path.exists(tmp_path_phase3):
            shutil.rmtree(tmp_path_phase3)
        shutil.copytree(path, tmp_path_phase3)

        for j, tg in enumerate(TOLGIE):
            path = get_path(tm, tg)
            if not os.path.exists(path):
                shutil.copytree(tmp_path_phase3, path)
            procman.run(get_reduction_cmd("4", tm, tg))
            procman.run(get_rotate_cmd(tm, tg))
            procman.run(get_animate_cmd(tm, tg), block=False)
            time.sleep(1.0)
        shutil.rmtree(tmp_path_phase3)
        # remove other tmp files
        for path in glob.glob('/tmp/sofa*'):
            shutil.rmtree(path)
    shutil.rmtree(tmp_path)

    procman.wait_all()

    # move videos
    outdir = os.path.join(VIDEO_ROOT, PREFIX + '_block')
    os.makedirs(outdir, exist_ok=True)
    for tm in TOLMODES:
        for tg in TOLGIE:
            name = get_name(tm, tg)
            path = os.path.join(ROOT, 'videos', name, 'animate_all_reduced.mp4')
            outpath = os.path.join(outdir, f'{name}.mp4')
            if os.path.exists(path):
                shutil.copyfile(path, outpath)



if __name__ == '__main__':
    procman = ProcManager()
    try:
        run_block(procman)
    except KeyboardInterrupt:
        procman.close_all()
