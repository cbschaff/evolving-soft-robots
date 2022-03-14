from scipy.spatial.transform import Rotation
import numpy as np
import pickle
import shutil
import os


class ModeUtil():
    def __init__(self, filename):
        self.filename = filename

    def load(self):
        modes = []
        with open(self.filename, 'r') as f:
            self.header = f.readline()
            for line in f:
                modes.append(np.array([float(t.strip()) for t in line.split()]))
                # self.modes.append(mode.reshape(-1, 3))
        modes = np.array(modes).T
        self.modes = [mode.reshape(-1, 3) for mode in modes]

    def rotate(self, quat):
        r = Rotation.from_quat(quat)
        self.modes = [r.apply(mode) for mode in self.modes]

    def save(self, output_file):
        modes = np.array([mode.ravel() for mode in self.modes]).T
        with open(output_file, 'w') as f:
            f.write(self.header)
            for mode in modes:
                mode_tokens = ["{:1.5f}".format(m).rjust(11) for m in mode]
                f.write(''.join(mode_tokens) + '\n')


def rotate_basis(mor_dir, output_dir, quat):
    shutil.copytree(mor_dir, output_dir)
    loader = ModeUtil(os.path.join(mor_dir, 'data/modes.txt'))
    loader.load()
    loader.rotate(quat)
    loader.save(os.path.join(output_dir, 'data/modes.txt'))

    with open(os.path.join(output_dir, 'params.pkl'), 'r') as f:
        params = pickle.load(f)

    params.dataDir = os.path.abspath(os.path.join(output_dir, 'data'))

    with open(os.path.join(output_dir, 'params.pkl'), 'w') as f:
        pickle.dump(params, f)


if __name__ == '__main__':
    import argparse
    import mor_util

    parser = argparse.ArgumentParser()
    parser.add_argument('reduction', help="the name of the reduction to rotate")
    args = parser.parse_args()

    path = mor_util.get_reduction_path(args.reduction)
    angle = int(args.reduction[-4:]) / 10.
    for i in range(1, 8):
        new_angle = angle + 45. * i
        print(angle, new_angle)
        new_reduction = args.reduction[:-4]+'{:04d}'.format(int(10*new_angle))
        if mor_util.get_reduction_path(new_reduction):
            mor_util.delete_reduction(new_reduction)
        new_path = mor_util.register_reduction(new_reduction)
        rotate_basis(
            path, new_path,
            Rotation.from_euler('z', new_angle - angle, degrees=True).as_quat()
        )
