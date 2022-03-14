from . import reduction_registry as r
import os
import pickle


def change_path(mor_dir, new_path):
    with open(os.path.join(mor_dir, 'params.pkl'), 'r') as f:
        params = pickle.load(f)

    params.dataDir = os.path.abspath(os.path.join(new_path, 'data'))

    with open(os.path.join(mor_dir, 'params.pkl'), 'w') as f:
        pickle.dump(params, f)


if __name__ == '__main__':
    names = r.list_reductions()
    for name in names:
        if 'leg' not in name:
            continue
        prefix, angle = name.split('leg')
        path = r.get_reduction_path(name)
        try:
            change_path(path, path)
        except IOError as e:
            print(e)
