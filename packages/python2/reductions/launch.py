import subprocess
import argparse
import os
import mor_util
import signal


def get_reduction_cls(reduction_name):
    reduction = __import__('reductions.{}'.format(reduction_name))
    reduction = getattr(reduction, reduction_name)
    return reduction.DiskWithLegsReduction


def register_animation_fn(reduction_name):
    reduction = __import__('reductions.{}'.format(reduction_name))
    reduction = getattr(reduction, reduction_name)
    mor_util.register_animation(reduction.ANIMATION_FN)


def reduce_part(args):
    R = get_reduction_cls(args.reduction_name)
    register_animation_fn(args.reduction_name)
    name = args.prefix + '_' + args.part
    asset_dir = mor_util.get_reduction_path(name)
    if asset_dir is None:
        asset_dir = mor_util.register_reduction(name)
        os.makedirs(asset_dir)

    reduction = R(asset_dir, params={
        'part_to_reduce': args.part,
        'reduction_prefix': args.prefix,
        'load_reduced': False,
        'use_all_reductions': False
    })

    reduction.reduce(tolModes=args.tolModes,
                     tolGIE=args.tolGIE,
                     ncpu=args.ncpu,
                     phases=args.phases)
    if 4 in args.phases:
        reduction.cleanup()


def make_command(args, part):
    cmd = 'python2 -m reductions.launch {} {} --tolModes {} --tolGIE {} --ncpu {} --part {}'
    cmd = cmd.format(args.reduction_name, args.prefix, args.tolModes,
                     args.tolGIE, args.ncpu, part)
    if len(args.phases) > 0:
        phase_str = ' '.join([str(phase) for phase in args.phases])
        cmd += ' --phases ' + phase_str
    return cmd


if __name__ == '__main__':
    parser = argparse.ArgumentParser("launch reduction")
    parser.add_argument("reduction_name")
    parser.add_argument("prefix")
    parser.add_argument("--tolModes", default=0.001, type=float, help="tolModes")
    parser.add_argument("--tolGIE", default=0.05, type=float, help="tolGIE")
    parser.add_argument("--ncpu", default=8, type=int, help="ncpu")
    parser.add_argument("--nparallel", default=1, type=int,
                        help="n parallel reductions")
    parser.add_argument("--part", default=None, help="part to reduce")
    parser.add_argument('--phases', type=int, nargs='+',
                        default=[1, 2, 3, 4], help='which phases to execute')
    args = parser.parse_args()

    if args.part is not None:
        reduce_part(args)
    else:

        commands = [[] for _ in range(args.nparallel)]
        R = get_reduction_cls(args.reduction_name)
        for i, part in enumerate(R.PARTS):
            commands[i % args.nparallel].append(make_command(args, part))
        try:
            procs = []
            for command_list in commands:
                print("STARTING subprocess")
                procs.append(subprocess.Popen('\n'.join(command_list), shell=True))

            for proc in procs:
                proc.wait()
        except KeyboardInterrupt:
            for proc in procs:
                proc.send_signal(signal.SIGINT)
                proc.terminate()
