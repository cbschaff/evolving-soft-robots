import argparse
import os
import shutil
import mor_util


REDUCTIONS_ROOT = '/code/src/evolving-soft-robots/assets/reductions'


def get_reduction_cls(reduction_name):
    reduction = __import__('reductions.{}'.format(reduction_name))
    reduction = getattr(reduction, reduction_name)
    return reduction.DiskWithLegsReduction


def register_animation_fn(reduction_name):
    reduction = __import__('reductions.{}'.format(reduction_name))
    reduction = getattr(reduction, reduction_name)
    mor_util.register_animation(reduction.ANIMATION_FN)


def get_asset_dir(name):
    asset_dir = mor_util.get_reduction_path(name)
    if not asset_dir:
        asset_dir = mor_util.register_reduction(name)
        os.makedirs(asset_dir)
    return asset_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser("launch reduction")
    parser.add_argument("reduction_name")
    parser.add_argument("prefix")
    parser.add_argument("--part", default=None,
                        help="which part to use for reduction. Pass 'all' to animate with all parts reduced")
    parser.add_argument("--mode", default=None,
                        help="the animation mode.")
    args = parser.parse_args()

    if args.part is not None:
        name = args.prefix + '_animation_' + args.part
    else:
        name = args.prefix + '_animation'
    asset_dir = get_asset_dir(name)

    register_animation_fn(args.reduction_name)
    R = get_reduction_cls(args.reduction_name)
    reduction = R(asset_dir, params={
        'part_to_reduce': args.part,
        'reduction_prefix': args.prefix,
        'load_reduced': args.part is not None,
        'use_all_reductions': args.part == 'all'
    })

    reduction.inspect_scene_graph()
    if args.mode is None:
        mode = [1] * len(reduction.make_object_list())
    else:
        mode = eval(args.mode)
    reduction.animate(mode, R.DURATION)

    if args.part is None:
        outfile = 'animate_unreduced.mp4'
    else:
        outfile = 'animate_{}_reduced.mp4'.format(args.part)
    new_path = os.path.join(REDUCTIONS_ROOT, 'videos', args.prefix, outfile)
    if not os.path.exists(os.path.dirname(new_path)):
        os.makedirs(os.path.dirname(new_path))
    os.rename(os.path.join(asset_dir, 'videos/animate.mp4'), new_path)
    shutil.rmtree(os.path.join(REDUCTIONS_ROOT, name))
