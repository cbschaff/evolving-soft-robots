import argparse
import os
import wandb
from matplotlib import pyplot as plt
import numpy as np

from training import utils
from dl.rl import set_env_to_eval_mode


def plot(episode_data, logdir, ckpt, unreduced):
    design = episode_data['design']
    length = episode_data['length']
    commanded_action = episode_data['commanded_action']
    smoothed_action = episode_data['smoothed_action']

    fig, axs = plt.subplots(len(design), 1, sharex=True, figsize=(6.4, len(design) * 1.25))
    x = np.arange(length)
    for i, ax in enumerate(axs):
        if design[i] == 0:
            continue
        ax.plot(x, commanded_action[:, design[i]-1], label='Policy Output')
        ax.plot(x, smoothed_action[:, i], label='Commanded Pressure')
        if i == 0:
            ax.legend(bbox_to_anchor=(0.5, 1.22), loc='upper center',
                      fontsize='xx-small', ncol=2)
    plt.xlabel('Time (ms)')
    ax_invis = fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel('Pressure (kPa)')
    os.makedirs(os.path.join(logdir, 'plots'), exist_ok=True)
    if unreduced:
        plt.savefig(os.path.join(logdir, f'plots/{ckpt:09d}_no_reduction.pdf'))
    else:
        plt.savefig(os.path.join(logdir, f'plots/{ckpt:09d}.pdf'))
    data = {
        'pi_out': commanded_action,
        'actions': smoothed_action
    }
    outdir = os.path.join(logdir, 'action_logs')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    np.savez(os.path.join(outdir, '{}_log.npz'.format(ckpt)), **data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('viz')
    parser.add_argument('logdir')
    parser.add_argument('-n', '--n_episodes', type=int, default=1)
    parser.add_argument('-t', '--ckpt', type=int, default=None)
    parser.add_argument('--unreduced', default=False, action='store_true')
    args = parser.parse_args()

    run_id = os.path.basename(args.logdir)
    if args.unreduced:
        job_id = run_id + '_record_unreduced'
    else:
        job_id = run_id + '_record'

    with wandb.init(project='evolving-soft-robots', group=run_id, 
                    id=job_id, resume='allow', job_type='record'):

        runner = utils.EpisodeRunner(args.logdir, args.ckpt, record=True, unreduced=args.unreduced)

        # only record unreduced videos every 10 checkpoints because they are slow.
        if not args.unreduced or runner.ckpt % (10 * runner.alg.ckptr.ckpt_period) < runner.alg.ckptr.ckpt_period:
            video_dir = os.path.join(args.logdir, 'video')
            os.makedirs(video_dir, exist_ok=True)
            if args.unreduced:
                outfile = os.path.join(video_dir, f'{runner.ckpt:09d}_unreduced.mp4')
            else:
                outfile = os.path.join(video_dir, f'{runner.ckpt:09d}.mp4')
            episode_data = runner.record_episodes(args.n_episodes, outfile)
            plot(episode_data[0], args.logdir, runner.ckpt, args.unreduced)
            if args.unreduced:
                wandb.log({'eval/video_no_reduction': wandb.Video(outfile),
                           'eval/pressures_no_reduction': plt}, step=int(runner.ckpt))
            else:
                wandb.log({'eval/video': wandb.Video(outfile),
                           'eval/pressures': plt}, step=int(runner.ckpt))
