import argparse
import wandb
from training import utils


if __name__=='__main__':
    parser = argparse.ArgumentParser('viz')
    parser.add_argument('logdir')
    parser.add_argument('-t', '--ckpt', type=int, default=None)
    args = parser.parse_args()

    wandb.init(mode='disabled')
    runner = utils.EpisodeRunner(args.logdir, args.ckpt, record=True, unreduced=False)
    runner.run_episode()
