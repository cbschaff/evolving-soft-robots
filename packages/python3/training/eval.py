import argparse
import json
import os
import numpy as np
import wandb

from training import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser('eval sac')
    parser.add_argument('logdir')
    parser.add_argument('-n', '--n_episodes', type=int, default=1)
    parser.add_argument('-t', '--ckpt', type=int, default=None)
    args = parser.parse_args()

    run_id = os.path.basename(args.logdir)

    with wandb.init(project='evolving-soft-robots', group=run_id,
                    id=run_id + '_eval', resume='allow', job_type='eval'):
        runner = utils.EpisodeRunner(args.logdir, args.ckpt)

        rewards = []
        lengths = []
        for _ in range(args.n_episodes):
            episode_data = runner.run_episode()
            rewards.append(episode_data['reward'])
            lengths.append(episode_data['length'])

        eval_dir = os.path.join(args.logdir, 'test')
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)

        data = {
            'rewards': rewards,
            'lengths': lengths,
            'mean': np.mean(rewards).item(),
            'std': np.std(rewards).item(),
            'min': min(rewards),
            'max': max(rewards),
        }
        with open(os.path.join(eval_dir, f'{runner.ckpt:09d}.json'), 'w') as f:
            json.dump(data, f)

        wandb.define_metric('eval/reward', summary='max')
        wandb.log({'eval/reward_histogram': wandb.Histogram(rewards), 
                   'eval/reward': np.mean(rewards).item()}, step=int(runner.ckpt))
