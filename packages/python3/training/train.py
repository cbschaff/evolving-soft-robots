import argparse
import dl
import yaml
import wandb
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Agent.')
    parser.add_argument('config', type=str, help='config')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    gin_bindings = []
    run_id = os.path.basename(config['logdir'])
    with wandb.init(project='evolving-soft-robots',
                    id=run_id+'_train', group=run_id,
                    job_type='train', resume='allow', config=config):
        config = wandb.config
        for k, v in config['gin_bindings'].items():
            if isinstance(v, str) and v[0] != '@':
                gin_bindings.append(f'{k}="{v}"')
            else:
                gin_bindings.append(f"{k}={v}")
        dl.load_config(config['base_config'], gin_bindings)
        dl.train(config['logdir'])
