"""Main script for training models."""
import dl
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Script.')
    parser.add_argument('logdir', type=str, help='logdir')
    parser.add_argument('config', type=str, help='gin config')
    parser.add_argument('-b', '--bindings', nargs='+',
                        help='gin bindings to overwrite config')
    args = parser.parse_args()

    dl.load_config(args.config, args.bindings)
    dl.train(args.logdir)
