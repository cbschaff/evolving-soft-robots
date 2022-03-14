"""Basic training loop."""
import gin
import os
import time
from dl import logger, rng, HardwareLogger


class Algorithm(object):
    """Interface for Deep Learning algorithms."""

    def __init__(self, logdir):
        """Init."""
        raise NotImplementedError

    def step(self):
        """Step the optimization.

        This function should return the step count of the algorithm.
        """
        raise NotImplementedError

    def evaluate(self):
        """Evaluate model."""
        raise NotImplementedError

    def save(self):
        """Save model."""
        raise NotImplementedError

    def load(self):
        """Load model.

        This function should return the step count of the algorithm.
        """
        raise NotImplementedError

    def close(self):
        """Clean up at end of training."""
        pass


@gin.configurable(blacklist=['logdir'])
def train(logdir,
          algorithm,
          seed=0,
          eval=False,
          eval_period=None,
          save_period=None,
          maxt=None,
          maxseconds=None,
          hardware_poll_period=1):
    """Basic training loop.

    Args:
        logdir (str):
            The base directory for the training run.
        algorithm_class (Algorithm):
            The algorithm class to use for training. A new instance of the class
            will be constructed.
        seed (int):
            The initial seed of this experiment.
        eval (bool):
            Whether or not to evaluate the model throughout training.
        eval_period (int):
            The period with which the model is evaluated.
        save_period (int):
            The period with which the model is saved.
        maxt (int):
            The maximum number of timesteps to train the model.
        maxseconds (float):
            The maximum amount of time to train the model.
        hardware_poll_period (float):
            The period in seconds at which cpu/gpu stats are polled and logged.
            Use 'None' to disable logging.
    """

    logger.configure(os.path.join(logdir, 'tb'))
    rng.seed(seed)
    alg = algorithm(logdir=logdir)
    config = gin.operative_config_str()
    logger.log("=================== CONFIG ===================")
    logger.log(config)
    with open(os.path.join(logdir, 'config.gin'), 'w') as f:
        f.write(config)
    time_start = time.monotonic()
    t = alg.load()
    if t == 0:
        cstr = config.replace('\n', '  \n')
        cstr = cstr.replace('#', '\\#')
        logger.add_text('config', cstr, 0, time.time())
    if maxt and t > maxt:
        return
    if save_period:
        last_save = (t // save_period) * save_period
    if eval_period:
        last_eval = (t // eval_period) * eval_period

    if hardware_poll_period is not None and hardware_poll_period > 0:
        hardware_logger = HardwareLogger(delay=hardware_poll_period)
    else:
        hardware_logger = None
    try:
        while True:
            if maxt and t >= maxt:
                break
            if maxseconds and time.monotonic() - time_start >= maxseconds:
                break
            t = alg.step()
            if save_period and (t - last_save) >= save_period:
                alg.save()
                last_save = t
            if eval and (t - last_eval) >= eval_period:
                alg.evaluate()
                last_eval = t
    except KeyboardInterrupt:
        logger.log("Caught Ctrl-C. Saving model and exiting...")
    alg.save()
    if hardware_logger:
        hardware_logger.stop()
    logger.flush()
    logger.close()
    alg.close()


if __name__ == '__main__':

    import unittest
    import shutil

    EVAL_PERIOD = 100
    SAVE_PERIOD = 50

    class A(Algorithm):
        """Dummy algorithm."""

        def __init__(self, logdir):
            """Init."""
            self.t = 0

        def step(self):
            """Step."""
            self.t += 1
            return self.t

        def evaluate(self):
            """Eval."""
            assert self.t % EVAL_PERIOD == 0

        def save(self):
            """Save."""
            assert self.t % SAVE_PERIOD == 0

        def load(self, t=None):
            """Load."""
            return self.t

    class TestTrainer(unittest.TestCase):
        """Test."""

        def test(self):
            """Test."""
            global SAVE_PERIOD
            train('logs', A, eval=True, eval_period=EVAL_PERIOD,
                  save_period=SAVE_PERIOD, maxt=1000)
            shutil.rmtree('logs')

            SAVE_PERIOD = 1
            train('logs', A, eval=True, eval_period=EVAL_PERIOD,
                  save_period=SAVE_PERIOD, maxseconds=2)
            shutil.rmtree('logs')

    unittest.main()
