"""Global Tensorboard writer."""
from torch.utils.tensorboard import SummaryWriter
import time


def log(out):
    """Print function."""
    print(out)


WRITER = None


def configure(logdir, **kwargs):
    """Configure logger."""
    global WRITER, LOGDIR
    WRITER = TBWriter(logdir, **kwargs)


def get_summary_writer():
    """Get global writer."""
    return WRITER


def get_dir():
    """Get log directory."""
    return None if WRITER is None else WRITER.log_dir


class TBWriter(SummaryWriter):
    """Subclass SummaryWriter to check inputs."""

    def __init__(self, logdir, *args, **kwargs):
        """Init."""
        super().__init__(logdir, *args, **kwargs)
        self.last_flush = time.time()

    def _unnumpy(self, x):
        """Numpy data types are not json serializable."""
        if hasattr(x, 'tolist'):
            return x.tolist()
        return x

    def _scalarize(self, x):
        """Turn into scalar."""
        x = self._unnumpy(x)
        if isinstance(x, list):
            if len(x) == 1:
                return self._scalarize(x[0])
            else:
                assert False, "Tried to log something that isn't a scalar!"
        return x

    def flush(self, force=True):
        """Flush logs to disk."""
        if time.time() - self.last_flush > 60 or force:
            for writer in self.all_writers.values():
                writer.flush()
            self.last_flush = time.time()

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        """Add scalar to TB."""
        scalar_value = self._scalarize(scalar_value)
        global_step = self._scalarize(global_step)
        walltime = self._scalarize(walltime)
        super().add_scalar(tag, scalar_value, global_step, walltime)
        self.flush(force=False)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None,
                    walltime=None):
        """Add scalar dict to TB."""
        for k, v in tag_scalar_dict.items():
            tag_scalar_dict[k] = self._scalarize(v)
        global_step = self._scalarize(global_step)
        walltime = self._scalarize(walltime)
        super().add_scalars(main_tag, tag_scalar_dict, global_step, walltime)
        self.flush(force=False)


def add_scalar(*args, **kwargs):
    """Log to TB. See pytorch documentation for interface."""
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_scalar(*args, **kwargs)


def add_scalars(*args, **kwargs):
    """Log to TB. See pytorch documentation for interface."""
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_scalars(*args, **kwargs)


def add_histogram(*args, **kwargs):
    """Log to TB. See pytorch documentation for interface."""
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_histogram(*args, **kwargs)
    WRITER.flush(force=True)


def add_image(*args, **kwargs):
    """Log to TB. See pytorch documentation for interface."""
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_image(*args, **kwargs)
    WRITER.flush(force=True)


def add_images(*args, **kwargs):
    """Log to TB. See pytorch documentation for interface."""
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_images(*args, **kwargs)
    WRITER.flush(force=True)


def add_figure(*args, **kwargs):
    """Log to TB. See pytorch documentation for interface."""
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_figure(*args, **kwargs)
    WRITER.flush(force=True)


def add_video(*args, **kwargs):
    """Log to TB. See pytorch documentation for interface."""
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_video(*args, **kwargs)
    WRITER.flush(force=True)


def add_audio(*args, **kwargs):
    """Log to TB. See pytorch documentation for interface."""
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_audio(*args, **kwargs)
    WRITER.flush(force=True)


def add_text(*args, **kwargs):
    """Log to TB. See pytorch documentation for interface."""
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_text(*args, **kwargs)
    WRITER.flush(force=True)


def add_graph(*args, **kwargs):
    """Log to TB. See pytorch documentation for interface."""
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_graph(*args, **kwargs)
    WRITER.flush(force=True)


def flush():
    """Flush logs."""
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.flush(force=True)


def close():
    """Close writer."""
    global WRITER
    if WRITER is not None:
        WRITER.flush(force=True)
        WRITER.close()
        WRITER = None
