"""Defines trainer and network for MNIST."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin
import time
import numpy as np
from dl import logger, Checkpointer, nest
from dl.util import StatefulSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


@gin.configurable
class MNISTNet(nn.Module):
    """MNIST convolutional network."""

    def __init__(self):
        """Init."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        """Forward."""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


@gin.configurable(blacklist=['logdir'])
class MNISTTrainer(object):
    """Trainer for mnist."""

    def __init__(self, logdir, model, opt, batch_size, num_workers, gpu=True):
        self.logdir = logdir
        self.ckptr = Checkpointer(os.path.join(logdir, 'ckpts'))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_train = datasets.MNIST('./data_train', download=True,
                                         transform=self.transform)
        self.data_test = datasets.MNIST('./data_test', download=True,
                                        train=False, transform=self.transform)
        self.sampler = StatefulSampler(self.data_train, shuffle=True)
        self.dtrain = DataLoader(self.data_train, sampler=self.sampler,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
        self.dtest = DataLoader(self.data_test, batch_size=batch_size,
                                num_workers=num_workers)
        self._diter = None
        self.t = 0
        self.epochs = 0
        self.batch_size = batch_size

        self.device = torch.device('cuda:0' if gpu and torch.cuda.is_available()
                                   else 'cpu')
        self.model = model
        self.model.to(self.device)
        self.opt = opt(self.model.parameters())

    def step(self):
        # Get batch.
        if self._diter is None:
            self._diter = self.dtrain.__iter__()
        try:
            batch = self._diter.__next__()
        except StopIteration:
            self.epochs += 1
            self._diter = None
            return self.epochs
        batch = nest.map_structure(lambda x: x.to(self.device), batch)

        # compute loss
        x, y = batch
        self.model.train()
        loss = F.nll_loss(self.model(x), y)

        logger.add_scalar('train/loss', loss.detach().cpu().numpy(),
                          self.t, time.time())

        # update model
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # increment step
        self.t += min(len(self.data_train) - (self.t % len(self.data_train)),
                      self.batch_size)
        return self.epochs

    def evaluate(self):
        """Evaluate model."""
        self.model.eval()

        accuracy = []
        with torch.no_grad():
            for batch in self.dtest:
                x, y = nest.map_structure(lambda x: x.to(self.device), batch)
                y_hat = self.model(x).argmax(-1)
                accuracy.append((y_hat == y).float().mean().cpu().numpy())

            logger.add_scalar(f'test_accuracy', np.mean(accuracy),
                              self.epochs, time.time())

    def save(self):
        state_dict = {}
        state_dict['model'] = self.model.state_dict()
        state_dict['opt'] = self.opt.state_dict()
        state_dict['sampler'] = self.sampler.state_dict(self._diter)
        state_dict['t'] = self.t
        state_dict['epochs'] = self.epochs
        self.ckptr.save(state_dict, self.t)

    def load(self, t=None):
        state_dict = self.ckptr.load()
        if state_dict is None:
            self.t = 0
            self.epochs = 0
            return self.epochs
        self.model.load_state_dict(state_dict['model'])
        self.opt.load_state_dict(state_dict['opt'])
        self.sampler.load_state_dict(state_dict['sampler'])
        self.t = state_dict['t']
        self.epochs = state_dict['epochs']
        if self._diter is not None:
            self._diter.__del__()
            self._diter = None

    def close(self):
        """Close data iterator."""
        if self._diter is not None:
            self._diter.__del__()
            self._diter = None
