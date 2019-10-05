#!/usr/bin/env python3
# Copyright 2019 Christian Henning
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A collection of helper functions
--------------------------------

The module :mod:`lib.utils` contains several general purpose utilities and
helper functions.

.. autosummary::

    lib.utils.regression_cubic_poly
    lib.utils.generate_data_from_teacher
    lib.utils.RegressionDataset
    lib.utils.plot_predictions
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from lib.mlp import MLP

def regression_cubic_poly(num_train=20, num_test=100, train_domain=(-4,-4),
                          test_domain=(-4, 4), rseed=42):
    r"""Generate a dataset for a 1D regression task with a cubic polynomial.

    The regression task modelled here is :math:`y = x^3 + \epsilon`,
    where :math:`\epsilon \sim \mathcal{N}(0, 9I)`.

    Args:
        num_train (int): Number of training samples.
        num_test (int): Number of test samples.
        train_domain (tuple): Input domain for training samples.
        test_domain (tuple): Input domain for training samples.
        rseed (int): To ensure reproducibility, the random seed for the data
            generation should be decoupled from the random seed of the
            simulation. Therefore, a new :class:`numpy.random.RandomState` is
            created for the purpose of generating the data.

    Returns:
        (tuple): Tuple containing:

        - **train_x**: Generated training inputs.
        - **test_x**: Generated test inputs.
        - **train_y**: Generated training outputs.
        - **test_y**: Generated test outputs.

        Data is returned in form of 2D arrays of class :class:`numpy.ndarray`.
    """
    rand = np.random.RandomState(rseed)

    train_domain = [-4, 4]

    train_x = rand.uniform(low=train_domain[0], high=train_domain[1],
                               size=(num_train, 1))
    test_x = np.linspace(start=test_domain[0], stop=test_domain[1],
                         num=num_test).reshape((num_test, 1))

    map_function = lambda x : (x**3.)
    train_y = map_function(train_x)
    test_y = map_function(test_x)

    # Add noise to training outputs.
    train_eps = rand.normal(loc=0.0, scale=3, size=(num_train, 1))
    train_y += train_eps

    return train_x, test_x, train_y, test_y

def generate_data_from_teacher(num_train=1000, num_test=100, n_in=5, n_out=5,
                               n_hidden=[10,10,10]):
    """Generate data for a regression task through a teacher model.

    This function generates random input patterns and creates a random MLP
    (fully-connected neural network), that is used as a teacher model. I.e., the
    generated input data is fed through the teacher model to produce target
    outputs. The so produced dataset can be used to train and assess a
    student model. Hence, a learning procedure can be verified by validating its
    capability of training a student network to mimic a given teacher network.

    Input samples will be uniformly drawn from a unit cube.

    .. warning::
        Since this is a synthetic dataset that uses random number generators,
        the generated dataset depends on externally configured random seeds
        (and in case of GPU computation, it also depends on whether CUDA
        operations are performed in a derterministic mode).

    Args:
        num_train (int): Number of training samples.
        num_test (int): Number of test samples.
        n_in (int): Passed as argument ``n_in`` to class :class:`lib.mlp.MLP`
            when building the teacher model.
        n_out (int): Passed as argument ``n_out`` to class :class:`lib.mlp.MLP`
            when building the teacher model.
        n_hidden (list): Passed as argument ``n_hidden`` to class
            :class:`lib.mlp.MLP` when building the teacher model.

    Returns:
        See return values of function :func:`regression_cubic_poly`.
    """
    # FIXME Disentangle the random seeds set in a simulation from the one used
    # to generate synthetic datasets.
    rand = np.random

    train_x = rand.uniform(low=0, high=1, size=(num_train, n_in))
    test_x = rand.uniform(low=0, high=1, size=(num_train, n_in))

    teacher = MLP(n_in=n_in, n_out=n_out, n_hidden=n_hidden)

    train_y = teacher.forward(torch.from_numpy(train_x).float()).detach(). \
        numpy()
    test_y = teacher.forward(torch.from_numpy(test_x).float()).detach(). \
        numpy()

    return train_x, test_x, train_y, test_y

class RegressionDataset(Dataset):
    """A simple regression dataset.

    Args:
        inputs (numpy.ndarray): The input samples.
        outputs (numpy.ndarray): The output samples.
    """
    def __init__(self, inputs, outputs):
        assert(len(inputs.shape) == 2)
        assert(len(outputs.shape) == 2)
        assert(inputs.shape[0] == outputs.shape[0])

        self.inputs = torch.from_numpy(inputs).float()
        self.outputs = torch.from_numpy(outputs).float()

    def __len__(self):
        return int(self.inputs.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        batch_in = self.inputs[idx, :]
        batch_out = self.outputs[idx, :]

        return batch_in, batch_out

def plot_predictions(device, test_loader, net):
    """Plot the predictions of 1D regression tasks.

    Args:
        (....): See docstring of function :func:`main.test`.
    """
    net.eval()

    data = test_loader.dataset

    assert(data.inputs.shape[1] == 1 and data.outputs.shape[1] == 1)

    inputs = data.inputs.detach().cpu().numpy()
    targets = data.outputs.detach().cpu().numpy()
    
    with torch.no_grad():
        # Note, for simplicity, we assume that the dataset is small and we don't
        # have t collect the predictions by iterating over mini-batches.
        predictions = net.forward(data.inputs).detach().cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.title("Predictions in 1D regression task", size=20)

    plt.plot(inputs, targets, color='k', label='Target function',
             linestyle='dashed', linewidth=.5)
    plt.scatter(inputs, predictions, color='r', label='Predictions')

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


if __name__ == '__main__':
    pass


