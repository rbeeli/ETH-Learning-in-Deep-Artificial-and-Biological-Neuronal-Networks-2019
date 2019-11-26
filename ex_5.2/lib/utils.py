#!/usr/bin/env python3
# Copyright 2019 Christian Henning, Maria Cervera
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
r"""
A collection of helper functions (:mod:`lib.utils`)
---------------------------------------------------

The module :mod:`lib.utils` contains several general purpose utilities and
helper functions.

The functions :meth:`utils.current2firing_time` and
:meth:`utils.sparse_data_generator` are taken directly from Friedemann Zenke's
Spytorch tutorial:

    https://github.com/fzenke/spytorch/

.. autosummary::
    lib.utils
    lib.utils.current2firing_time
    lib.utils.sparse_data_generator
    lib.utils.plot_weight_hist

"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import os

def current2firing_time(x, tau=20, thr=0.2, tmax=1.0, epsilon=1e-7):
    r"""Converts MNIST pixel values to latency-coded spikes.

    Computes first firing time latency for a current input x assuming the
    charge time of a current based LIF neuron. Images to spikes using a spike
    latency code, i.e. the higher the input intensity, the earlier the first 
    spike will be fired.

    Args:
        x (numpy.ndarray): The "current" values for each pixel in each image.
            Shape: (samples, 784)

    Keyword args:
        tau (float): The membrane time constant of the LIF neuron to be charged
        thr (float): The firing threshold value
        tmax (int): The maximum time returned
        epsilon (float): A generic (small) epsilon > 0

    Returns:
        T (numpy.ndarray): Time to first spike for each "current" x.
            Shape: (samples, 784)

    """

    idx = x<thr 
    x = np.clip(x,thr+epsilon,1e9) 
    T = tau*np.log(x/(x-thr))  
    T[idx] = tmax 
    return T
 

def sparse_data_generator(x, y, args, shuffle=True):
    r"""A generator that takes mini-batches in analog format and transforms 
    them to spike trains as sparse tensors.

    Args:
        x: The data ( sample x event x 2 ) the last dim holds (time,neuron) 
            tuples, (samples, 28 x 28)
        y: The labels (samples,)
        args (argparse.Namespace): The command-line arguments.
        shuffle (boolean): Whether batches should be shuffled.

    Yields:
        (tuple): Tuple containing:

            - **X_batch**: Spiking mini-batch.
            - **y_batch**: Target classes for the current mini-batch.

    """

    labels_ = np.array(y,dtype=np.int)
    number_of_batches = len(x)//args.batch_size
    sample_index = np.arange(len(x))

    # compute discrete firing times
    tau_eff = 20e-3/args.delta_t
    firing_times = np.array(current2firing_time(x, tau=tau_eff, \
                                            tmax=args.t_max), dtype=np.int)
    n_in = x.shape[1]
    unit_numbers = np.arange(n_in)

    if shuffle:
        np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    while counter<number_of_batches:
        batch_index = sample_index[args.batch_size*counter: \
                                                    args.batch_size*(counter+1)]

        coo = [ [] for i in range(3) ]
        for bc,idx in enumerate(batch_index): 
            c = firing_times[idx]<args.t_max
            times, units = firing_times[idx][c], unit_numbers[c]

            batch = [bc for _ in range(len(times))] 
            coo[0].extend(batch) # aggregates all image id for each spike
            coo[1].extend(times) # aggregates all spike times
            coo[2].extend(units) # aggregates all ids for neurons that spike

        i = torch.LongTensor(coo)
        v = torch.FloatTensor(np.ones(len(coo[0])))

        X_batch = torch.sparse.FloatTensor(i, v, \
                         torch.Size([args.batch_size,args.t_max,n_in]))
        y_batch = torch.tensor(labels_[batch_index]) 

        yield X_batch, y_batch

        counter += 1


def load_MNIST(path='data/'):
    r"""(Down)Loads data for a classification task on MNIST images.

    The Torchvision library provides methods to load the MNIST dataset from a
    local directory if the data have been downloaded previously, or to
    download and load the data if they cannot be found locally.

    The data are split into train and test sets and are preprocessed to convert
    the pixel values from [0,255] to [0,1]. Each 28 x 28 image is flattened
    to a 784 vector.

    Args:
        path (str): The path to the local MNIST dataset or the directory into
            which the MNIST dataset will be downloaded.

    Returns:
        (tuple): Tuple containing:

            - **train_x**: Training set images.
            - **test_x**:  Test set images.
            - **train_y**: Training set labels.
            - **test_y**:  Test set labels.

    """

    # Create data folder if necessary
    if not os.path.exists(path):
        os.mkdir(path)

    train_dataset = torchvision.datasets.MNIST(path, train=True, 
                        transform=None, target_transform=None, download=True)
    test_dataset  = torchvision.datasets.MNIST(path, train=False, 
                        transform=None, target_transform=None, download=True)
    # Transform and standardize data
    train_x = np.array(train_dataset.data, dtype=np.float)
    train_x = train_x.reshape(train_x.shape[0],-1)/255
    train_y = np.array(train_dataset.targets, dtype=np.int)

    test_x = np.array(test_dataset.data, dtype=np.float)
    test_x = test_x.reshape(test_x.shape[0],-1)/255
    test_y  = np.array(test_dataset.targets, dtype=np.int)

    return train_x, test_x, train_y, test_y


def plot_weight_hist(parameters, initial_weights):
    """Plot histogram of the initial and trained weights in each layer.

    For each layer, a different subplot with overlapping weight distributions
    before and after training will be shown.

    Args:
        parameters: The set of weights after training.
        initial_weights (list): The set of initial weights.
    """

    trained_weights = []
    for weights in parameters:
        trained_weights.append(weights.clone().data.cpu().numpy())

    plt.figure(figsize=(12,5))
    for layer in range(len(trained_weights)):
        plt.subplot(1, len(trained_weights), layer+1)
        plt.hist(initial_weights[layer].reshape(-1), alpha=0.5, label='initial')
        plt.hist(trained_weights[layer].reshape(-1), alpha=0.5, label='trained')
        if layer == 0:
            if len(trained_weights)==1:
                plt.title(r'$w_{inputs \rightarrow output}$', fontsize=15)
            else:
                plt.title(r'$w_{inputs \rightarrow h1}$', fontsize=15)
            plt.ylabel('number of weights', fontsize=12)
        if layer == len(trained_weights)-1 and len(trained_weights)>1:
            plt.title(r'$w_{h%i \rightarrow output}$'%layer, fontsize=15)
        plt.tick_params(which='both', top=False, left=False, right=False,  
            labelleft=False) 
        plt.xlabel('weight values')
    plt.legend(fontsize=12)
    plt.show()
