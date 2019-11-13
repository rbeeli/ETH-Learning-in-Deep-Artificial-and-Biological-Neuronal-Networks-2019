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
Controller for simulations (:mod:`main`)
----------------------------------------

The module :mod:`main` is an executable script that controls the simulations
(i.e., the training and testing of MNIST digit classification tasks).

For more usage information, check out:

.. code-block:: console

  $ python3 main.py --help

.. autosummary::
    main.train
    main.test
    main.run

"""

import argparse
import numpy as np
import random
import torch

import lib.spiking_functions as sf
from lib.snn import SNN
from lib import utils

def train(args, device, x, y, net):
    r"""Trains the given network on the MNIST dataset.

    The :mod:`main.train` method takes data (x, y) and a spiking neural net,
    puts the net in training mode, and sets up the optimiser. Then, for each
    epoch, it runs through the whole MNIST dataset once, updating the weights
    once every mini-batch, after the images in this mini-batch have been
    converted to spike trains. 
    Note, the ``Function``
    :func:`lib.spiking_functions.loss_on_voltage` is used to compute 
    the loss.

    Args:
        args (argparse.Namespace): The command-line arguments.
        device (torch.device): The PyTorch device to be used.
        x (torch.Tensor): The training inputs.
        y (torch.Tensor): The training targets.
        net (lib.snn.SNN): The spiking neural network.
    """

    print('Training network ...')
    net.train()  # Puts the SNN in training mode

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9,0.99))

    for e in range(args.epochs):
        for inputs, targets in utils.sparse_data_generator(x, y, args):
            inputs, targets = inputs.to(device), targets.to(device)

            voltage, spikes = net.forward(inputs.to_dense())

            output_voltage = voltage[-1]
            output_spikes  = spikes[-1]

            optimizer.zero_grad()

            loss = sf.loss_on_voltage(output_voltage, targets)

            loss.backward()
            optimizer.step()

        print('Epoch %i -- loss = %.3f.'%(e+1, loss))

    print('Training network ... Done')


def test(args, device, x, y, net):
    r"""Tests a trained network by computing the classification accuracy on the 
    test set.

    Args:
        (....): See docstring of function :func:`train`.
        x (torch.Tensor): The testing inputs.
        y (torch.Tensor): The testing targets.

    Returns:
        (float): The classification accuracy for the
        test data (x, y) when using the network ``net``.
        Note, the ``Function``
        :func:`lib.spiking_functions.accuracy_on_voltage` is used to compute 
        the accuracy.
    """

    net.eval()

    with torch.no_grad():
        num_samples = 0
        accu = 0.

        for inputs, targets in utils.sparse_data_generator(x, y, args):
            inputs, targets = inputs.to(device), targets.to(device)

            batch_size = int(inputs.shape[0])
            num_samples += batch_size

            voltage, spikes = net.forward(inputs.to_dense())

            output_voltage = voltage[-1]
            output_spikes = spikes[-1]

            batch_accu = sf.accuracy_on_voltage(output_voltage, targets)
            accu += batch_size * batch_accu

        accu /= num_samples

    print('Test accuracy: %.2f%%.'%(accu * 100))

    return float(accu.cpu().detach().numpy())

def run():
    r"""Runs the script.

    The :mod:`main.run` method performs the following actions:

    - Parses command-line arguments
    - Sets random seeds to ensure deterministic computation
    - Loads MNIST dataset
    - Initiates training process
    - Tests accuracy of final network
    - Plots weight histograms if required
    """

    ### Parse CLI arguments.
    parser = argparse.ArgumentParser(description='MNIST classification with ' +
                                     'spiking neural networks.')

    dgroup = parser.add_argument_group('Neuronal dynamics options')
    dgroup.add_argument('--tau_mem', type=float, default=10e-3,
                        help='Membrane time constant (in s). Default: ' +
                             '%(default)s.')
    dgroup.add_argument('--tau_syn', type=float, default=12e-3,
                        help='Synaptic time constant (in s). Default: ' +
                             '%(default)s.')
    dgroup.add_argument('--tau_rise', type=float, default=1e-3,
                        help='Synaptic rise time constant (in s). Default: ' +
                             '%(default)s.')
    dgroup.add_argument('--u_rest', type=float, default=0,
                        help='Resting membrane potential (in volts). ' +
                             'Default: %(default)s.')
    dgroup.add_argument('--u_threshold', type=float, default=1,
                        help='Threshold voltage for spike generation (in ' +
                             'volts). Default: %(default)s.')
    dgroup.add_argument('--R', type=float, default=1,
                        help='Membrane resistance (in ohms). ' +
                             'Default: %(default)s.')

    tgroup = parser.add_argument_group('Training options')
    tgroup.add_argument('--epochs', type=int, metavar='N', default=2,
                        help='Number of training epochs. ' +
                             'Default: %(default)s.')
    tgroup.add_argument('--batch_size', type=int, metavar='N', default=256,
                        help='Training batch size. Default: %(default)s.')
    tgroup.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate of optimizer. Default: ' +
                             '%(default)s.')

    ngroup = parser.add_argument_group('Network options')
    ngroup.add_argument('--num_hidden', type=int, default=1,
                        help='Number of hidden layers in the network. ' +
                             'Default: %(default)s.')
    ngroup.add_argument('--size_hidden', type=int, default=100,
                        help='Number of units in each hidden layer of the ' +
                             'network. Default: %(default)s.')
    ngroup.add_argument('--weight_scale', type=float, default=0.2,
                        help='Scale for the initialization of the weights. ' +
                             'Default: %(default)s.')

    mgroup = parser.add_argument_group('Miscellaneous options')
    mgroup.add_argument('--delta_t', type=float, default=1e-3,
                        help='Time step size (in s). Default: %(default)s.')
    mgroup.add_argument('--t_max', type=int, default=100,
                        help='Number of time steps used for each sample. ' +
                             'Default: %(default)s.')
    mgroup.add_argument('--use_cuda', action='store_true',
                        help='Flag to enable GPU usage.')
    mgroup.add_argument('--random_seed', type=int, metavar='N', default=42,
                        help='Random seed. Default: %(default)s.')
    mgroup.add_argument('--plot_weight_hist', action='store_true',
                        help='Whether histograms of the weights before and ' +
                             'after learning should be plotted.')

    args = parser.parse_args()

    ### Ensure deterministic computation.
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Ensure that runs are reproducible even on GPU. Note, this slows down
    # training!
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using cuda: ' + str(use_cuda))

    ### Generate datasets and data handlers.
    print('### Learning to classify MNIST digits with a spiking network ###')
    n_in = 784
    n_out = 10

    train_x, test_x, train_y, test_y = utils.load_MNIST()

    ### Generate network.
    n_hidden = [args.size_hidden] * args.num_hidden
    net = SNN(args, n_in=n_in, n_out=n_out, n_hidden=n_hidden).to(device)

    ### Store initial weights.
    initial_weights = []
    for weights in net.parameters():
        initial_weights.append(weights.clone().data.cpu().numpy())

    ### Train network.
    train(args, device, train_x, train_y, net)

    ### Test network.
    accuracy = test(args, device, test_x, test_y, net)

    ### Plot weight histograms if asked to.
    if args.plot_weight_hist:
        utils.plot_weight_hist(net.parameters(), initial_weights)


if __name__ == '__main__':
    run()
