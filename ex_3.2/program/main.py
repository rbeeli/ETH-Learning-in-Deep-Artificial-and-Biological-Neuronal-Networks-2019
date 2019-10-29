#!/usr/bin/env python3
# Copyright 2019 Christian Henning, Alexander Meulemans
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
Controller for simulations
--------------------------

The module :mod:`main` is an executable script that controls the simulations
(i.e., the training of regression tasks).

For more usage information, please check out

.. code-block:: console

  $ python3 main --help

There are is one dataset we want to train on:


1D polynomial regression
^^^^^^^^^^^^^^^^^^^^^^^^

In case of 1D regression tasks, the result can be visualized to a human, who
can assess the quality of the optimizer.

An example 1D regression task can be retrieved using the function
:func:`lib.utils.regression_cubic_poly`.

In this exercise, we want to understand the difference between standart
maximum likelihood and a bayesian treatment of learning. For the latter, we need
to approximate the intractable weight posterior to do inference. 
First, we take a closer look at:

Maximum likelihood
^^^^^^^^^^^^^^^^^^

Juhuu! 
For this part, you don't have to programm anything to make the script work.

In the case of maximum likelihodd, we train a standart neural network 
with backpropagation to minimize a certain loss. We obtain one set of weights
that supposedly solves the problem.

To visualize and run the training, execute 

.. code-block:: console

  $ python3 main --show_plot
  
This script can be used to train on this dataset via the option
option.

.. code-block:: console

  $ python main.py --show_plot --data_random_seed 753

Furthermore, you can train on this dataset also via the option
``--random_seed``.

.. code-block:: console

  $ python main.py --show_plot --random_seed 656

These random seed options change the generation of the training data a little
or how your initial weights, the training starting points, are determined. 

You can think of these random seeds, as specification how and when data was 
collected and in what state your algorithm was when we started training. 
(One could even imagine that the algorithm was already trained on another task.)

Oberserve the difference when you change one/both of the seeds (753, 656 above) 
to other numbers and describe it and the possible shortcomings of 
maximum likelihood (TODO).


Bayes-by-Backprop
^^^^^^^^^^^^^^^^^^

In the case of bayes-by-Backprop, to run the training, execute 

.. code-block:: console

  $ python3 main --bbb

Add to visualise ``--random_seed``.


.. autosummary::

    main.run
    main.train
"""
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader

from lib.mlp import MLP
from lib import utils

def test(device, test_loader, net):
    """test a train network by computing the MSE on the test set.

    Args:
        (....): See docstring of function :func:`train`.
        test_loader (torch.utils.data.DataLoader): The data handler for
            test data.

    Returns:
        (float): The mean-squared error for the test set ``test_loader`` when
        using the network ``net``. Note, the ``Function``
        :func:`lib.backprop_functions.mse_loss` is used to compute the MSE
        value.
    """
    #######################################################################
    ### NOTE, the function `mse_loss` divides by the current batch size. In
    ### order to compute the MSE across several mini-batches, one needs to
    ### correct for this behavior.
    #######################################################################

    net.eval()

    with torch.no_grad():
        num_samples = 0
        mse = 0.

        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            batch_size = int(inputs.shape[0])
            num_samples += batch_size

            predictions = net.forward(inputs)

            mse += 0.5*F.mse_loss(predictions, targets)

        mse /= num_samples

    print('Test MSE: {}'.format(mse))

    return float(mse.cpu().detach().numpy())

def train(args, device, train_loader, net):
    """Train the given network on the given (regression) dataset.

    Args:
        args (argparse.Namespace): The command-line arguments.
        device: The PyTorch device to be used.
        train_loader (torch.utils.data.DataLoader): The data handler for
            training data.
        net: The neural network.
    """
    print('Training network ...')
    net.train()

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                momentum=args.momentum)

    for e in range(args.epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            predictions = net.forward(inputs, args.bbb)
            optimizer.zero_grad()

            loss = 0.5*F.mse_loss(predictions, targets, reduction='mean')
            
            if args.bbb:
                
                nll, kl = utils.computeELBO(net, predictions, targets, device, 
                                                                          args)
                loss = nll + kl
            else:
                kl = 0
            loss.backward()
            optimizer.step()

        if e % 500 == 0:
            print('Epoch {} -- loss = {}.'.format(e+1, loss - kl))
            if args.bbb:
                print('Epoch {} -- prior-matching-loss = {}.'.format(e+1, kl))
    print('Training network ... Done')

def run():
    """Run the script.

    - Parsing command-line arguments
    - Creating synthetic regression data
    - Initiating training process
    - Testing final network
    """
    ### Parse CLI arguments.
    parser = argparse.ArgumentParser(description='Nonlinear regression with ' +
                                     'neural networks.')

    tgroup = parser.add_argument_group('Training options')
    tgroup.add_argument('--epochs', type=int, metavar='N', default=10000,
                        help='Number of training epochs. ' +
                             'Default: %(default)s.')
    tgroup.add_argument('--batch_size', type=int, metavar='N', default=128,
                        help='Training batch size. Default: %(default)s.')
    tgroup.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate of optimizer. Default: ' +
                             '%(default)s.')
    tgroup.add_argument('--momentum', type=float, default=0.0,
                        help='Momentum of the optimizer. ' +
                             'Default: %(default)s.')
    sgroup = parser.add_argument_group('Network options')
    sgroup.add_argument('--num_hidden', type=int, metavar='N', default=2,
                        help='Number of hidden layer in the (student) ' +
                             'network. Default: %(default)s.')
    sgroup.add_argument('--size_hidden', type=int, metavar='N', default=10,
                        help='Number of units in each hidden layer of the ' +
                             '(student) network. Default: %(default)s.')
    sgroup.add_argument('--num_train_samples', type=int, default=20,
                        help='Number of data training points.')

    mgroup = parser.add_argument_group('Miscellaneous options')
    mgroup.add_argument('--use_cuda', action='store_true',
                        help='Flag to enable GPU usage.')
    mgroup.add_argument('--random_seed', type=int, metavar='N', default=42,
                        help='Random seed. Default: %(default)s.')
    mgroup.add_argument('--data_random_seed', type=int, metavar='N', default=42,
                        help='Data random seed. Default: %(default)s.')
    mgroup.add_argument('--dont_show_plot', action='store_false',
                        help='Dont show the final regression results as plot.' +
                             'Note, only applies to 1D regression tasks.')

    bgroup = parser.add_argument_group('Bayes by Backprop options')
    bgroup.add_argument('--bbb', action='store_true',
                        help='Start training of BbB.')
    bgroup.add_argument('--prior_variance', type=float, default=1.0,
                        help='Variance of the Gaussian prior.')
    bgroup.add_argument('--weight_samples', type=int, default=100,
                        help='Number of weight samples used.')
                        

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
    print('### Learning to regress a 1D cubic polynomial ###')
    n_in = n_out = 1

    train_x, test_x, train_y, test_y = utils.regression_cubic_poly(rseed= \
                        args.data_random_seed, num_train=args.num_train_samples)

    train_loader = DataLoader(utils.RegressionDataset(train_x, train_y),
                              batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(utils.RegressionDataset(test_x, test_y),
                             batch_size=args.batch_size, shuffle=False)

    ### Generate network.
    n_hidden = [args.size_hidden] * args.num_hidden
    net = MLP(n_in=n_in, n_out=n_out, n_hidden=n_hidden).to(device)

    ### Train network.
    train(args, device, train_loader, net)

    ### Test network.
    test(device, test_loader, net)

    if args.dont_show_plot and n_in == 1 and n_out == 1:
        utils.plot_predictions(device, test_loader, train_loader, net, args)

if __name__ == '__main__':
    run()


