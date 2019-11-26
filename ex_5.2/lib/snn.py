#!/usr/bin/env python3
# Copyright 2019 Maria Cervera

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

r"""Implementation of a spiking multilayer perceptron (:mod:`lib.snn`)
----------------------------------------------------------------------

The module :mod:`lib.snn` implements a fully-connected spiking neural network.

Internally, it will make use of ``Functions`` implemented in module
:mod:`lib.spiking_layer` to define the spiking dynamics of all layers of the
network.

.. autosummary::

    lib.snn.SNN
    lib.snn.SNN.forward

"""

import torch.nn as nn

from lib.spiking_layer import SpikingLayer
import lib.spiking_functions as sf


class SNN(nn.Module):
    r"""Implementation of a fully-connected spiking neural network.

    The :class:`SNN` is implemented as a :class:`torch.nn.Module`, 
    which is a convenient object for building neural networks, since ``Modules``
    can contain other ``Modules``.  ``Modules`` can be instantiated
    multiple times, as with multiple instances of the same type of layers.
    Submodules, (which are themselves ``Modules``) can
    easily be manipulated together as a whole ``Module``. For example, the
    :attr:`parameters` for a ``Module`` includes all the :attr:`parameters`
    attributes of its submodules which you can feed to the
    optimiser together. 

    The ``Module`` built here is a spiking neural network, constructed from
    layers of spiking neurons defined by in the :mod:`spiking_layer` script.

    Attributes:
        depth (int): Number of hidden layers.
        spiking_layers (torch.nn.ModuleList): A container for your spiking
            layers.

    Args:
        n_in (int): Network input size.
        n_out (int): Network output size.
        n_hidden (list): Size of each hidden layer of the network. This
            argument implicitly defines the :attr:`depth` of the network.
        args (argparse.Namespace): The command-line arguments.

    """

    def __init__(self, args, n_in=1, n_out=1, n_hidden=[10]):
        nn.Module.__init__(self)

        self._depth = len(n_hidden)
        n_all = [n_in] + n_hidden + [n_out]

        self.spiking_layers = nn.ModuleList()
        for i in range(1, len(n_all)):
            layer = SpikingLayer(n_all[i-1], n_all[i], args)
            self.spiking_layers.append(layer)

    @property
    def depth(self):
        r"""Getter for read-only attribute :attr:`depth`."""
        return self._depth

    def forward(self, x):
        r"""Compute the outputs :math:`y` of the network.

        Args:
            x (torch.Tensor): A tensor of shape 
                :math:`B \times t_{max} \times N`, where 
                :math:`B` is mini-batch size, :math:`t_{max}` is number of
                timesteps, and :math:`N` is the dimension of a flattened MNIST
                image (i.e. 784).

        Returns:
            (tuple): Tuple containing:

            - **U_layers** (list): A list of tensors of membrane potentials in
                    each layer(other than the input), each with shape 
                    :math:`B \times t_{max} \times M`, where 
                    :math:`B` is mini-batch size, :math:`t_{max}` is number of
                    timesteps, and :math:`M` is the number of neurons in the
                    layer.
            - **S_layers** (list): A list of tensors of spiking activities in 
                    each layer (other than the input), each
                    with shape :math:`B \times t_{max} \times M`,
                    where :math:`B` is mini-batch size, :math:`t_{max}` is 
                    number of timesteps, and :math:`M` is the number of 
                    neurons in the layer.

        """

        S = x
        U_layers, S_layers = [], []

        for l, layer in enumerate(self.spiking_layers):
            U, S = layer.forward(S)
            U_layers.append(U)  # membrane potentials
            S_layers.append(S)  # spikes

        return U_layers, S_layers

if __name__ == '__main__':
    pass
