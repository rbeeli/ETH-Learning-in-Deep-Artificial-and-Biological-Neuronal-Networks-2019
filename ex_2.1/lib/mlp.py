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
Implementation of a Multi-layer Perceptron (MLP)
------------------------------------------------

The module :mod:`lib.mlp` implements a simple fully-connected neural network,
a so called multi-layer perceptron (MLP).

Internally, it will make use of ``Functions`` implemented in module
:mod:`lib.backprop_functions` to realize fully-connected linear layers and
sigmoid activation functions.

.. autosummary::

    lib.mlp.MLP
"""
import torch.nn as nn

from lib.linear_layer import LinearLayer
import lib.backprop_functions as bf

class MLP(nn.Module):
    """Implementation of a fully-connected neural network with sigmoid non-
    linearities as activation functions after all hidden layers.

    Attributes:
        depth (int): Number of hidden layers.

    Args:
        n_in (int): Network input size.
        n_out (int): Network output size.
        n_hidden (list): Size of each hidden layer of the network. This
            argument implicitly defines the :attr:`depth` of the network.
    """
    def __init__(self, n_in=1, n_out=1, n_hidden=[10]):
        nn.Module.__init__(self)

        self._depth = len(n_hidden)

        n_all = [n_in] + n_hidden + [n_out]

        self._linear_layers = nn.ModuleList()
        for i in range(1, len(n_all)):
            self._linear_layers.append(LinearLayer(n_all[i-1], n_all[i]))

    @property
    def depth(self):
        """Getter for read-only attribute :attr:`depth`."""
        return self._depth

    def forward(self, x):
        """Compute the output :math:`y` of the network.

        After every linear hidden layer a sigmoid nonlinearity will be applied.

        .. note::
            The output of the network will be linear, i.e., no activation
            function is applied to the linear layer that connects the last
            hidden layer with the output layer.

        Args:
            x: The input to the network.

        Returns:
            The output ``y`` of the network.
        """
        y = x

        for i, layer in enumerate(self._linear_layers):
            y = layer.forward(y)
            if i < self.depth:
                y = bf.sigmoid_function(y)

        return y

if __name__ == '__main__':
    pass


