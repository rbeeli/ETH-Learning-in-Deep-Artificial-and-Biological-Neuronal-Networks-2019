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
A linear layer module that maintains its own parameters
-------------------------------------------------------

The module :mod:`lib.linear_layer` contains our own implementation of the
PyTorch class :class:`torch.nn.Linear`. The goal is to utilize the custom
``Functions`` implemented in module :mod:`lib.backprop_functions` (in
particular, the ``Function`` :class:`lib.backprop_functions.LinearFunction`)
and to provide a wrapper that takes care of managing the parameters
(:math:`W` and :math:`\mathbf{b}`) of such a linear layer.

.. note::
    PyTorch its :class:`torch.autograd.Function` interface is stateless.
    Therefore, the wrapper provided in this module is necessary in order to
    obtain a convinient interface for linear layers, such that the user doesn't
    have to maintain the parameters manually.

.. autosummary::

    lib.linear_layer.LinearLayer
"""
import torch
import torch.nn as nn

import lib.backprop_functions as bf

class LinearLayer(nn.Module):
    """Wrapper for ``Function`` :class:`lib.backprop_functions.LinearFunction`.

    The interface is inspired by the implementation of class
    :class:`torch.nn.Linear`.

    Attributes:
        weights (torch.nn.Parameter): The weight matrix :math:`W` of the layer.
        bias (torch.nn.Parameter): The bias vector :math:`\mathbf{b}` of the
            layer. Attribute is ``None`` if argument ``bias`` was passed as
            ``None`` in the constructor.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If set to ``False``, the layer will not learn an additive
            bias. 
    """
    def __init__(self, in_features, out_features, bias=True):
        nn.Module.__init__(self)

        self._weights = nn.Parameter(torch.Tensor(out_features, in_features),
                                     requires_grad=True)
        # Initialize weight matrix. Note, the `torch.nn.Linear` implementation
        # uses `nn.init.kaiming_uniform_` for initialization.
        nn.init.xavier_uniform_(self._weights)

        if bias:
            self._bias = nn.Parameter(torch.Tensor(out_features),
                                      requires_grad=True)
            torch.nn.init.constant_(self._bias, 0)
        else:
            self._bias = None

    @property
    def weights(self):
        """Getter for read-only attribute :attr:`weights`."""
        return self._weights

    @property
    def bias(self):
        """Getter for read-only attribute :attr:`bias`."""
        return self._bias

    def forward(self, x):
        """Compute the output activation of a linear layer.

        This method simply applies the
        :class:`lib.backprop_functions.LinearFunction` ``Function`` using the
        internally maintained weights.

        Args:
            x: See description of argument ``A`` of method
                :meth:`lib.backprop_functions.LinearFunction.forward`.

        Returns:
            See return value of method
            :meth:`lib.backprop_functions.LinearFunction.forward`.
        """
        return bf.linear_function(x, self.weights, b=self.bias)

if __name__ == '__main__':
    pass


