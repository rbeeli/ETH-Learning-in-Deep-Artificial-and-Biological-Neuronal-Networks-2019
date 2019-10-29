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
import torch.nn.functional as F

from lib.utils import sampleGaussian

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

        self._w_mu = nn.Parameter(torch.Tensor(out_features, in_features),
                                     requires_grad=True)
        self._w_logvar = nn.Parameter(torch.Tensor(out_features, in_features),
                                     requires_grad=True)
        # Initialize weight matrix. Note, the `torch.nn.Linear` implementation
        # uses `nn.init.kaiming_uniform_` for initialization.
        nn.init.xavier_uniform_(self._w_mu)
        nn.init.uniform_(self._w_logvar, -5,-4)

        if bias:
            self._b_mu = nn.Parameter(torch.Tensor(out_features),
                                      requires_grad=True)
            self._b_logvar = nn.Parameter(torch.Tensor(out_features),
                                      requires_grad=True)
            torch.nn.init.constant_(self._b_mu, 0.0)
            torch.nn.init.uniform_(self._b_logvar, -5,-4)
        else:
            self._b_mu = None
            self._b_logvar = None

    @property
    def w_mu(self):
        """Getter for read-only attribute :attr:`w_mu`."""
        return self._w_mu

    @property
    def w_logvar(self):
        """Getter for read-only attribute :attr:`w_sigma`."""
        return self._w_logvar

    @property
    def b_mu(self):
        """Getter for read-only attribute :attr:`b_mu`."""
        return self._b_mu

    @property
    def b_logvar(self):
        """Getter for read-only attribute :attr:`b_sigma`."""
        return self._b_logvar

    def forward(self, x, bbb=False):
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

        if bbb:
            cur_weights = sampleGaussian(self.w_mu, self.w_logvar)
            cur_bias = sampleGaussian(self.b_mu, self.b_logvar)
        else:
            cur_weights = self.w_mu
            cur_bias = self.b_mu
        return F.linear(x, cur_weights, bias=cur_bias) 

if __name__ == '__main__':
    pass


