#!/usr/bin/env python3
# Copyright 2019 Alexander Meulemans
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
A linear layer module for FA that maintains its own parameters
--------------------------------------------------------------

The module :mod:`lib.linear_layer_fa` contains our own implementation of the
PyTorch class :class:`torch.nn.Linear` with a modified backwards function that
implements feedback alignment (FA). The goal is to utilize the custom
``Functions`` implemented in module :mod:`lib.feedback_alignment_functions` (in
particular, the ``Function``
:class:`lib.feedback_alignment_functions.LinearFunctionFA`)
and to provide a wrapper that takes care of managing the parameters
(:math:`W` and :math:`\mathbf{b}`) of such a linear layer.

.. note::
    PyTorch its :class:`torch.autograd.Function` interface is stateless.
    Therefore, the wrapper provided in this module is necessary in order to
    obtain a convenient interface for linear layers, such that the user doesn't
    have to maintain the parameters manually.

.. autosummary::

    lib.linear_layer_fa.LinearLayerFA
"""
import torch
import torch.nn as nn

import lib.backprop_functions as bf
import lib.feedback_alignment_functions as fa

class LinearLayerFA(nn.Module):
    """Wrapper for ``Function``
    :class:`lib.feedback_alignment_functions.LinearFunctionFA`.

    The interface is inspired by the implementation of class
    :class:`torch.nn.Linear`.

    Attributes:
        weights (torch.nn.Parameter): The weight matrix :math:`W` of the layer.
        bias (torch.nn.Parameter): The bias vector :math:`\mathbf{b}` of the
            layer. Attribute is ``None`` if argument ``bias`` was passed as
            ``None`` in the constructor.
        feedbackweights (torch.nn.Parameter): The random feedback weight matrix
            :math:`B` of the layer.
    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If set to ``False``, the layer will not learn an additive
            bias.
        gain (float): A scaling term used for initializing the forward weights.
            Default 0.01.
    """
    def __init__(self, in_features, out_features, bias=True, gain=0.01):
        nn.Module.__init__(self)

        self._weights = nn.Parameter(torch.Tensor(out_features, in_features),
                                     requires_grad=True)
        # Note, the random feedback weights should have the same dimensions as
        # the transpose of the forward weights
        self._feedbackweights = nn.Parameter(torch.Tensor(in_features,
                                                          out_features),
                                     requires_grad=False)

        # Initialize weight matrices. Note, the `torch.nn.Linear` implementation
        # uses `nn.init.kaiming_uniform_` for initialization.
        nn.init.xavier_uniform_(self._weights, gain=gain)
        nn.init.xavier_uniform_(self._feedbackweights, gain=0.5)

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

    @property
    def feedbackweights(self):
        """Getter for read-only attribute :attr:`feedbackweights`."""
        return self._feedbackweights

    def forward(self, x):
        """Compute the output activation of a linear layer.

        This method simply applies the
        :class:`lib.feedback_alignment_functions.LinearFunctionFA`
        ``Function`` using the internally maintained weights.

        Args:
            x: See description of argument ``A`` of method
                :meth:`lib.backprop_functions.LinearFunction.forward`.

        Returns:
            See return value of method
            :meth:`lib.backprop_functions.LinearFunction.forward`.

        """

        return fa.linear_function_fa(x, self.weights,
                                     self.feedbackweights,
                                     b=self.bias)

if __name__ == '__main__':
    pass


