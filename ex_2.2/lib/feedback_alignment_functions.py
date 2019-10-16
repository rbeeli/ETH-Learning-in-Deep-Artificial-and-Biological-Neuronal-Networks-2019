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
Adding custom functions for feedback alignment to PyTorch's autograd
--------------------------------------------------------------------
.. note::
    The forward and backward functions for a linear layer, sigmoid layer
    and MSE loss were already implemented in tutorial 2.1. In this tutorial,
    we will make a new class for a linear layer that applies feedback alignment
    for its backward function. Most of the code for the forward and backward
    function of the normal linear layer can be reused.


The module :mod:`lib.feedback_alingment_functions` contains custom
implementations of the linear layers used in the neural network,
that are compatible with PyTorch its autograd_ package.

Please checkout the corresponding documentation of class
:class:`torch.autograd.Function`.

.. autosummary::

    lib.feedback_alignment_functions.LinearFunctionFA
    lib.feedback_alignment_functions.linear_function_fa

.. _autograd:
    https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
"""

import torch
from torch.autograd import Function

class LinearFunctionFA(Function):
    r"""Implementation of a fully-connected layer w/o activation function that
    will use feedback alignment (FA) as a training method.

    This class is a ``Function`` that behaves similar to PyTorch's class
    :class:`torch.nn.Linear`, but it has a different backward function that
    implements FA. Since this class implements the interface
    :class:`torch.autograd.Function`, we can use it to specify a custom
    backpropagation behavior.

    Assuming column vectors: layer input :math:`\mathbf{a} \in \mathbb{R}^M`,
    bias vector :math:`\mathbf{b} \in \mathbb{R}^N` and a weight matrix
    :math:`W \in \mathbb{R}^{N \times M}`, this layer simply computes

    .. math::
        :label: eq-single-sample-fa

        \mathbf{z} = W \mathbf{a} + \mathbf{b}

    (or :math:`\mathbf{z} = W \mathbf{a}` if :math:`\mathbf{b}` is ``None``).

    Note, since we want to process mini-batches (containing :math:`B` samples
    each), the input to the :meth:`forward` method is actually a set of samples
    :math:`\mathbf{a}` collected into a matrix
    :math:`A \in \mathbb{R}^{B \times M}`.

    We additionally introduce the matrix
    :math:`\tilde{B} \in \mathbb{R}^{B \times N}` which simply contains copies
    of the bias vector :math:`\mathbf{b}` in its rows.

    The mathematical operation described for single samples in eq.
    :eq:`eq-single-sample-fa`, is stated for a the case of mini-batches below

    .. math::
        :label: eq-mini-batch-fa

        Z = A W^T + \tilde{B}

    where :math:`Z \in \mathbb{R}^{B \times N}` is the output matrix.
    """
    @staticmethod
    def forward(ctx, A, W, B, b=None):
        r"""Compute the output of a linear layer.

        This method implements eq. :eq:`eq-mini-batch-fa`.

        Args:
            ctx: A context. Should be used to store activations which are needed
                in the backward pass.
            A: A mini-batch of input activations :math:`A`.
            W: The weight matrix :math:`W`.
            B: The random feedback weight matrix :math:`B`. This matrix is not
                used in the forward pass, but will be saved in ctx for the
                backward pass. Note that :math:`B` needs to have the same
                dimensions as :math:`W^T`
            b (optional): The bias vector :math:`\mathbf{b}`.

        Returns:
            The output activations :math:`Z` as defined by eq.
            :eq:`eq-mini-batch-fa`.
        """
        ctx.save_for_backward(A, W, B, b)

        Z = A.mm(W.t()) + b

        return Z
        # Solution inspired by:
        # https://pytorch.org/docs/master/notes/extending.html

    @staticmethod
    def backward(ctx, grad_Z):
        r"""propagate the error signals for :math:`Z` through this linear layer,
        according to the feedback alignment method.

        The matrix ``grad_Z``, which we denote by
        :math:`\delta_Z \in \mathbb{R}^{B \times N}`, contains the error signals
        of the scalar loss function with respect to each element
        from the :meth:`forward` output matrix :math:`Z`, propagated by
        feedback alignment.

        This method propagates the global error (encoded in
        :math:`\delta_Z`) to all input tensors of the :meth:`forward` method,
        essentially computing :math:`\delta_A`, :math:`\delta_W`,
        :math:`\delta_\mathbf{b}`.

        As shown in the tutorial, these partial derivatives can be computed as
        follows:

        .. math::

            \delta_A &= \delta_Z B^T \\
            \delta_W &= \delta_Z^T A \\
            \delta_\mathbf{b} &= \sum_{b=1}^B \delta_{Z_{b,:}}

        where :math:`\delta_{Z_{b,:}}` denotes the vector retrieved from the
        :math:`b`-th row of :math:`\delta_Z`.

        Args:
            ctx: See description of argument ``ctx`` of method :meth:`forward`.
            grad_Z: The propagated error :math:`\delta_Z`.

        Returns:
            (tuple): Tuple containing:

            - **grad_A**: The propagated error signal according to
                feedback alignment, i.e., :math:`\delta_A`.
            - **grad_W**: The direction for the weight update
              , i.e., :math:`\delta_W`.
            - **grad_B**: This output will always be ``None``, but needs to be
                there for complying with the Pytorch framework.
            - **grad_b**: The direction for the bias update
                i.e., :math:`\delta_\mathbf{b}`; or ``None`` if ``b`` was
                passed as ``None`` to the :meth:`forward` method.

            .. note::
                Error signals/update directions for input tensors
                are only computed if their keyword
                ``requires_grad`` is set to ``True``, otherwise ``None`` is
                returned for the corresponding Tensor.
        """
        A, W, B, b = ctx.saved_tensors

        grad_A = None
        grad_W = None
        grad_B = None
        grad_b = None

        # We only need to compute gradients for tensors that are flagged to
        # require gradients!
        if ctx.needs_input_grad[0]:
            grad_A = grad_Z.mm(B.t())

        if ctx.needs_input_grad[1]:
            grad_W = grad_Z.t().mm(A)

        if b is not None and ctx.needs_input_grad[3]:
            grad_b = grad_Z.sum(0)

        return grad_A, grad_W, grad_B, grad_b

def linear_function_fa(A, W, B, b=None):
    """An alias for using class :class:`LinearFunctionFA`.

    Args:
        (....): See docstring of method :meth:`LinearFunctionFA.forward`.
    """
    # Note, `apply()` doesn't allow keyword arguments, which is why we build
    # this wrapper.
    if b is None:
        return LinearFunctionFA.apply(A, W, B)
    else:
        return LinearFunctionFA.apply(A, W, B, b)

if __name__ == '__main__':
    pass


