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
Adding custom functions to PyTorch's autograd
---------------------------------------------

The module :mod:`lib.backprop_functions` contains custom implementations of
neural network components (layers, activation functions, loss functions, ...),
that are compatible with PyTorch its autograd_ package.

A new functionality can be added to autograd_ by creating a subclass of class
:class:`torch.autograd.Function`. In particular, we have to implement the
:meth:`torch.autograd.Function.forward` method (which computes the output of a
differentiable function) and the :meth:`torch.autograd.Function.backward`
method (which computes the partial derivatives of the output of the implemented
:meth:`torch.autograd.Function.forward` method with respect to all input tensors
that are flagged to require gradients).

Please checkout the corresponding documentation of class
:class:`torch.autograd.Function`.

.. autosummary::

    lib.backprop_functions.LinearFunction
    lib.backprop_functions.linear_function
    lib.backprop_functions.SigmoidFunction
    lib.backprop_functions.sigmoid_function
    lib.backprop_functions.MSELossFunction
    lib.backprop_functions.mse_loss

.. _autograd:
    https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
"""
import torch
from torch.autograd import Function

class LinearFunction(Function):
    r"""Implementation of a fully-connected layer w/o activation function.

    This class is a ``Function`` that behaves just like PyTorch's class
    :class:`torch.nn.Linear`. Since this class implements the interface
    :class:`torch.autograd.Function`, we can use it to specify a custom
    backpropagation behavior.

    In this specific case, the ``Function`` shall behave just as in classic
    backpropagation (i.e., it shall behave identical to the proprietory PyTorch
    implementation).

    Assuming column vectors: layer input :math:`\mathbf{a} \in \mathbb{R}^M`,
    bias vector :math:`\mathbf{b} \in \mathbb{R}^N` and a weight matrix
    :math:`W \in \mathbb{R}^{N \times M}`, this layer simply computes

    .. math::
        :label: eq-single-sample

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
    :eq:`eq-single-sample`, is stated for a the case of mini-batches below

    .. math::
        :label: eq-mini-batch

        Z = A W^T + \tilde{B}

    where :math:`Z \in \mathbb{R}^{B \times N}` is the output matrix.
    """
    @staticmethod
    def forward(ctx, A, W, b=None):
        r"""Compute the output of a linear layer.

        This method implements eq. :eq:`eq-mini-batch`.

        Args:
            ctx: A context. Should be used to store activations which are needed
                in the backward pass.
            A: A mini-batch of input activations :math:`A`.
            W: The weight matrix :math:`W`.
            b (optional): The bias vector :math:`\mathbf{b}`.

        Returns:
            The output activations :math:`Z` as defined by eq.
            :eq:`eq-mini-batch`.
        """
        ctx.save_for_backward(A, W, b)

        Z = A.mm(W.t()) + b

        return Z

    @staticmethod
    def backward(ctx, grad_Z):
        r"""Backpropagate the gradients of :math:`Z` through this linear layer.

        The matrix ``grad_Z``, which we denote by
        :math:`\delta_Z \in \mathbb{R}^{B \times N}`, contains the partial
        derivatives of the scalar loss function with respect to each element
        from the :meth:`forward` output matrix :math:`Z`.

        This method backpropagates the global error (encoded in
        :math:`\delta_Z`) to all input tensors of the :meth:`forward` method,
        essentially computing :math:`\delta_A`, :math:`\delta_W`,
        :math:`\delta_\mathbf{b}`.

        As shown in the tutorial, these partial derivatives can be computed as
        follows:

        .. math::

            \delta_A &= \delta_Z W \\
            \delta_W &= \delta_Z^T A \\
            \delta_\mathbf{b} &= \sum_{b=1}^B \delta_{Z_{b,:}}

        where :math:`\delta_{Z_{b,:}}` denotes the vector retrieved from the
        :math:`b`-th row of :math:`\delta_Z`.

        Args:
            ctx: See description of argument ``ctx`` of method :meth:`forward`.
            grad_Z: The backpropagated error :math:`\delta_Z`.

        Returns:
            (tuple): Tuple containing:

            - **grad_A**: The derivative of the loss with respect to the input
              activations, i.e., :math:`\delta_A`.
            - **grad_W**: The derivative of the loss with respect to the weight
              matrix, i.e., :math:`\delta_W`.
            - **grad_b**: The derivative of the loss with respect to the bias
              vector, i.e., :math:`\delta_\mathbf{b}`; or ``None`` if ``b`` was
              passed as ``None`` to the :meth:`forward` method.

            .. note::
                Gradients for input tensors are only computed if their keyword
                ``requires_grad`` is set to ``True``, otherwise ``None`` is
                returned for the corresponding Tensor.
        """
        A, W, b = ctx.saved_tensors

        grad_A = None
        grad_W = None
        grad_b = None

        # We only need to compute gradients for tensors that are flagged to
        # require gradients!
        if ctx.needs_input_grad[0]:
            grad_A = grad_Z.mm(W)

        if ctx.needs_input_grad[1]:
            grad_W = grad_Z.t().mm(A)

        if b is not None and ctx.needs_input_grad[2]:
            grad_b = grad_Z.sum(0)

        return grad_A, grad_W, grad_b

def linear_function(A, W, b=None):
    """An alias for using class :class:`LinearFunction`.

    Args:
        (....): See docstring of method :meth:`LinearFunction.forward`.
    """
    # Note, `apply()` doesn't allow keyword arguments, which is why we build
    # this wrapper.
    if b is None:
        return LinearFunction.apply(A, W)
    else:
        return LinearFunction.apply(A, W, b)

class SigmoidFunction(Function):
    r"""Implementation of a sigmoid non-linearity.

    This ``Function`` applies a sigmoid_ function as non-linearity to a vector
    of activations :math:`\mathbf{z}` (e.g., the output of a linear layer
    :class:`LinearFunction`).

    This function (denoted by :math:`\sigma(\cdot)`) operates element-wise when
    computing the output activations :math:`\mathbf{a}`:

    .. math::

        \mathbf{a} = \sigma(\mathbf{z})

    Similarly to the notation used in the docstring of class
    :class:`LinearFunction`, we consider a mini-batch of input activations
    given as a matrix :math:`Z` and denote the corresponding output activations
    by :math:`A`.

    Example:

        Consider the output activation :math:`A^{(l-1)}` of layer :math:`l-1`.
        Assuming a linear layer with sigmoid non-linearity, the output
        activations :math:`A^{(l)}` of layer :math:`l` are computed as

        .. math::

            A^{(l)} = \sigma(Z^{(l)}) = \sigma(A^{(l-1)} W^{(l), T} + b^{(l)})

    .. _sigmoid: https://en.wikipedia.org/wiki/Sigmoid_function
    """
    @staticmethod
    def forward(ctx, Z):
        r"""Applies the sigmoid function element-wise to ``Z``.

        Args:
            ctx: See description of argument ``ctx`` of method
                :meth:`LinearFunction.forward`.
            Z: The input activations to this non-linearity, i.e., the matrix
                :math:`Z`.

        Returns:
            The output activations :math:`A = \sigma(Z)`.
        """
        ctx.save_for_backward(Z)

        A = torch.sigmoid(Z)

        return A

    @staticmethod
    def backward(ctx, grad_A):
        r"""Backpropagate gradients through sigmoid-nonlinearity.

        In this method, we compute the backprop-error :math:`\delta_Z` given the
        error :math:`\delta_A`, which is the derivative of the global scalar
        loss with respect to the output of the :meth:`forward` method.

        As shown in the tutorial, this partial derivative can be computed as
        follows:

        .. math::

            \delta_Z &= \delta_A \, \text{diag} \big( \sigma^{'}(Z) \big) \\
            &= \delta_A \odot \sigma^{'}(Z)

        where the function :math:`\sigma^{'}(\cdot)` applies the derivative of
        the sigmoid non-linearity element-wise to its input tensor. The operator
        :math:`\odot` denotes the Hadamard_ product (element-wise product).
        
        .. _Hadamard: https://en.wikipedia.org/wiki/Hadamard_product_(matrices)

        Args:
            ctx: See description of argument ``ctx`` of method
                :meth:`LinearFunction.forward`.
            grad_A: The backpropagated error :math:`\delta_A`.

        Returns:
            The derivative of the loss with respect to the input activations
            :math:`Z`, i.e., :math:`\delta_Z`.

            .. note::
                Gradients for input tensors are only computed if their keyword
                ``requires_grad`` is set to ``True``, otherwise ``None`` is
                returned for the corresponding Tensor.
        """
        Z, = ctx.saved_tensors

        grad_Z = None

        # We only need to compute gradients for tensors that are flagged to
        # require gradients!
        if ctx.needs_input_grad[0]:
            A = 1 / (1 + torch.exp(-Z))
            grad_Z = grad_A * (A * (1 - A))

        return grad_Z

sigmoid_function = SigmoidFunction.apply
"""An alias for using class :class:`SigmoidFunction`."""

class MSELossFunction(Function):
    r"""Implementation of a mean-squared-error (MSE) loss.

    Assuming a set of input activations in form of a matrix
    :math:`A \in \mathbb{R}^{B \times M}`, where :math:`B` denotes the size of
    the mini-batch and :math:`M` the size of each sample. Additionally, this
    ``Function`` requires a set of target activations
    :math:`T \in \mathbb{R}^{B \times M}`.

    This method computes

    .. math::

        \mathcal{L}(A, T) = \frac{1}{B} \sum_{b = 1}^B \frac{1}{2} \lVert
        A_{b,:} - T_{b,:} \rVert^2

    where :math:`A_{b,:}` denotes the :math:`b`-th row of matrix :math:`A`.
    """
    @staticmethod
    def forward(ctx, A, T):
        r"""Computes the MSE loss between activations :math:`A` and targets
        :math:`T`: :math:`\mathcal{L}(A, T)`.

        Args:
            ctx: See description of argument ``ctx`` of method
                :meth:`LinearFunction.forward`.
            A: The input activations, i.e., the matrix :math:`A`.
            T: The target activations, i.e., the matrix :math:`T`.

                .. note::
                    Normally, targets will be constant values, that we do not
                    wish to backpropagate through (i.e., the keyword
                    ``requires_grad`` will be set to ``False``). For reasons of
                    generality, this ``Function`` will also allow the
                    backpropagation through targets, the keyword
                    ``requires_grad`` of parameter ``T`` is set to ``True``.

        Returns:
            A scalar loss value ``L`` denoting the computed MSE value.
        """
        ctx.save_for_backward(A, T)

        B = A.shape[0]

        L = 1/(2*B) * ((A - T)**2).sum()

        return L

    @staticmethod
    def backward(ctx, grad_L):
        r"""Backpropagate gradients through MSE loss.

        The input ``grad_L`` is the derivative :math:`\delta_L` of the final
        global scalar loss with respect to the scalar output of the
        :meth:`forward` method.
        If the :meth:`forward` is used to compute the final loss from which the
        backpropagation algorithm starts, then PyTorch will pass the value ``1``
        for the parameter ``grad_L``.

        As shown in the tutorial, the partial derivative :math:`\delta_A` can be
        computed via

        .. math::

            \delta_A = \frac{1}{B} (A - T)

        Hence, :math:`\delta_T` can be computed analoguously.

        Args:
            ctx: See description of argument ``ctx`` of method
                :meth:`LinearFunction.forward`.
            grad_A: The backpropagated error :math:`\delta_A`.

        Returns:
            (tuple): Tuple containing:

            - **grad_A**: The derivative of the loss with respect to the input
              activations, i.e., :math:`\delta_A`.
            - **grad_T**: The derivative of the loss with respect to the target
              activations, i.e., :math:`\delta_T`.

            .. note::
                Gradients for input tensors are only computed if their keyword
                ``requires_grad`` is set to ``True``, otherwise ``None`` is
                returned for the corresponding Tensor.
        """
        A, T = ctx.saved_tensors

        grad_A = None
        grad_T = None

        B = A.shape[0]

        # We only need to compute gradients for tensors that are flagged to
        # require gradients!
        if ctx.needs_input_grad[0]:
            grad_A = 1/B*(A - T)

        if ctx.needs_input_grad[1]:
            grad_T = 1/B*(T - A)

        return grad_A, grad_T

mse_loss = MSELossFunction.apply
"""An alias for using class :class:`MSELossFunction`."""

if __name__ == '__main__':
    pass


