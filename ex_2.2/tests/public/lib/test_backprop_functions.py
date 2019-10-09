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
Public test cases for module :mod:`lib.backprop_functions`
----------------------------------------------------------
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import unittest

import lib.backprop_functions as bf

class BackpropFunctionsTestCase(unittest.TestCase):
    """A set of public test cases for module :mod:`lib.backprop_functions`.

    Here, we assess whether the ``forward`` and ``backward`` methods for the
    ``Functions`` implemented in the module :mod:`lib.backprop_functions` are
    correctly implemented.
    """
    def setUp(self):
        # Ensure reproducibility.
        rand = np.random.RandomState(42)

        # Data
        A1_np = rand.rand(32, 5)
        A2_np = rand.rand(32, 10)
        Z_np = rand.rand(32, 10)
        T_np = rand.rand(32, 10)

        # Parameters
        W_np = rand.rand(10, 5)
        b_np = rand.rand(10)

        self.A1 = nn.Parameter(torch.from_numpy(A1_np), requires_grad=True)
        self.A2 = nn.Parameter(torch.from_numpy(A2_np), requires_grad=True)
        self.Z = nn.Parameter(torch.from_numpy(Z_np), requires_grad=True)
        self.T = nn.Parameter(torch.from_numpy(T_np), requires_grad=False)

        self.W = nn.Parameter(torch.from_numpy(W_np), requires_grad=True)
        self.b = nn.Parameter(torch.from_numpy(b_np), requires_grad=True)

    def test_mse_loss(self):
        """Testing class :class:`lib.backprop_functions.MSELossFunction`."""
        # Simple test of the forward method only.
        a = torch.from_numpy(np.array([[2., 3.]]))
        t = torch.from_numpy(np.array([[1., 1.]]))
        target_mse = 0.5 * (1**2 + 2**2) # 2.5
        our_mse = bf.mse_loss(a, t).detach().numpy()
        self.assertAlmostEqual(our_mse, target_mse, 5,
                               'MSE value not correctly computed.')

        # Compare forward and backward method to PyTorch its implementation.
        mse_ours = bf.mse_loss(self.A2, self.T)
        mse_torch = self._pytorch_mse(self.A2, self.T)
        self.assertAlmostEqual(mse_ours.detach().numpy(),
                               mse_torch.detach().numpy(), 5,
                               'MSE value not correctly computed.')

        # Check if gradient computation is correct (i.e., backward path)
        if self.A2.grad is not None:
            self.A2.grad.zero_()
        mse_ours.backward()
        our_grad_A = self.A2.grad.clone()

        self.A2.grad.zero_()
        mse_torch.backward()
        torch_grad_A = self.A2.grad.clone()

        grad_error = torch.sum((our_grad_A - torch_grad_A)**2).detach().numpy()
        self.assertAlmostEqual(grad_error, 0., 5, 'MSE loss gradients not ' +
                               'correctly computed.')

    def test_mse_loss_targets(self):
        """Testing if gradients of targets in class
        :class:`lib.backprop_functions.MSELossFunction` are correctly
        implemented.
        """
        # Ensure that the gradients with respect to the targets are also
        # correctly computed.
        mse_ours = bf.mse_loss(self.T, self.A2)
        mse_torch = self._pytorch_mse(self.T, self.A2)

        if self.A2.grad is not None:
            self.A2.grad.zero_()
        mse_ours.backward()
        our_grad_A = self.A2.grad.clone()

        self.A2.grad.zero_()
        mse_torch.backward()
        torch_grad_A = self.A2.grad.clone()

        grad_error = torch.sum((our_grad_A - torch_grad_A) ** 2).detach(). \
            numpy()
        self.assertAlmostEqual(grad_error, 0., 5,
            'MSE loss gradients for targets not correctly computed.')

    def test_sigmoid(self):
        """Testing class :class:`lib.backprop_functions.SigmoidFunction`."""
        sigma_ours = bf.sigmoid_function(self.Z)
        sigma_torch = torch.sigmoid(self.Z)
        sigma_error = torch.sum((sigma_ours - sigma_torch)**2).detach().numpy()
        self.assertAlmostEqual(sigma_error, 0, 5,
                               'Sigmoid values are incorrectly computed.')

        # Note, we need a scalar loss to use `backward()`.
        mse_ours = F.mse_loss(sigma_ours, self.T)
        mse_torch = F.mse_loss(sigma_torch, self.T)

        if self.Z.grad is not None:
            self.Z.grad.zero_()
        mse_ours.backward()
        our_grad_Z = self.Z.grad.clone()

        self.Z.grad.zero_()
        mse_torch.backward()
        torch_grad_Z = self.Z.grad.clone()

        grad_error = torch.sum((our_grad_Z - torch_grad_Z) ** 2).detach(). \
            numpy()
        self.assertAlmostEqual(grad_error, 0., 5,
                         'Sigmoid gradients not correctly computed.')

    def test_linear(self):
        """Testing class :class:`lib.backprop_functions.LinearFunction`."""
        Z_ours = bf.linear_function(self.A1, self.W, b=self.b)
        Z_torch = F.linear(self.A1, self.W, bias=self.b)

        Z_error = torch.sum((Z_ours - Z_torch)**2).detach().numpy()
        self.assertAlmostEqual(Z_error, 0, 5,
            'Linear layer output is incorrectly computed.')

        # Note, we need a scalar loss to use `backward()`.
        mse_ours = F.mse_loss(Z_ours, self.T)
        mse_torch = F.mse_loss(Z_torch, self.T)

        if self.A1.grad is not None:
            self.A1.grad.zero_()
        if self.W.grad is not None:
            self.W.grad.zero_()
        if self.b.grad is not None:
            self.b.grad.zero_()

        mse_ours.backward()

        our_grad_A = self.A1.grad.clone()
        our_grad_W = self.W.grad.clone()
        our_grad_b = self.b.grad.clone()

        self.A1.grad.zero_()
        self.W.grad.zero_()
        self.b.grad.zero_()

        mse_torch.backward()

        torch_grad_A = self.A1.grad.clone()
        torch_grad_W = self.W.grad.clone()
        torch_grad_b = self.b.grad.clone()

        grad_error_A = torch.sum((our_grad_A - torch_grad_A) ** 2).detach(). \
            numpy()
        grad_error_W = torch.sum((our_grad_W - torch_grad_W) ** 2).detach(). \
            numpy()
        grad_error_b = torch.sum((our_grad_b - torch_grad_b) ** 2).detach(). \
            numpy()

        self.assertAlmostEqual(grad_error_A, 0., 5,
            'Gradients with respects to inputs of linear layer are not ' +
            'correctly computed.')
        self.assertAlmostEqual(grad_error_W, 0., 5,
            'Gradients with respects to weight matrix of linear layer are ' +
            'not correctly computed.')
        self.assertAlmostEqual(grad_error_b, 0., 5,
            'Gradients with respects to bias vector of linear layer are not ' +
            'correctly computed.')

    @staticmethod
    def _pytorch_mse(A, T):
        """Helper method to compute MSE value using PyTorch its
        :func:`torch.nn.functional.mse_loss` function.

        Note, that the reduction methods implemented by this PyTorch method
        are mathematically not well justified but should effect loss
        optimization only marginally.

        This method simply replaces the implemented reduction methods by the
        mathematically sound reduction method.

        Args:
            (....): See docstring of method
                :meth:`lib.backprop_functions.MSELossFunction.forward`.

        Returns:
            MSE value.
        """
        mse_torch = F.mse_loss(A, T, reduction='none')
        mse_torch = 0.5 * mse_torch.sum(dim=1).mean()

        return mse_torch

if __name__ == '__main__':
    unittest.main()