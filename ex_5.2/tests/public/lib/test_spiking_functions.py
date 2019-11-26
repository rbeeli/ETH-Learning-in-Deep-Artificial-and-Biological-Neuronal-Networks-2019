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
"""
Public test cases for module :mod:`lib.spiking_functions`
----------------------------------------------------------
"""
import numpy as np
import torch
from torch import nn
import unittest
from torch.autograd import Function
import random

import lib.spiking_functions as sf

class SpikingFunctionsTestCase(unittest.TestCase):
    """A set of public test cases for module :mod:`lib.spiking_functions`.

    Here, we assess whether the ``Functions`` implemented in the module 
    :mod:`lib.spiking_functions` are correctly implemented.
    """

    def setUp(self):
        
        # Ensure reproducibility.
        rand = np.random.RandomState(42)

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        random.seed(42)

        n_h = [5, 10]
        batch_size = 4
        num_steps = 10

        # Data
        self.S = [(torch.randn((batch_size, num_steps, n_hi))>.5).float() \
                                                            for n_hi in n_h]

        self.U = [torch.randn((batch_size, num_steps, n_hi)) for n_hi in n_h]
        self.U_threshold = -0.070

    ######################## FUNCTIONS FROM WEEK 1 #############################

    def test_spike_function(self):
        """Testing class 
        :class:`lib.spiking_functions.spike_function`."""

        a = [[2., 0.1, -1], [-2., 1., 1.]]
        target = [[1.,  1., 0.], [0.,  1., 1.]]
        a =      torch.from_numpy(np.array(a)).float()
        target = torch.from_numpy(np.array(target)).float()
        
        # Compute spikes with naive spike implementation
        spikes_a = sf.spike_function(a)
        self.assertTupleEqual(target.numpy().shape, spikes_a.numpy().shape, \
                       'Spike function has a dimensionality issue.')
        self.assertTrue(torch.all(torch.eq(spikes_a, target)),
                       'Spike function not correctly computed.')

    ######################## FUNCTIONS FROM WEEK 2 #############################

    def test_loss_on_spikes(self):
        """Testing class :class:`lib.spiking_functions.loss_on_spikes`."""

        T = torch.tensor([0, 3, 1, 2])
        target_loss = 1.89470160

        our_loss = sf.loss_on_spikes(self.S[0], T)
        self.assertAlmostEqual(our_loss.item(), target_loss, 5,
                           'Loss on spikes value not correctly computed.')

    def test_accuracy_on_spikes(self):
        """Testing class :class:`lib.spiking_functions.accuracy_on_spikes`."""

        T = torch.tensor([1, 3, 1, 2])
        target_accuracy = 0.25

        our_accuracy = sf.accuracy_on_spikes(self.S[0], T)
        self.assertAlmostEqual(our_accuracy.item(), target_accuracy, 5,
                           'Accuracy on spikes value not correctly computed.')


    def test_spike_regularizer(self):
        """Testing class :class:`lib.spiking_functions.spike_regularizer`."""

        target_spike_reg = 607.7000122
        our_spike_reg = sf.spike_regularizer(self.S)

        diff_wrong_spike_reg_1 = np.abs(232.50000000 - our_spike_reg.item()) #^2
        diff_wrong_spike_reg_2 = np.abs(205.00000000 - our_spike_reg.item()) #L2
        diff_wrong_spike_reg_3 = np.abs(402.70001220 - our_spike_reg.item()) #L1

        self.assertTupleEqual(our_spike_reg.numpy().shape, (), \
                        'Spike regularizer should output a scalar.')
        self.assertGreater(diff_wrong_spike_reg_1, 1e-4, 
                        'Spike regularizer wrongly implemented: please ' +
                        'make sure your term in the L2 loss has been squared.')
        self.assertGreater(diff_wrong_spike_reg_2, 1e-4, 
                        'Spike regularizer wrongly implemented: please ' +
                        'make sure you add the L2 loss.')
        self.assertGreater(diff_wrong_spike_reg_3, 1e-4, 
                        'Spike regularizer wrongly implemented: please ' +
                        'make sure you add the L1 loss.')
        self.assertAlmostEqual(our_spike_reg.item(), target_spike_reg, 5,
                       'Spike regularizer not correctly computed.')

    def test_derivative_logistic(self):
        """Testing class :class:`lib.spiking_functions.derivative_logistic`."""

        our_derivative = sf.derivative_logistic(self.U[0] - self.U_threshold)
        target_derivative0 = torch.tensor(
                             [[0.1758, 0.2465, 0.2169, 0.2184, 0.1929],
                             [0.2284, 0.2454, 0.2499, 0.2485, 0.2355],
                             [0.2397, 0.0626, 0.2488, 0.2104, 0.2384],
                             [0.2159, 0.1817, 0.0925, 0.2015, 0.2405],
                             [0.2411, 0.2500, 0.2492, 0.2203, 0.2100],
                             [0.2355, 0.2445, 0.2217, 0.2495, 0.2129],
                             [0.2465, 0.2244, 0.2078, 0.2463, 0.1207],
                             [0.2250, 0.1399, 0.2087, 0.1565, 0.2467],
                             [0.2283, 0.2096, 0.1857, 0.2435, 0.1326],
                             [0.2189, 0.2487, 0.1030, 0.1408, 0.2493]])

        diff_derivative = np.abs((target_derivative0 - our_derivative[0]).mean().item())
        self.assertLess(diff_derivative, 1e-4, 'Derivative of the logistic ' +
                         'function not correctly implemented. Please make ' +
                         'sure both your derivation and implementations are ' +
                         'correct.')


if __name__ == '__main__':
    unittest.main()
