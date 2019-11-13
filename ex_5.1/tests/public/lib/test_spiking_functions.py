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

import lib.spiking_functions as sf

class SpikingFunctionsTestCase(unittest.TestCase):
    """A set of public test cases for module :mod:`lib.spiking_functions`.

    Here, we assess whether the ``Functions`` implemented in the module 
    :mod:`lib.spiking_functions` are correctly implemented.
    """

    def test_spike_function(self):
        """Testing class 
        :class:`lib.spiking_functions.spike_function`."""

        a =      torch.from_numpy(np.array([[2., 0.1, -1], [-2., 1., 1.]]))
        target = torch.from_numpy(np.array([[1.,  1., 0.], [0.,  1., 1.]]))
        
        # Compute spikes with naive spike implementation
        spikes_a = sf.spike_function(a)
        self.assertTupleEqual(target.numpy().shape, spikes_a.numpy().shape, \
                       'Spike function has a dimensionality issue.')
        self.assertTrue(torch.all(torch.eq(spikes_a, target)),
                       'Spike function not correctly computed.')


    def test_loss_on_voltage(self):
        """Testing class :class:`lib.spiking_functions.loss_on_voltage`."""

        U = torch.tensor([[[1., 2., 0., -1.],[-1., -2., 1., 1.]], \
            [[0., 5., 0., 1.],[2., -2., 1., -1.]],\
            [[0., 0., 1., 3.],[2., 1., -1., 2.]]])
        T = torch.tensor([0, 3, 1])
        target_loss = 2.7734554

        our_loss = sf.loss_on_voltage(U, T)
        self.assertAlmostEqual(our_loss.item(), target_loss, 5,
                           'Loss on voltage value not correctly computed.')

    def test_accuracy_on_voltage(self):
        """Testing class :class:`lib.spiking_functions.accuracy_on_voltage`."""

        U = torch.tensor([[[1., 2., 0., -1.],[-1., -2., 1., 1.]], \
            [[0., 5., 0., 1.],[2., -2., 1., -1.]],\
            [[0., 0., 1., 3.],[2., 1., -1., 2.]]])
        T = torch.tensor([0, 1, 3])
        target_accuracy = 0.6666667

        our_accuracy = sf.accuracy_on_voltage(U, T)
        self.assertAlmostEqual(our_accuracy.item(), target_accuracy, 5,
                           'Accuracy on voltage value not correctly computed.')


if __name__ == '__main__':
    unittest.main()