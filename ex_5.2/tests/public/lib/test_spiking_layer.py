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
Public test cases for module :mod:`lib.spiking_layer`
----------------------------------------------------------
"""
import numpy as np
import torch
from torch import nn
import unittest
from torch.autograd import Function
import random

import lib.spiking_layer as sl

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

args = Namespace(R=0.5, tau_mem=0.010, tau_syn=0.012, tau_rise=0.001, 
                    u_rest=-0.050, u_threshold=0.5, weight_scale=0.2,
                    delta_t=0.001)

class SpikingLayerTestCase(unittest.TestCase):
    """A set of public test cases for module :mod:`lib.spiking_layer`.

    Here, we assess whether the ``Functions`` implemented in the module 
    :mod:`lib.spiking_layer` are correctly implemented.
    """

    def setUp(self):
        # Ensure reproducibility.
        rand = np.random.RandomState(42)

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        random.seed(42)

        n_in = 5
        n_h = 3
        batch_size = 4
        num_steps = 10

        # Data
        self.U = torch.rand(batch_size, n_h)
        self.I = torch.rand(batch_size, n_h)
        self.H = torch.rand(batch_size, n_h)
        self.S = torch.rand(batch_size, n_h)
        self.S[self.S >= 0.5] = 1.
        self.S[self.S <  0.5] = 0.

        self.inputs = torch.rand(batch_size, n_h)
        self.X = torch.rand(batch_size, num_steps, n_in)

        self.layer = sl.SpikingLayer(n_in, n_h, args)

    def test_update_U(self):
        """Testing class :class:`lib.spiking_layer.update_U`."""

        target_U = np.array([[-0.09978271,  0.06632668, -0.49361348],
                               [ 0.71898687, -0.5520569 ,  0.32930937],
                               [-0.39320755, -0.07051149,  0.22450066],
                               [-0.03392766,  0.20656866, -0.34393513]])

        our_U = self.layer.update_U(self.U, self.I, self.S).numpy()
        diff = np.abs((target_U - our_U).mean().item())

        self.assertTupleEqual(our_U.shape, target_U.shape, \
                        'U update has a dimensionality issue.')
        self.assertLess(diff, 1e-4, 'U update not correctly computed.')

    def test_update_I(self):
        """Testing class :class:`lib.spiking_layer.update_I`."""

        target_I = np.array([[0.9052, 0.7918, 1.0407],
                            [0.5944, 1.3618, 0.5342],
                            [1.1968, 0.6525, 1.1341],
                            [0.9893, 0.6108, 1.5742]])

        our_I = self.layer.update_I(self.I, self.H).numpy()
        diff = np.abs((target_I - our_I).mean().item())

        self.assertTupleEqual(our_I.shape, target_I.shape, \
                        'I update has a dimensionality issue.')
        self.assertLess(diff, 1e-4, 'I update not correctly computed.')

    def test_update_H(self):
        """Testing class :class:`lib.spiking_layer.update_H`."""

        target_H = np.array([[0.7926607,  0.29438907, 0.13704556],
                             [0.38016158, 0.3177891,  0.9125357 ],
                             [0.99407303, 0.7347956,  0.98407686],
                             [0.70585287, 1.0155178,  0.44234848]])

        our_H = self.layer.update_H(self.H, self.inputs).numpy()
        diff = np.abs((target_H - our_H).mean().item())

        self.assertTupleEqual(our_H.shape, target_H.shape, \
                        'H update has a dimensionality issue.')
        self.assertLess(diff, 1e-4, 'H update not correctly computed.')


    def test_forward(self):
        """Testing class :class:`lib.spiking_layer.forward`."""
        
        self.layer.forward(self.X)

        target_Ulast = np.array([[-0.0500, -0.0500, -0.0500],
                                [ 0.0000,  0.0000,  0.0000],
                                [ 0.0452,  0.0452,  0.0452],
                                [ 0.0589,  0.1143,  0.0325],
                                [ 0.0375,  0.1927, -0.0594],
                                [ 0.0664,  0.3204, -0.2025],
                                [ 0.1512,  0.4719, -0.4024],
                                [ 0.2639,  0.6924, -0.6083],
                                [ 0.3758,  0.3671, -0.8335],
                                [ 0.4392,  0.6158, -1.0612]])
        target_Ulast_wrong = np.array([[-0.0500, -0.0500, -0.0500],
                                    [ 0.0000,  0.0000,  0.0000],
                                    [ 0.0452,  0.0452,  0.0452],
                                    [ 0.0589,  0.1143,  0.0325],
                                    [ 0.0375,  0.1927, -0.0594],
                                    [ 0.0664,  0.3204, -0.2025],
                                    [ 0.1512,  0.4719, -0.4024],
                                    [ 0.2639,  0.6924, -0.6083],
                                    [ 0.3758,  0.9171, -0.8335],
                                    [ 0.4392,  0.5634, -1.0612]])
        target_Slast = np.array([[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 0., 0.],
                                [0., 0., 0.],
                                [0., 0., 0.],
                                [0., 0., 0.],
                                [0., 0., 0.],
                                [0., 1., 0.],
                                [0., 0., 0.],
                                [0., 1., 0.]])

        our_U, our_S = self.layer.forward(self.X)
        our_U = our_U.detach().numpy()
        our_S = our_S.detach().numpy()
        diff_U = np.abs((target_Ulast - our_U[-1]).mean().item())
        diff_S = np.abs((target_Slast - our_S[-1]).mean().item())
        diff_U_wrong = np.abs((target_Ulast_wrong - our_U[-1]).mean().item())

        self.assertTupleEqual(our_U[0].shape, target_Ulast.shape, \
                'Forward function has a dimensionality issue with U.')
        self.assertTupleEqual(our_S[0].shape, target_Slast.shape, \
                'Forward function has a dimensionality issue with S.')
        self.assertEqual(len(our_U), self.X.shape[0], \
                'Forward function has an issue with the length of U list.')
        self.assertEqual(len(our_S), self.X.shape[0], \
                'Forward function has an issue with the length of S list.')
        self.assertGreater(diff_U_wrong, 1e-4, 'The time-step chosen for ' +
            'the inputs U to update_U is not correct.')
        self.assertLess(diff_U, 1e-4, 'Forward spiking layer pass not '+
            'correctly computed. Please make sure that you pass variables ' +
            'to your update_U and spike functions corresponding to the right ' +
            'time step. For this, you can refer to the computation flow ' +
            'diagram of the tutorial slides.')
        self.assertLess(diff_S, 1e-4, 'Forward spiking layer pass not '+
            'correctly computed. Please make sure that you pass variables ' +
            'to your update_S and spike functions corresponding to the right ' +
            'time step. For this, you can refer to the computation flow ' +
            'diagram of the tutorial slides.')

if __name__ == '__main__':
    unittest.main()