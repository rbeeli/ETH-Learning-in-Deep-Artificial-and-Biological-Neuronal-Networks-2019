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
Public test cases for module :mod:`main`
----------------------------------------
"""
import contextlib
import numpy as np
import os
import unittest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lib.mlp import MLP
from lib import utils
import main

class MainTestCase(unittest.TestCase):
    """A set of public test cases."""
    @unittest.skip('Test method not modified by students.')
    def test_test_function(self):
        """Testing function :func:`main.test`."""
        # Ensure reproducibility.
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        rand = np.random.RandomState(42)

        n_in = 5
        n_out = 5
        n_samples = 2

        x = rand.uniform(low=0, high=1, size=(n_samples, n_in))
        y = rand.uniform(low=0, high=1, size=(n_samples, n_out))

        data = utils.RegressionDataset(x, y)
        data_loader1 = DataLoader(data, batch_size=n_samples)
        assert(len(data_loader1) == 1)
        data_loader2 = DataLoader(data, batch_size=n_samples // 2)
        assert(len(data_loader2) > 1)

        device = torch.device("cpu")

        net = MLP(n_in=n_in, n_out=n_out)
        net.eval()

        with torch.no_grad():
            predictions = net.forward(data.inputs)

        # Avoid any console prints.
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                mse1_test = main.test(device, data_loader1, net)

        # See docstring of method `_pytorch_mse` in
        # `tests.public.lib.test_backprop_functions`.
        mse1 = F.mse_loss(predictions, data.outputs, reduction='none')
        mse1 = 0.5 * mse1.sum(dim=1).mean()

        self.assertAlmostEqual(mse1_test, float(mse1), 5,
            'Method "main.test" does not work correctly.')

        ### Check if `test` handles multiple batches correctly correctly.

        # Avoid any console prints.
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                mse2_test = main.test(device, data_loader2, net)

        self.assertAlmostEqual(mse2_test, float(mse1), 5,
            'Method "main.test" does not work correctly when iterating ' +
            'over multiple mini batches.')

if __name__ == '__main__':
    unittest.main()
