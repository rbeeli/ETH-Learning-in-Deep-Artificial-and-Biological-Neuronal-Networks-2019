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
A collection of helper functions
--------------------------------

The module :mod:`lib.utils` contains several general purpose utilities and
helper functions.

.. autosummary::

    lib.utils.regression_cubic_poly
    lib.utils.RegressionDataset
    lib.utils.plot_predictions
    lib.utils.sampleGaussian
    lib.utils.computeELBO
    lib.utils.computeKLD
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

def regression_cubic_poly(num_train=20, num_test=100, train_domain=(-3.5, 3.5),
                          test_domain=(-5, 5), rseed=7):
    r"""Generate a dataset for a 1D regression task with a cubic polynomial.

    The regression task modelled here is :math:`y = x^3 + \epsilon`,
    where :math:`\epsilon \sim \mathcal{N}(0, 9I)`.

    Args:
        num_train (int): Number of training samples.
        num_test (int): Number of test samples.
        train_domain (tuple): Input domain for training samples.
        test_domain (tuple): Input domain for training samples.
        rseed (int): To ensure reproducibility, the random seed for the data
            generation should be decoupled from the random seed of the
            simulation. Therefore, a new :class:`numpy.random.RandomState` is
            created for the purpose of generating the data.

    Returns:
        (tuple): Tuple containing:

        - **train_x**: Generated training inputs.
        - **test_x**: Generated test inputs.
        - **train_y**: Generated training outputs.
        - **test_y**: Generated test outputs.

        Data is returned in form of 2D arrays of class :class:`numpy.ndarray`.
    """
    rand = np.random.RandomState(rseed)

    train_x = rand.uniform(low=train_domain[0], high=train_domain[1],
                               size=(num_train, 1))
    test_x = np.linspace(start=test_domain[0], stop=test_domain[1],
                         num=num_test).reshape((num_test, 1))

    map_function = lambda x : (x**3)
    train_y = map_function(train_x) 
    test_y = map_function(test_x)

    # Add noise to training outputs.
    train_eps = rand.normal(loc=0.0, scale=3, size=(num_train, 1))
    train_y += train_eps

    return train_x, test_x, train_y, test_y

def sampleGaussian(mu, logvar):
    r"""Get one sample from a Gaussian using the reparametrization trick.
     
    Assume you parametrize a Gaussian with mu and var.

    Args:
        mu: Vector of numbers inticating the mean of the Gaussian.
        logvar: Vector of numbers inticating the logvar of the Gaussian.
    
    Returns:
        A vector (one sample) drawn from a Gaussian parametrized 
        by mu and logvar.
    """

    var = torch.exp(logvar)
    sample = mu + torch.sqrt(var) * torch.randn(mu.shape)

    return sample


def computeELBO(net, predictions, targets, device, args):
    r"""Computes the ELBO (Evidence lower bound).
     
    Computes the negative log likelihood (nll) and Kullback-Leibler divergence.

    Args:
        net: The neural network.
        predictions: Predictions from the neural network.
        targets: Tragets for the predictions of the neural network.
        device: The PyTorch device to be used.
        args (argparse.Namespace): The command-line arguments.
    Returns:
        Two scalar values: The negative log likelihood and the KL divergence.
    """

    kl = net.compute_kld(device)
    
    # HINT: be sure to appropriately scale each term in the ELBO. 
    # Be aware of the reduction='mean' in the computation of the mse loss below.
    # NOTE: We use a sample size of 1 during training and in args you
    # can get the information of the datasize size.
    N = args.num_train_samples
    B = 1

    # reduction='mean' equals factor 1/M, hence we can ignore it
    loss = N/B * 0.5*F.mse_loss(predictions, targets, reduction='mean')

    return loss, kl

def computeKLD(mean_a_flat, logvar_a_flat, device,
                                            mean_b_flat=0.0, logvar_b_flat=1.0):
    """Compute the kullback-leibler divergence between two Gaussians.

        Args:
            mean_a_flat: mean of the Gaussian a.
            logvar_a_flat: Log variance of the Gaussian a.
            mean_b_flat: mean of the Gaussian b.
            logvar_b_flat: Log variance of the Gaussian b. 
            device:         
        Returns:
            LKD between two gausian with given parameters.
    """

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114

    var_a_flat = torch.exp(logvar_a_flat)
    kl = -0.5 * torch.sum((1 + logvar_a_flat) - mean_a_flat**2 + var_a_flat)

    return kl

class RegressionDataset(Dataset):
    """A simple regression dataset.

    Args:
        inputs (numpy.ndarray): The input samples.
        outputs (numpy.ndarray): The output samples.
    """
    def __init__(self, inputs, outputs):
        assert(len(inputs.shape) == 2)
        assert(len(outputs.shape) == 2)
        assert(inputs.shape[0] == outputs.shape[0])

        self.inputs = torch.from_numpy(inputs).float()
        self.outputs = torch.from_numpy(outputs).float()

    def __len__(self):
        return int(self.inputs.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        batch_in = self.inputs[idx, :]
        batch_out = self.outputs[idx, :]

        return batch_in, batch_out

def plot_predictions(device, test_loader, trainings_loader, net, args):
    """Plot the predictions of 1D regression tasks.

    Args:
        (....): See docstring of function :func:`main.test`.
    """
    net.eval()

    test_data = test_loader.dataset

    assert(test_data.inputs.shape[1] == 1 and test_data.outputs.shape[1] == 1)

    test_inputs = test_data.inputs.detach().cpu().numpy()
    test_targets = test_data.outputs.detach().cpu().numpy()

    training_data = trainings_loader.dataset

    assert(training_data.inputs.shape[1] == 1 and \
                                           training_data.outputs.shape[1] == 1)

    train_inputs = training_data.inputs.detach().cpu().numpy()
    train_targets = training_data.outputs.detach().cpu().numpy()
    
    with torch.no_grad():
        # Note, for simplicity, we assume that the dataset is small and we don't
        # have t collect the predictions by iterating over mini-batches.
        test_predictions = net.forward(test_data.inputs, 
                                                args.bbb).detach().cpu().numpy()
        if args.bbb:   
            pred_list = []
            for i in range(args.weight_samples):
                test_predictions = net.forward(test_data.inputs, 
                                                args.bbb).detach().cpu().numpy()
                pred_list.append(test_predictions)
                error_bar = np.var(np.array(pred_list),0).flatten()
                test_predictions = np.mean(np.array(pred_list),0).flatten()
        else:
            error_bar = np.zeros_like(test_predictions).flatten()

    plt.figure(figsize=(10, 6))
    plt.title("Predictions in 1D regression task", size=20)

    plt.plot(test_inputs, test_targets, color='b', label='Target function',
             linestyle='dashed', linewidth=.5)

    plt.errorbar(test_inputs, test_predictions, error_bar,
                                                 color='r', label='Predictions')
    plt.scatter(train_inputs, train_targets, color='b',label='Trainings data')
    borders = np.concatenate([np.linspace(start=-100, stop=100, num=100),\
                                    np.linspace(start=-100, stop=100,num=100)])
    plt.scatter([3.5]*100+[-3.5]*100, borders, color='k', s=0.1,
                                          label='Out-of-training distribution ')

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


if __name__ == '__main__':
    pass


