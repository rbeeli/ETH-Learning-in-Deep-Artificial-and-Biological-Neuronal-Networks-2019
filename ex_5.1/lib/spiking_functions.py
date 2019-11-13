#!/usr/bin/env python3
# Copyright 2019 Maria Cervera
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
r"""Implementing, training, and evaluating a spiking neural network (:mod:`lib.spiking_functions`)
--------------------------------------------------------------------------------------------------

The module :mod:`lib.spiking_functions` contains custom functions that should
be used for running, training and evaluating spiking networks. Specifically,
you must implement the spike nonlinearity function, as well as the functions
computing the loss and the accuracy on the membrane potential of the output
neurons. 

.. autosummary::
    lib.spiking_functions.spike_function
    lib.spiking_functions.loss_on_voltage
    lib.spiking_functions.accuracy_on_voltage

.. math:
    \usepackage{dsfont}

"""

# Please fill out identification details in
#   lib/spiking_layer.py
#   lib/spiking_functions.py
#   theory_question.txt

# Student name :
# Student ID   :
# Email address:


import torch
import torch.nn as nn
from torch.autograd import Function

cross_entropy_loss = nn.CrossEntropyLoss()

def spike_function(D):
    r"""Outputs a spike when the input is greater than zero.

    This function takes :math:`D = ( U - U_{threshold} )` as input,
    which is the amount by which the membrane potential of neurons is above the
    membrane threshold :math:`U_{threshold} \in \mathbb{R}`. There are :math:`M`
    neurons in a layer and minibatch size is :math:`B`, hence
    :math:`D \in \mathbb{R}^{B \times M}`.

    This function computes the spiking nonlinearity, which should
    produce a spike when a neuron's membrane potential exceeds or is equal
    to the membrane threshold potential i.e. when :math:`U_i - U_{threshold}
    \geq 0`.

    The spiking nonlinearity we use here is the simple Heaviside step function,
    :math:`\Theta (\cdot)`, defined as

    .. math::
        \Theta(x) :=
        \begin{cases}
            0, & x < 0 \\
            1, & x \geq 0
        \end{cases}
        :label: eq-heaviside

    You must code the :meth:`spike_function` method to
    take :math:`D = ( U - U_{threshold} ) \in \mathbb{R}^{B \times M}` as 
    input and compute :math:`\Theta(D)` elementwise for each entry in the 
    matrix.

    Args:
        D: A matrix of shape :math:`B \times M` representing
            :math:`U - U_{threshold}`, the difference between the membrane
            potential of each of the :math:`M` neurons in each of the
            :math:`B` images of the mini-batch.

    Returns:
        The output spikes, obtained by applying :math:`\Theta (\cdot)`
        (defined in eq. :eq:`eq-heaviside`) elementwise to D.

    """
    raise NotImplementedError('TODO implement')
    # S = ...

    # return S


def loss_on_voltage(U, T):
    r"""Computes cross entropy loss on the maximum membrane voltage of the 
    output units.

    Specifically, this function takes a set of output voltages in form of a 
    matrix :math:`U \in \mathbb{R}^{B \times t_{max} \times M}`,
    where :math:`B` denotes the size of
    the mini-batch, :math:`t_{max}` the number of timesteps during which each
    mini-batch is presented, and :math:`M` the number of output units. 
    Additionally, it takes a set of target labels 
    :math:`T \in \mathbb{N}^{B}`, indicating the actual class of each image 
    :math:`b` in the current mini-batch.

    This function finds the maximum membrane voltage for each output unit and 
    for each element of the mini-batch and calculates the mean of the cross 
    entropies.

    The calculation is as follows:

    Letting :math:`Q_{b,i} = \max_t S_{b,:,i}` be the maximum membrane potential
    across all timesteps for each output neuron :math:`i` for each image
    :math:`b`, we calculate the cross entropy loss:

    .. math::
        CELoss(Q, T) = \frac{1}{B} \sum_{b=1}^{B} - Q_{b, T_b} +
         \log \left( \sum_j \exp ( Q_{b, j} ) \right)

    You may wish to refer to the pytorch documentation for its native
    :class:`torch.nn.CrossEntropyLoss` class and use it.

    Args:
        U: The output voltage, i.e., the matrix :math:`U`.
        T: The target class, i.e., the matrix :math:`T`.

    Returns:
        (float): The cross entropy loss for the maximum membrane potentials.
    """
    raise NotImplementedError('TODO implement')
    # Q ... 

    # return ...


def accuracy_on_voltage(U, T):
    r"""Computes the classification accuracy of the spiking network based on 
    the maximum voltage of the output units.

    Takes a set of output voltages in form of a matrix
    :math:`U \in \mathbb{R}^{B \times t_{max} \times M}`,
    where :math:`B` denotes the size of
    the mini-batch, :math:`t_{max}` the number of timesteps during which each
    mini-batch is presented, and :math:`M` the number of output units. 
    Additionally, this ``Function`` requires a set of targets
    :math:`T \in \mathbb{N}^{B}`, indicating the correct classes of the current
    mini-batch.

    Using these two arguments, it finds the output neurons that have the highest
    membrane voltage for each image, and compares these with the target labels 
    to compute the accuracy.

    Letting :math:`Q_{b,i} = \max_t S_{b,:,i}` be the maximum membrane potential
    across all timesteps for each output neuron :math:`i` for each image
    :math:`b`,

    .. math::
        Accuracy = \frac{1}{B} \sum_{b=1}^{B} 1[ \arg\max_{i} Q_{b,:} = T_b ]

    where :math:`1[\cdot]` is the indicator function.

    Args:
        U: The output voltages, i.e., the matrix :math:`U`.
        T: The target classes, i.e., the vector :math:`T`.

    Returns:
        (float): The classification accuracy of the current batch.

    """
    raise NotImplementedError('TODO implement')
    # Q ...
 
    # return ...
