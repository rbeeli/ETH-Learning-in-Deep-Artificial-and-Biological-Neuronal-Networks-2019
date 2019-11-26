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
r"""
A spiking layer module that maintains its own parameters (:mod:`lib.spiking_layer`)
-----------------------------------------------------------------------------------

The module :mod:`lib.spiking_layer` contains the implementation of a single
spiking layer. The goal is to utilize the custom
``Functions`` implemented in module :mod:`lib.spiking_functions` 
and to provide a wrapper that takes care of managing the parameters
(:math:`W`) of such a layer. The layers defined here will then be used in
:mod:`lib.snn` to define a multi-layer spiking network.

In biological networks, the electrical activity from a pre-synaptic spike leads
to changes in the membrane potential of a post-synaptic neuron.
In nature, this is a process involving many ions and channels.
Here we will use a simplified model: the leaky integrate-and-fire
model for spiking neurons. Such models are composed of 1) a description of the
dynamics of the membrane potential, and 2) a mechanism for triggering spikes.

In our implementation, the dynamics of the membrane potential, together with the
dynamics of the current and spiking variables, are updated at each timestep
based on a discrete implementation of a set of differential equations described
below.

The ODEs that you will implement sequentially update the membrane potentials
:math:`U`, the auxiliary variable for the alpha function :math:`H`, and the
current :math:`I`. The equation for the membrane potential of neuron :math:`i`
is:

.. math::
    \frac{dU_i}{dt} =& - \frac{1}{\tau_{mem}} \left[ (U_i - U_{rest}) - R I_i
    \right] + S_i(t) \left( U_{rest} - U_{threshold} \right) \\
    :label: eq-U-ODE

Most commonly, post-synaptic currents resulting from spiking inputs from
pre-synaptic neurons are modeled as exponential decay functions, where a spike 
causes an instantaneous increase in the post-synaptic membrane potential, which
then decays exponentially with time i.e. :math:`u(t) = e^{-t}`, assuming the
pre-synaptic spike occurs at :math:`t=0`, and a resting potential of 0. 
A more biologically plausible model is an alpha-shaped post-synaptic current,
where the post-synaptic current following a pre-synaptic spike has a finite rise
time. In this case we would have :math:`u(t) = te^{-t}`. In this tutorial, we
ask you to implement alpha-shaped post-synaptic currents by filling the methods 
:meth:`lib.spiking_layer.update_H` and :meth:`lib.spiking_layer.update_I`.
These are based on the following equations:

.. math::
    \frac{dH_i}{dt} =& - \frac{1}{\tau_{rise}} H_i (t) + \sum_j W_{ij} S_j (t)
    :label: eq-H-ODE

.. math::
    \frac{dI_i}{dt} =& - \frac{1}{\tau_{syn}}  I_i (t) + H_i (t)
    :label: eq-I-ODE

All equations were derived during the tutorial session, and can be found in
the tutorial slides. Note, however, that only a discrete version for the case
of exponential-shaped post-synaptic currents was derived, and not for the
alpha-shaped case, which is the one you need here. Therefore you will need to
figure out how to turn the ODEs into code appropriately. 


.. autosummary::

    lib.spiking_layer.SpikingLayer
    lib.spiking_layer.SpikingLayer.update_U
    lib.spiking_layer.SpikingLayer.update_I
    lib.spiking_layer.SpikingLayer.update_H
    lib.spiking_layer.SpikingLayer.forward

.. _`Neftci et al. (2019)`:
    https://arxiv.org/pdf/1901.09948.pdf

"""

# Name:           Rino Beeli
# Student ID:     15-709-371 (UZH external)
# Email:          rbeeli@student.ethz.ch

import torch
import torch.nn as nn
import numpy as np

import lib.spiking_functions as sf

class SpikingLayer(nn.Module):
    r"""Implements a single spiking layer.

    The :class:`lib.SpikingLayer` class contains all the parameters and
    variables necessary to implement a single spiking layer. It will be a
    submodule of the ``torch.nn.Module`` instance named :class:`lib.snn.SNN`.
    You will use the class :class:`lib.SpikingLayer` to define your hidden
    layer and your output layer.

    Attributes:
        tau_mem (float): Membrane time constant.
        tau_syn (float): Synaptic time constant.
        tau_rise (float): Rise synaptic time constant.
        u_rest (float): Resting membrane potential.
        u_threshold (float): Firing threshold.
        R (float): Membrane resistance.
        gamma (float): Decay rate for the post-synaptic current :math:`I`.
        beta (float): Decay rate for the membrane potential :math:`U`.
        phi (float): Decay rate for the auxiliary current variable :math:`H`.
        weights (torch.nn.Parameter): The weight matrix :math:`W` of the layer.
        compute_spikes (func): Spike non-linearity function.


    Args:
        in_features (int): Size of the pre-synaptic layer.
        out_features (int): Size of the current layer.
        args (argparse.Namespace): The command-line arguments.

    """

    def __init__(self, in_features, out_features, args):
        nn.Module.__init__(self)

        # Store parameters of the layer.
        self.tau_mem     = args.tau_mem
        self.tau_syn     = args.tau_syn
        self.tau_rise    = args.tau_rise
        self.u_rest      = args.u_rest
        self.u_threshold = args.u_threshold
        self.R           = args.R

        # Calculate the decay scales.
        self.gamma = float(np.exp(-args.delta_t/args.tau_syn))
        self.beta  = float(np.exp(-args.delta_t/args.tau_mem))
        self.phi   = float(np.exp(-args.delta_t/args.tau_rise))

        # Define which spike nonlinearity function to use.
        self.compute_spikes = sf.spike_function

        # Create and initialize weights.
        self._weights = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        nn.init.normal_(self._weights, mean=0.0, std=args.weight_scale/np.sqrt(in_features))

    @property
    def weights(self):
        r"""Getter for read-only attribute :attr:`weights`."""
        return self._weights


    def update_U(self, U, I, S):
        r"""Updates the membrane potential.

        Updates the membrane potential given the current and past states
        of the network as specified in eq. :eq:`eq-U-ODE`.

        Args:
            U: The membrane potential :math:`U`.
            I: The post-synaptic current :math:`I`.
            S: The spiking activity :math:`S`.

        Returns:
            The updated membrane potential of the neurons in the layer.
        """
        return self.beta * ((U - self.u_rest) - self.R * I) - S * (self.u_threshold - self.u_rest)

    def update_H(self, H, inputs):
        r"""Updates the state of the auxiliary variable for alpha-shaped
        post-synaptic current.

        Implementation of eq. :eq:`eq-H-ODE`. Please note that the pre-synaptic
        inputs have already been multiplied by the weights at the beginning of
        the forward method. Therefore, you should only make sure you give the
        correct inputs argument to this function.

        Args:
            H: The auxiliary variable for the alpha-shaped post-synaptic
                currents :math:`H`.
            inputs: The inputs (weighted spikes) to the layer in the current 
                time step.

        Returns:
            The updated auxiliary current variable for the neurons in the layer.
        """
        return self.phi * H + inputs

    def update_I(self, I, H):
        r"""Updates the post-synaptic current.

        Updates the post-synaptic current given the current and past states
        of the network as specified in eq. :eq:`eq-I-ODE`.

        Args: 
            I: The post-synaptic current :math:`I`.
            H: The auxiliary variable for the alpha-shaped post-synaptic
                currents :math:`H`.

        Returns:
            The updated post-synaptic current of the neurons in the layer.
            
        """
        return self.gamma * I + H

    def forward(self, X):
        r"""Computes the output activation of a spiking layer.

        This method computes the membrane potential and spiking activity of the
        current layer across all time steps given the pre-synaptic spiking 
        activity. For this, the state of the layer is
        updated time step by time step; i.e. the post-synaptic current,
        membrane potential and spiking activity are computed in each time step.
        The states are updated based on the computational graph provided in 
        Figure 2 in `Neftci et al. (2019)`_, and when filling in the missing
        lines you should pay extra attention and make sure that the values
        you provide to the update methods belong to the right time step
        according to this computational graph.

        Note that since we deal with alpha-shaped post-synaptic currents here
        (and not exponential decay post-synaptic currents), the computational
        graph has an extra variable :math:`H` that is updated based on the 
        inputs in the previous time step, and its own value in the previous
        time step. :math:`H` in a given time step is then used to compute the 
        post-synaptic current :math:`I` in the following time step. Notice that 
        for this extra equation, the :class:`lib.SpikingLayer` class has an 
        attribute :math:`phi` that governs the decay rate of the variable 
        :math:`H`. For further discussion see :mod:`lib.spiking_layer`.

        Args:
            X: The spiking activity of the previous layer.

        Returns: 
            (tuple): Tuple containing:   

            - **U**: The membrane potential for all neurons of the layer in all 
              time steps.
            - **S**: The spiking activity of all neurons of the layer in all 
              time steps.
        """

        device = X.device
        dtype = torch.float

        num_h = self.weights.shape[0]  # number of neurons in the layer
        t_max = X.shape[1]
        batch_size = X.shape[0]

        # Compute the total input to the layer in all time steps
        inputs = torch.einsum("abc,cd->abd", [X, self.weights.t()])

        # Create lists to store the network states in each time step
        U = [[]]*t_max  # membrane potential
        S = [[]]*t_max  # spiking activity
        H = [[]]*t_max  # auxiliary current state
        I = [[]]*t_max  # post-synaptic current

        # Initialize states
        U[0] = self.u_rest * torch.ones((batch_size, num_h), device=device, dtype=dtype)
        S[0] = torch.zeros_like(U[0])
        H[0] = torch.zeros_like(U[0])
        I[0] = torch.zeros_like(U[0])

        # Compute hidden layer state at each time step
        for t in range(1, t_max):
            # Compute the post-synaptic current
            H[t] = self.update_H(H[t-1], inputs[:, t-1, :])
            I[t] = self.update_I(I[t-1], H[t-1])

            # Compute the membrane potential
            U[t] = self.update_U(U[t-1], I[t-1], S[t-1])

            # Compute the spiking activity
            S[t] = self.compute_spikes(U[t] - self.u_threshold)

        U = torch.stack(U, dim=1)
        S = torch.stack(S, dim=1)

        return U, S


if __name__ == '__main__':
    pass
